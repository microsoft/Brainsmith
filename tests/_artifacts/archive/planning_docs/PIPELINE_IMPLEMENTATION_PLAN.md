# Staged Implementation Plan: 3-Stage Pipeline Integration

**Date**: 2025-10-31
**Objective**: Extend test infrastructure to support complete ONNX → Base → Backend flow
**Approach**: Leverage proven code, minimize new code, validate each stage

---

## Inventory: What Working Code Do We Have?

### 1. FINN's Proven Pattern (deps/finn/tests/fpgadataflow/test_fpgadataflow_addstreams.py)

**Lines 93-140: Complete flow we want to replicate**

```python
def test_fpgadataflow_addstreams(idts, ch, fold, exec_mode):
    # Stage 1: ONNX (line 104-114)
    model = make_addstreams_modelwrapper(ch, idts)
    y_expected = x1 + x2
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    # Stage 2: Base kernel (line 116-119)
    model = model.transform(to_hw.InferAddStreamsLayer())
    addstreams_node = getHWCustomOp(addstreams_node)
    addstreams_node.set_nodeattr("PE", pe)

    # Stage 3: Backend specialization (line 120)
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Stage 3: Execution preparation (line 122-133)
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP("xc7z020clg400-1", 5))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())

    # Stage 3: Backend execution (line 136-139)
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
```

**Key insight**: FINN already does this! We just need to make it the default in our framework.

### 2. Our Working PipelineRunner (tests/support/pipeline.py:39-141)

**What it does well:**
- ✅ Stage 1 → Stage 2: ONNX → Base kernel (lines 95-127)
- ✅ Configuration hooks (lines 129-135)
- ✅ KernelOp initialization (lines 124-127, 133-135)
- ✅ Node finding logic (lines 105-119)

**What it's missing:**
- ❌ Stage 2 → Stage 3: Base → Backend specialization
- ❌ Backend type selection (hls vs rtl)

### 3. Our Working Executors (tests/support/executors.py)

**CppSimExecutor (lines 147-265):**
- ✅ Checks for HLSBackend inheritance (lines 183-191)
- ✅ PrepareCppSim + CompileCppSim (lines 198-230)
- ✅ SetExecMode("cppsim") implicitly via execute_node (line 215)
- ✅ Execution via execute_node (lines 232-242)

**RTLSimExecutor (lines 267-400+):**
- ✅ Checks for HLSBackend/RTLBackend (lines 300-320)
- ✅ PrepareIP + HLSSynthIP + PrepareRTLSim (implementation in execute method)
- ✅ SetExecMode("rtlsim") handling
- ✅ Execution via execute_node

**Key insight**: Executors already handle preparation transforms! They just need backend-specialized nodes.

### 4. Our Working Test Framework (tests/frameworks/single_kernel_test.py)

**What it does well:**
- ✅ Clean test structure (6 inherited tests, lines 189-400)
- ✅ Uses PipelineRunner (lines 143-151)
- ✅ Uses Executors (lines 311-357, 392-399)
- ✅ Validates against golden (lines 157-185)

**What it's missing:**
- ❌ No way to specialize base kernel before passing to executor
- ❌ cppsim/rtlsim tests skip because op is base kernel (not backend)

---

## Problem Analysis

### Current Flow (BROKEN for cppsim/rtlsim):
```
make_test_model()
  ↓
PipelineRunner.run() → AddStreams (base)
  ↓
CppSimExecutor.execute(AddStreams) → isinstance(op, HLSBackend)? NO → SKIP
```

### Desired Flow (WORKING):
```
make_test_model()
  ↓
PipelineRunner.run() → AddStreams (base)
  ↓
SpecializeLayers() → AddStreams_hls (backend)
  ↓
CppSimExecutor.execute(AddStreams_hls) → isinstance(op, HLSBackend)? YES → EXECUTE
```

### Missing Link:
**SpecializeLayers call between PipelineRunner and Executor**

---

## Design Principles

1. **Minimize New Code**: Reuse FINN transforms directly
2. **Compose, Don't Duplicate**: PipelineRunner + SpecializeLayers, not new class
3. **Backward Compatible**: Specialization is optional
4. **Single Responsibility**: Separation of concerns
   - PipelineRunner: ONNX → HW transformation
   - SpecializeLayers: Base → Backend transformation
   - Executors: Backend execution
5. **Follow FINN**: Match proven patterns exactly

---

## Staged Implementation Plan

### Stage 0: Spike - Validate Pattern (30 min)

**Goal**: Prove the pattern works with minimal code

**Action**: Create throwaway test script
```python
# tests/spike_backend_specialization.py
def test_spike_backend_flow():
    # Stage 2: Get base kernel (existing code)
    runner = PipelineRunner()
    op, model = runner.run(
        model_factory=make_addstreams_model,
        transform=InferAddStreamsLayer()
    )

    # Stage 3: Specialize (FINN code, direct call)
    from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # Find specialized node (FINN pattern)
    original_name = op.onnx_node.name
    specialized_node = None
    for node in model.graph.node:
        if node.name == original_name:
            specialized_node = node
            break

    specialized_op = getHWCustomOp(specialized_node, model)

    # Verify it worked
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    assert isinstance(specialized_op, HLSBackend)
    assert specialized_op.onnx_node.op_type == "AddStreams_hls"

    # Execute with existing executor
    executor = CppSimExecutor()
    outputs = executor.execute(specialized_op, model, inputs)
    # Should NOT skip!
```

**Validation**:
- Does SpecializeLayers work on our models?
- Can we find the specialized node?
- Does CppSimExecutor execute (not skip)?

**Decision Point**: If spike fails, diagnose why. If succeeds, proceed to Stage 1.

---

### Stage 1: Extract Helper Function (1 hour)

**Goal**: Extract node-finding logic into reusable helper

**Rationale**: FINN pattern of finding specialized node will be used multiple times

**Action**: Create `tests/support/backend_utils.py`

```python
"""Backend specialization utilities.

Minimal helpers for backend specialization, following FINN patterns exactly.
"""

from typing import Tuple
from finn.core.modelwrapper import ModelWrapper
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.util.basic import getHWCustomOp
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers


def specialize_to_backend(
    op: HWCustomOp,
    model: ModelWrapper,
    fpgapart: str
) -> Tuple[HWCustomOp, ModelWrapper]:
    """Specialize base kernel to backend variant.

    Applies SpecializeLayers and returns the specialized operator.
    Follows FINN test pattern from test_fpgadataflow_addstreams.py.

    Args:
        op: Base kernel operator (e.g., AddStreams)
        model: Model containing the operator
        fpgapart: FPGA part for specialization

    Returns:
        (specialized_op, model) where specialized_op has backend inheritance

    Example:
        # Before: AddStreams (base)
        base_op, model = pipeline_runner.run(...)

        # After: AddStreams_hls (backend)
        backend_op, model = specialize_to_backend(base_op, model, "xc7z020clg400-1")

        # Now cppsim/rtlsim work
        executor = CppSimExecutor()
        outputs = executor.execute(backend_op, model, inputs)
    """
    # Store original node name (SpecializeLayers preserves names)
    original_name = op.onnx_node.name

    # Apply FINN's SpecializeLayers transform
    model = model.transform(SpecializeLayers(fpgapart))

    # Find specialized node (FINN pattern from test_fpgadataflow_addstreams.py:142)
    specialized_node = None
    for node in model.graph.node:
        if node.name == original_name:
            specialized_node = node
            break

    if specialized_node is None:
        raise RuntimeError(
            f"Failed to find specialized node '{original_name}' after SpecializeLayers. "
            f"Available nodes: {[n.name for n in model.graph.node]}"
        )

    # Get specialized operator instance
    specialized_op = getHWCustomOp(specialized_node, model)

    return specialized_op, model
```

**Why separate file:**
- Single responsibility (specialization only)
- Reusable across test frameworks
- Easy to test independently
- Mirrors FINN's module structure

**Test**:
```python
# tests/test_backend_utils.py
def test_specialize_to_backend():
    """Validate specialize_to_backend helper."""
    runner = PipelineRunner()
    base_op, model = runner.run(
        model_factory=make_addstreams_model,
        transform=InferAddStreamsLayer()
    )

    # Should be base kernel
    assert base_op.onnx_node.op_type == "AddStreams"

    # Specialize
    backend_op, model = specialize_to_backend(base_op, model, "xc7z020clg400-1")

    # Should be backend
    assert backend_op.onnx_node.op_type == "AddStreams_hls"
    from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
    assert isinstance(backend_op, HLSBackend)
```

**Validation**: Helper works independently

---

### Stage 2: Integrate with SingleKernelTest (1 hour)

**Goal**: Add backend support to SingleKernelTest without breaking existing tests

**Action**: Update `tests/frameworks/single_kernel_test.py`

```python
# Add import
from tests.support.backend_utils import specialize_to_backend

class SingleKernelTest(KernelTestConfig):
    # ... existing code ...

    def run_inference_pipeline(self, to_backend: bool = False):
        """Run pipeline to Stage 2 (base) or Stage 3 (backend).

        Args:
            to_backend: If True and fpgapart configured, specialize to backend

        Returns:
            (op, model) at requested stage
        """
        # Stage 2: Base kernel (EXISTING CODE - NO CHANGES)
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_test_model,
            transform=self.get_kernel_inference_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model)
        )

        # Stage 3: Backend specialization (NEW - uses helper)
        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart is None:
                pytest.skip(
                    "Backend testing not enabled. "
                    "Override get_backend_fpgapart() to return FPGA part."
                )

            op, model = specialize_to_backend(op, model, fpgapart)

        return op, model

    # Update execution tests (MINIMAL CHANGES)
    def test_cppsim_execution_vs_golden(self):
        """Test HLS C++ simulation matches golden reference."""
        # OLD: op, model = self.run_inference_pipeline()
        # NEW: op, model = self.run_inference_pipeline(to_backend=True)
        op, model = self.run_inference_pipeline(to_backend=True)

        # Rest unchanged
        np.random.seed(42)
        inputs = make_execution_context(model, op)
        golden_outputs = self.compute_golden_reference(inputs)

        executor = CppSimExecutor()
        actual_outputs = executor.execute(op, model, inputs)

        tolerance = self.get_tolerance_cppsim()
        self.validate_against_golden(
            actual_outputs, golden_outputs, "HLS simulation (cppsim)", tolerance
        )

    def test_rtlsim_execution_vs_golden(self):
        """Test RTL simulation matches golden reference."""
        op, model = self.run_inference_pipeline(to_backend=True)  # Same change
        # ... rest unchanged
```

**Why this works:**
- `to_backend=False` → existing behavior (base kernel, Python execution)
- `to_backend=True` → new behavior (backend, cppsim/rtlsim)
- Backward compatible (default `to_backend=False`)
- Minimal code changes (1 parameter, 1 helper call)

**Test**:
```bash
# Existing tests should still pass (base kernel, Python only)
pytest tests/frameworks/test_single_kernel_base.py -v

# Backend tests should now work (if fpgapart configured)
pytest tests/frameworks/test_single_kernel_with_backend.py -v -m cppsim
```

**Validation**: Existing tests unchanged, backend tests work if fpgapart provided

---

### Stage 3: Add Backend Configuration to KernelTestConfig (30 min)

**Goal**: Provide hook for tests to opt-in to backend testing

**Action**: Update `tests/frameworks/kernel_test_base.py`

```python
class KernelTestConfig(ABC):
    # ... existing methods ...

    # NEW: Backend configuration hooks
    def get_backend_fpgapart(self) -> Optional[str]:
        """Return FPGA part for backend specialization.

        Override to enable cppsim/rtlsim testing.

        Returns:
            None: Python execution only (default, backward compatible)
            str: FPGA part for SpecializeLayers (enables backend tests)

        Example:
            def get_backend_fpgapart(self):
                return "xc7z020clg400-1"  # Zynq-7000
                # return "xczu7ev-ffvc1156-2-e"  # Zynq UltraScale+
        """
        return None

    def get_backend_type(self) -> str:
        """Return backend type preference.

        Returns:
            Backend type: "hls" or "rtl"

        Default: "hls" (most common)

        Note: SpecializeLayers determines actual backend based on:
        - preferred_impl_style nodeattr
        - Available backend implementations
        - FPGA part capabilities

        This method is informational only; SpecializeLayers makes final decision.
        """
        return "hls"
```

**Why these hooks:**
- **Opt-in**: Default None = backward compatible
- **Minimal**: Just 2 simple methods
- **Flexible**: Each test chooses backend support
- **Clear**: Obvious what they do

**Validation**: Existing tests unaffected (return None by default)

---

### Stage 4: Create Example Test (1 hour)

**Goal**: Demonstrate the pattern with working AddStreams test

**Action**: Create `tests/examples/test_addstreams_backend.py`

```python
"""Example: Complete 3-stage pipeline testing for AddStreams.

Demonstrates:
- Stage 2: Base kernel (Python execution)
- Stage 3: Backend (cppsim/rtlsim execution)
"""

from tests.frameworks.single_kernel_test import SingleKernelTest
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer


class TestAddStreamsBackend(SingleKernelTest):
    """AddStreams with backend verification."""

    def make_test_model(self):
        # ... create Add ONNX model
        return model, "Add_0"

    def get_kernel_inference_transform(self):
        return InferAddStreamsLayer

    def compute_golden_reference(self, inputs):
        return {"output": inputs["input0"] + inputs["input1"]}

    def get_num_inputs(self):
        return 2

    def get_num_outputs(self):
        return 1

    # ENABLE BACKEND TESTING (single line!)
    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"

    # Inherits 6 tests:
    # ✓ test_pipeline_creates_hw_node (Stage 2)
    # ✓ test_shapes_preserved_through_pipeline (Stage 2)
    # ✓ test_datatypes_preserved_through_pipeline (Stage 2)
    # ✓ test_python_execution_vs_golden (Stage 2)
    # ✓ test_cppsim_execution_vs_golden (Stage 3) - NOW WORKS!
    # ✓ test_rtlsim_execution_vs_golden (Stage 3) - NOW WORKS!
```

**Test**:
```bash
# Python tests (Stage 2) - fast
pytest tests/examples/test_addstreams_backend.py::TestAddStreamsBackend::test_python_execution_vs_golden -v

# Backend tests (Stage 3) - slow, requires Vitis
pytest tests/examples/test_addstreams_backend.py::TestAddStreamsBackend::test_cppsim_execution_vs_golden -v -m cppsim

# Full suite
pytest tests/examples/test_addstreams_backend.py -v
```

**Expected Results**:
```
test_pipeline_creates_hw_node             PASSED
test_shapes_preserved_through_pipeline    PASSED
test_datatypes_preserved_through_pipeline PASSED
test_python_execution_vs_golden           PASSED
test_cppsim_execution_vs_golden           PASSED (or SKIP if no Vitis)
test_rtlsim_execution_vs_golden           PASSED (or SKIP if no Vivado)
```

**Validation**: All 6 tests run (no skips due to missing backend)

---

### Stage 5: Update DualKernelTest (1 hour)

**Goal**: Apply same pattern to DualKernelTest

**Action**: Update `tests/frameworks/dual_kernel_test.py`

```python
from tests.support.backend_utils import specialize_to_backend

class DualKernelTest(KernelTestConfig):
    # ... existing code ...

    def run_manual_pipeline(self, to_backend: bool = False):
        """Run manual (FINN) pipeline to Stage 2 or 3."""
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_test_model,
            transform=self.get_manual_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model)
        )

        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart:
                op, model = specialize_to_backend(op, model, fpgapart)

        return op, model

    def run_auto_pipeline(self, to_backend: bool = False):
        """Run auto (Brainsmith) pipeline to Stage 2 or 3."""
        runner = PipelineRunner()
        op, model = runner.run(
            model_factory=self.make_test_model,
            transform=self.get_auto_transform(),
            configure_fn=lambda op, model: self.configure_kernel_node(op, model)
        )

        if to_backend:
            fpgapart = self.get_backend_fpgapart()
            if fpgapart:
                op, model = specialize_to_backend(op, model, fpgapart)

        return op, model

    # Update execution tests
    def test_manual_vs_golden_cppsim(self, ...):
        op, model = self.run_manual_pipeline(to_backend=True)  # Changed
        # ... rest unchanged

    def test_auto_vs_golden_cppsim(self, ...):
        op, model = self.run_auto_pipeline(to_backend=True)  # Changed
        # ... rest unchanged
```

**Validation**: DualKernelTest gains backend support with minimal changes

---

### Stage 6: Update Existing Tests (30 min)

**Goal**: Enable backend testing for test_addstreams_v2.py

**Action**: Update `tests/dual_pipeline/test_addstreams_v2.py`

```python
class TestAddStreamsV2(DualKernelTest):
    # ... existing methods ...

    # ADD THIS ONE LINE
    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"

    # Immediately enables:
    # - test_manual_vs_golden_cppsim (was skipping, now runs)
    # - test_manual_vs_golden_rtlsim (was skipping, now runs)
    # - test_auto_vs_golden_cppsim (was skipping, now runs)
    # - test_auto_vs_golden_rtlsim (was skipping, now runs)
```

**Validation**:
```bash
# Before: 6 skipped (cppsim/rtlsim tests)
# After: 0 skipped (all tests run)
pytest tests/dual_pipeline/test_addstreams_v2.py -v
```

---

## Validation Strategy

### Stage-by-Stage:
1. **Stage 0**: Spike validates pattern works
2. **Stage 1**: Unit test for `backend_utils.py`
3. **Stage 2**: Existing SingleKernelTest tests still pass
4. **Stage 3**: No tests yet (just hooks)
5. **Stage 4**: Example test runs all 6 tests
6. **Stage 5**: DualKernelTest tests still pass
7. **Stage 6**: test_addstreams_v2.py gains 6 new passing tests

### Integration Test:
```bash
# Full test suite should have MORE passing tests, ZERO new failures
pytest tests/ -v --tb=short

# Expected changes:
# - Same number of passing tests for base kernel (Python)
# + More passing tests for backend (cppsim/rtlsim)
# - Fewer skipped tests (backend tests now run)
```

---

## Risk Mitigation

### Risk 1: SpecializeLayers breaks on our models
**Mitigation**: Stage 0 spike catches this early
**Fallback**: Debug why, fix model creation

### Risk 2: Executors don't work with specialized nodes
**Mitigation**: Stage 1 tests this explicitly
**Fallback**: Check executor backend detection logic

### Risk 3: Existing tests break
**Mitigation**: Backward compatible (optional `to_backend` parameter)
**Fallback**: Revert changes, rethink approach

### Risk 4: FINN transforms have side effects
**Mitigation**: Use FINN's exact pattern from working tests
**Fallback**: Consult FINN documentation/team

---

## Success Criteria

### Must Have:
- ✅ Stage 0 spike passes
- ✅ All existing tests still pass
- ✅ Example test runs 6/6 tests (not 3/6 with 3 skipped)
- ✅ Code changes < 200 lines total
- ✅ Zero breaking changes

### Nice to Have:
- ✅ test_addstreams_v2.py goes from 16/22 to 22/22 tests passing
- ✅ Documentation shows clear before/after
- ✅ Pattern easily replicable for other kernels

---

## Time Estimate

- Stage 0: 30 min (spike)
- Stage 1: 1 hour (helper + test)
- Stage 2: 1 hour (SingleKernelTest integration)
- Stage 3: 30 min (config hooks)
- Stage 4: 1 hour (example test)
- Stage 5: 1 hour (DualKernelTest)
- Stage 6: 30 min (enable in existing test)

**Total: 5.5 hours**

**Contingency: +1 hour for debugging**

**Realistic: 6-7 hours total**

---

## Next Step

**Begin Stage 0**: Create spike test to validate the pattern works with our models.

This validates assumptions before writing production code. If spike fails, we learn why early and cheaply.
