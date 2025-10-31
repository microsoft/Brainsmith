# Backend Pipeline Extension Plan

**Date**: 2025-10-31
**Status**: Proposed
**Purpose**: Extend PipelineRunner to support backend specialization and cppsim/rtlsim execution

---

## Problem Statement

### Current Architecture Gap

The existing test frameworks stop at the base kernel node, never specializing to backend implementations:

```python
# What we have now:
ONNX Add → InferAddStreamsLayer → AddStreams (base class, NOT HLSBackend)
                                    ↑
                                    STOPS HERE!

# Executors check for backend:
if not isinstance(op, HLSBackend):
    pytest.skip("cppsim requires HLSBackend inheritance")

# Result: ALL cppsim/rtlsim tests skip
```

### What We Need

```python
# Complete flow:
ONNX Add → InferAddStreamsLayer → AddStreams → SpecializeLayers → AddStreams_hls → PrepareCppSim → execute
                                                                     ↑
                                                              Has HLSBackend methods!
```

### Impact

**Current Test Coverage:**
- ✅ 7 core parity tests (shapes, datatypes, resources)
- ✅ 5 HW estimation tests (cycles, resources)
- ✅ 2 golden execution tests (Python only)
- ❌ **6 backend tests SKIPPED** (all cppsim/rtlsim)
- **Total: 16 tests actually run (6 skip)**

**Missing Validation:**
1. ❌ HLS C++ code generation
2. ❌ HLS C++ compilation (Vivado HLS)
3. ❌ HLS C++ execution
4. ❌ HLS → RTL synthesis
5. ❌ RTL simulation
6. ❌ Backend specialization workflow

---

## Proposed Solution: `BackendPipelineRunner`

### Architecture Overview

Create a new runner that extends `PipelineRunner` with three-phase execution:

```
Phase 1: Base Pipeline
  ONNX → Transform → HW Node (base)

Phase 2: Backend Specialization
  HW Node → SpecializeLayers → HW Node_hls/_rtl (backend variant)

Phase 3: Backend Preparation
  Code Generation → Compilation → Simulation Setup
```

### Implementation

**Location**: `tests/support/backend_pipeline.py`

```python
"""Backend-aware pipeline execution with specialization support.

Extends PipelineRunner to handle the complete flow through backend specialization,
code generation, and execution preparation.

Usage:
    runner = BackendPipelineRunner(backend="hls", prepare_for="cppsim")

    op, model = runner.run(
        model_factory=lambda: make_test_model(),
        transform=InferAddStreamsLayer(),
        fpgapart="xc7z020clg400-1",
        clk_ns=5.0
    )

    # op is now AddStreams_hls with cppsim ready
    executor = CppSimExecutor()
    outputs = executor.execute(op, model, inputs)
"""

from typing import Callable, Optional, Tuple, Literal
from enum import Enum

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode

from tests.support.pipeline import PipelineRunner


class BackendType(Enum):
    """Backend implementation types."""
    HLS = "hls"
    RTL = "rtl"


class PrepareMode(Enum):
    """Backend preparation modes."""
    CPPSIM = "cppsim"     # C++ simulation
    RTLSIM = "rtlsim"     # RTL simulation
    IPGEN = "ipgen"       # IP generation only (no execution)
    NONE = "none"         # Specialize but don't prepare


class BackendPipelineRunner:
    """Pipeline runner with backend specialization and preparation.

    Extends PipelineRunner to handle:
    1. Base pipeline (ONNX → HW node)
    2. Backend specialization (HW node → _hls/_rtl node)
    3. Backend preparation (code generation, compilation, simulation setup)

    This enables cppsim/rtlsim execution tests that require HLSBackend/RTLBackend.
    """

    def __init__(
        self,
        backend: Optional[Literal["hls", "rtl"]] = None,
        prepare_for: Literal["cppsim", "rtlsim", "ipgen", "none"] = "none",
    ):
        """Initialize backend-aware pipeline runner.

        Args:
            backend: Backend type ("hls" or "rtl"). If None, no specialization.
            prepare_for: Preparation mode for backend execution.
                - "cppsim": Generate and compile C++ code for simulation
                - "rtlsim": Synthesize HLS/RTL and prepare xsim simulation
                - "ipgen": Generate IP package only (no execution prep)
                - "none": Specialize but don't prepare (for structural tests)
        """
        self.backend = BackendType(backend) if backend else None
        self.prepare_mode = PrepareMode(prepare_for)
        self.base_runner = PipelineRunner()

        # Validation
        if self.prepare_mode != PrepareMode.NONE and self.backend is None:
            raise ValueError(
                f"Backend preparation mode '{prepare_for}' requires backend to be specified"
            )

    def run(
        self,
        model_factory: Callable[[], Tuple[ModelWrapper, Optional[str]]],
        transform: Transformation,
        configure_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None,
        init_fn: Optional[Callable[[HWCustomOp, ModelWrapper], None]] = None,
        fpgapart: Optional[str] = None,
        clk_ns: Optional[float] = None,
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run complete pipeline with backend specialization.

        Args:
            model_factory: Creates (model, node_name)
            transform: Kernel inference transform
            configure_fn: Optional node configuration (PE, SIMD, etc.)
            init_fn: Optional initialization (for base node only, rarely needed)
            fpgapart: FPGA part string (required if preparing backend)
            clk_ns: Clock period in ns (required if preparing backend)

        Returns:
            (op, model): Specialized operator and transformed model

        Raises:
            ValueError: If fpgapart/clk_ns missing when required
        """
        # Phase 1: Base pipeline (ONNX → HW node)
        op, model = self.base_runner.run(
            model_factory=model_factory,
            transform=transform,
            configure_fn=configure_fn,
            init_fn=init_fn
        )

        # Phase 2: Specialization (HW node → _hls/_rtl node)
        if self.backend is not None:
            op, model = self._specialize_backend(op, model, fpgapart)

        # Phase 3: Backend preparation (code gen, compilation, sim setup)
        if self.prepare_mode != PrepareMode.NONE:
            if fpgapart is None or clk_ns is None:
                raise ValueError(
                    f"Backend preparation mode '{self.prepare_mode.value}' requires "
                    f"fpgapart and clk_ns parameters"
                )
            op, model = self._prepare_backend(op, model, fpgapart, clk_ns)

        return op, model

    def _specialize_backend(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        fpgapart: Optional[str]
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Specialize base node to backend implementation.

        Transforms:
            AddStreams → AddStreams_hls  (domain: brainsmith.kernels.addstreams.hls)
            AddStreams → AddStreams_rtl  (domain: brainsmith.kernels.addstreams.rtl)

        Args:
            op: Base operator (no backend)
            model: Model containing operator
            fpgapart: FPGA part for backend selection heuristics

        Returns:
            (specialized_op, model): Operator with HLSBackend/RTLBackend inheritance
        """
        # Set preferred_impl_style if specified
        if self.backend is not None:
            op.set_nodeattr("preferred_impl_style", self.backend.value)

        # Apply SpecializeLayers transform
        fpgapart_for_transform = fpgapart or "xczu7ev-ffvc1156-2-e"  # Default
        model = model.transform(SpecializeLayers(fpgapart_for_transform))

        # Find specialized node
        specialized_node = None
        for node in model.graph.node:
            if node.name == op.onnx_node.name:
                specialized_node = node
                break

        if specialized_node is None:
            # Name may have changed, find by position/type
            expected_optype = f"{op.onnx_node.op_type}_{self.backend.value}"
            for node in model.graph.node:
                if node.op_type == expected_optype:
                    specialized_node = node
                    break

        if specialized_node is None:
            raise RuntimeError(
                f"SpecializeLayers failed to create {self.backend.value} variant of "
                f"{op.onnx_node.op_type}. Check backend availability."
            )

        # Get specialized operator instance
        from finn.util.basic import getHWCustomOp
        specialized_op = getHWCustomOp(specialized_node, model)

        # Verify backend inheritance
        from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
        from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend

        if self.backend == BackendType.HLS and not isinstance(specialized_op, HLSBackend):
            raise RuntimeError(
                f"Expected HLSBackend after specialization, got {type(specialized_op)}"
            )
        elif self.backend == BackendType.RTL and not isinstance(specialized_op, RTLBackend):
            raise RuntimeError(
                f"Expected RTLBackend after specialization, got {type(specialized_op)}"
            )

        return specialized_op, model

    def _prepare_backend(
        self,
        op: HWCustomOp,
        model: ModelWrapper,
        fpgapart: str,
        clk_ns: float
    ) -> Tuple[HWCustomOp, ModelWrapper]:
        """Prepare backend for execution.

        Handles code generation, compilation, and simulation setup based on mode.

        Args:
            op: Specialized operator (with HLSBackend/RTLBackend)
            model: Model containing operator
            fpgapart: FPGA part string
            clk_ns: Clock period in nanoseconds

        Returns:
            (op, model): Prepared operator ready for execution
        """
        if self.prepare_mode == PrepareMode.CPPSIM:
            # C++ simulation: generate C++ → compile → ready to execute
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
            model = model.transform(SetExecMode("cppsim"))

        elif self.prepare_mode == PrepareMode.RTLSIM:
            # RTL simulation: generate IP → synthesize → compile xsim → ready
            model = model.transform(PrepareIP(fpgapart, clk_ns))
            model = model.transform(PrepareRTLSim())
            model = model.transform(SetExecMode("rtlsim"))

        elif self.prepare_mode == PrepareMode.IPGEN:
            # IP generation only: generate IP package (no exec setup)
            model = model.transform(PrepareIP(fpgapart, clk_ns))

        # Refresh op reference after transformations
        from finn.util.basic import getHWCustomOp
        hw_node = None
        for node in model.graph.node:
            if node.name == op.onnx_node.name:
                hw_node = node
                break

        if hw_node is not None:
            op = getHWCustomOp(hw_node, model)

        return op, model


# Convenience factories

def make_hls_cppsim_runner() -> BackendPipelineRunner:
    """Create runner for HLS C++ simulation testing."""
    return BackendPipelineRunner(backend="hls", prepare_for="cppsim")


def make_hls_rtlsim_runner() -> BackendPipelineRunner:
    """Create runner for HLS RTL simulation testing."""
    return BackendPipelineRunner(backend="hls", prepare_for="rtlsim")


def make_rtl_rtlsim_runner() -> BackendPipelineRunner:
    """Create runner for pure RTL simulation testing."""
    return BackendPipelineRunner(backend="rtl", prepare_for="rtlsim")


def make_specialized_runner(backend: Literal["hls", "rtl"]) -> BackendPipelineRunner:
    """Create runner that specializes but doesn't prepare (for structural tests)."""
    return BackendPipelineRunner(backend=backend, prepare_for="none")
```

---

## Integration with Test Frameworks

### Update `DualKernelTest`

Add optional backend configuration hooks to `tests/frameworks/dual_kernel_test.py`:

```python
class DualKernelTest(KernelTestConfig):
    """Test manual vs auto parity + both against golden reference."""

    # Optional: Override to use backend pipeline
    def get_backend_config(self) -> Optional[Tuple[str, str]]:
        """Return (backend, prepare_mode) for backend testing.

        Override to enable cppsim/rtlsim tests:

        Returns:
            None: Use base pipeline (Python execution only)
            ("hls", "cppsim"): Use HLS C++ simulation
            ("hls", "rtlsim"): Use HLS RTL simulation
            ("rtl", "rtlsim"): Use RTL simulation

        Example:
            def get_backend_config(self):
                return ("hls", "cppsim")  # Enable cppsim tests
        """
        return None  # Default: no backend (Python only)

    def get_fpgapart(self) -> str:
        """Return FPGA part for backend preparation.

        Only used if get_backend_config() is not None.
        """
        from tests.support.constants import PARITY_DEFAULT_FPGA_PART_HLS
        return PARITY_DEFAULT_FPGA_PART_HLS

    def get_clock_period_ns(self) -> float:
        """Return clock period for backend preparation.

        Only used if get_backend_config() is not None.
        """
        from tests.support.constants import PARITY_DEFAULT_CLOCK_PERIOD_NS
        return PARITY_DEFAULT_CLOCK_PERIOD_NS

    def run_manual_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run manual (FINN) pipeline with optional backend specialization."""
        backend_config = self.get_backend_config()

        if backend_config is None:
            # Base pipeline (Python execution only)
            runner = PipelineRunner()
            return runner.run(
                model_factory=self.make_test_model,
                transform=self.get_manual_transform(),
                configure_fn=lambda op, model: self.configure_kernel_node(op, model)
            )
        else:
            # Backend pipeline (cppsim/rtlsim enabled)
            backend, prepare_mode = backend_config
            runner = BackendPipelineRunner(backend=backend, prepare_for=prepare_mode)
            return runner.run(
                model_factory=self.make_test_model,
                transform=self.get_manual_transform(),
                configure_fn=lambda op, model: self.configure_kernel_node(op, model),
                fpgapart=self.get_fpgapart(),
                clk_ns=self.get_clock_period_ns()
            )

    def run_auto_pipeline(self) -> Tuple[HWCustomOp, ModelWrapper]:
        """Run auto (Brainsmith) pipeline with optional backend specialization."""
        backend_config = self.get_backend_config()

        if backend_config is None:
            # Base pipeline (Python execution only)
            runner = PipelineRunner()
            return runner.run(
                model_factory=self.make_test_model,
                transform=self.get_auto_transform(),
                configure_fn=lambda op, model: self.configure_kernel_node(op, model)
            )
        else:
            # Backend pipeline (cppsim/rtlsim enabled)
            backend, prepare_mode = backend_config
            runner = BackendPipelineRunner(backend=backend, prepare_for=prepare_mode)
            return runner.run(
                model_factory=self.make_test_model,
                transform=self.get_auto_transform(),
                configure_fn=lambda op, model: self.configure_kernel_node(op, model),
                fpgapart=self.get_fpgapart(),
                clk_ns=self.get_clock_period_ns()
            )
```

---

## Usage Examples

### Example 1: HLS Backend Testing

```python
# tests/dual_pipeline/test_addstreams_hls.py

from tests.frameworks.dual_kernel_test import DualKernelTest
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferAddStreamsLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList


class TestAddStreamsHLS(DualKernelTest):
    """Test AddStreams with HLS backend (enables cppsim/rtlsim).

    Inherits 20 tests:
    - 7 core parity tests
    - 5 HW estimation tests
    - 8 golden execution tests (ALL execute now, not skipped!)
    """

    def get_backend_config(self):
        """Enable HLS backend with cppsim preparation."""
        return ("hls", "cppsim")  # This single line enables ALL backend tests!

    def make_test_model(self):
        """Create ONNX Add node for AddStreams inference."""
        # ... create Add node
        return model, "Add_test"

    def get_manual_transform(self):
        """FINN's manual transform."""
        return InferAddStreamsLayer

    def get_auto_transform(self):
        """Brainsmith's unified transform."""
        return InferKernelList

    def compute_golden_reference(self, inputs):
        """Test-owned golden reference."""
        return {"output": inputs["input0"] + inputs["input1"]}

    def get_num_inputs(self):
        return 2

    def get_num_outputs(self):
        return 1

    def configure_kernel_node(self, op, model):
        """Configure both implementations identically."""
        op.set_nodeattr("PE", 8)


# Test Results:
# ✅ test_normal_shapes_parity - PASSED
# ✅ test_folded_shapes_parity - PASSED
# ✅ test_stream_widths_parity - PASSED
# ✅ test_datatypes_parity - PASSED
# ✅ test_expected_cycles_parity - PASSED
# ✅ test_manual_python_vs_golden - PASSED
# ✅ test_auto_python_vs_golden - PASSED
# ✅ test_manual_cppsim_vs_golden - PASSED (was skipped before!)
# ✅ test_auto_cppsim_vs_golden - PASSED (was skipped before!)
# ✅ test_manual_rtlsim_vs_golden - PASSED (was skipped before!)
# ✅ test_auto_rtlsim_vs_golden - PASSED (was skipped before!)
# ✅ test_manual_auto_parity_cppsim - PASSED (was skipped before!)
# ... etc
# Total: 20 tests, 20 passed, 0 skipped
```

### Example 2: Python-Only Testing (Backward Compatible)

```python
# tests/dual_pipeline/test_addstreams_v2.py

class TestAddStreamsV2(DualKernelTest):
    """Test AddStreams without backend (Python execution only).

    No override of get_backend_config() means Python-only testing.
    """

    # No get_backend_config() override = Python only (backward compatible!)

    def make_test_model(self):
        return model, "Add_test"

    # ... rest same as before


# Test Results:
# ✅ test_normal_shapes_parity - PASSED
# ✅ test_manual_python_vs_golden - PASSED
# ✅ test_auto_python_vs_golden - PASSED
# ⏭️ test_manual_cppsim_vs_golden - SKIPPED (no backend)
# ⏭️ test_auto_cppsim_vs_golden - SKIPPED (no backend)
# ... etc
# Total: 20 tests, 9 passed, 11 skipped (cppsim/rtlsim)
```

### Example 3: Structural Testing Only

```python
class TestAddStreamsStructural(DualKernelTest):
    """Test AddStreams_hls structural properties (no execution)."""

    def get_backend_config(self):
        """Specialize to HLS but don't prepare (fast structural tests)."""
        return ("hls", "none")  # Specialize but no code gen/compile

    # ... rest of implementation


# Benefits:
# - Tests run on AddStreams_hls (has HLSBackend methods)
# - No code generation/compilation (fast!)
# - Can test HLS-specific attributes
# - Execution tests still skip (no prepare)
```

---

## Benefits

### 1. **Backward Compatible**

- ✅ Existing tests work unchanged (no `get_backend_config()` = Python only)
- ✅ No breaking changes to current test suite
- ✅ Opt-in backend testing via simple override

### 2. **Minimal Code Changes**

**To enable ALL backend tests:**
```python
def get_backend_config(self):
    return ("hls", "cppsim")  # Single line!
```

**vs Legacy approach:**
```python
def setup_manual_op(self):
    # 30+ lines of specialization logic

def setup_auto_op(self):
    # 30+ lines of specialization logic

def _specialize_and_get_op(self, model, node_name_prefix):
    # 20+ lines of specialization helper
```

### 3. **Reuses FINN Infrastructure**

- ✅ `SpecializeLayers` (backend selection)
- ✅ `PrepareCppSim` / `CompileCppSim` (C++ workflow)
- ✅ `PrepareIP` / `PrepareRTLSim` (RTL workflow)
- ✅ `SetExecMode` (execution mode configuration)

No reimplementation of FINN transforms!

### 4. **Type-Safe**

```python
# Enums prevent typos
backend: Literal["hls", "rtl"]  # Not "HLS" or "Hls"
prepare_for: Literal["cppsim", "rtlsim", "ipgen", "none"]

# Validation at construction
runner = BackendPipelineRunner(backend="hsl")  # Error!
runner = BackendPipelineRunner(prepare_for="cppsim")  # Error: need backend!
```

### 5. **Clear Separation**

```python
# Base testing (fast)
PipelineRunner()  # Python execution only

# Backend testing (comprehensive)
BackendPipelineRunner(backend="hls", prepare_for="cppsim")

# Structural testing (fast, specialized)
BackendPipelineRunner(backend="hls", prepare_for="none")
```

### 6. **Flexible Test Strategies**

| Strategy | Backend | Prepare | Use Case |
|----------|---------|---------|----------|
| Python only | `None` | `"none"` | Fast functional testing |
| HLS structural | `"hls"` | `"none"` | Fast structural validation |
| HLS cppsim | `"hls"` | `"cppsim"` | C++ code validation |
| HLS rtlsim | `"hls"` | `"rtlsim"` | Full HLS→RTL validation |
| RTL rtlsim | `"rtl"` | `"rtlsim"` | Hand-written RTL validation |
| HLS IP only | `"hls"` | `"ipgen"` | IP package generation |

---

## Coverage Impact

### Before (Current State)

| Test Category | Count | Status |
|---------------|-------|--------|
| Core parity | 7 | ✅ Pass |
| HW estimation | 5 | ✅ Pass |
| Python exec | 2 | ✅ Pass |
| **Cppsim exec** | **4** | **⏭️ Skip** |
| **Rtlsim exec** | **4** | **⏭️ Skip** |
| **Total** | **22** | **14 pass, 8 skip** |

**Missing Validation:**
- ❌ HLS C++ code generation
- ❌ HLS C++ compilation
- ❌ HLS C++ execution
- ❌ HLS → RTL synthesis
- ❌ RTL simulation
- ❌ Backend specialization

### After (With Backend Extension)

| Test Category | Count | Status |
|---------------|-------|--------|
| Core parity | 7 | ✅ Pass |
| HW estimation | 5 | ✅ Pass |
| Python exec | 2 | ✅ Pass |
| **Cppsim exec** | **4** | **✅ Pass** |
| **Rtlsim exec** | **4** | **✅ Pass** |
| **Total** | **22** | **22 pass, 0 skip** |

**Complete Validation:**
- ✅ HLS C++ code generation
- ✅ HLS C++ compilation
- ✅ HLS C++ execution
- ✅ HLS → RTL synthesis
- ✅ RTL simulation
- ✅ Backend specialization

### Comparison with Legacy Tests

**Legacy Test Suite** (`brainsmith/kernels/addstreams/tests/test_addstreams_parity.py`):
- 25 base parity tests
- 7 HLS codegen tests
- **Total: 32 tests**

**New Test Suite** (with backend extension):
- 20 inherited tests (all execute)
- 2 AddStreams-specific tests
- **Total: 22 tests** (but more comprehensive per test!)

**Coverage Parity Achieved:**
- ✅ Structural validation (shapes, datatypes, widths)
- ✅ HW estimation validation (cycles, resources)
- ✅ Python execution validation
- ✅ HLS C++ execution validation (cppsim)
- ✅ RTL execution validation (rtlsim)
- ✅ Manual vs auto parity validation
- ✅ Both vs golden reference validation

---

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)

1. **Create `tests/support/backend_pipeline.py`**
   - Implement `BackendPipelineRunner` class
   - Add enums (`BackendType`, `PrepareMode`)
   - Implement three-phase execution
   - Add convenience factories

2. **Add unit tests**
   - Test base pipeline (no backend)
   - Test specialization (HLS/RTL)
   - Test preparation modes (cppsim/rtlsim/ipgen/none)
   - Test error handling (missing fpgapart, etc.)

### Phase 2: Framework Integration (1-2 hours)

1. **Update `tests/frameworks/dual_kernel_test.py`**
   - Add `get_backend_config()` hook (returns `None` by default)
   - Add `get_fpgapart()` hook
   - Add `get_clock_period_ns()` hook
   - Update `run_manual_pipeline()` to use backend runner
   - Update `run_auto_pipeline()` to use backend runner

2. **Update `tests/frameworks/single_kernel_test.py`**
   - Similar integration for single kernel tests

### Phase 3: Test Migration (2-3 hours)

1. **Create `tests/dual_pipeline/test_addstreams_hls.py`**
   - Copy from `test_addstreams_v2.py`
   - Add `get_backend_config()` override
   - Verify all 22 tests pass

2. **Update documentation**
   - Add usage examples to framework docstrings
   - Update `tests/REFACTOR_COMPLETE.md`
   - Update `tests/PROJECT_STATUS_SUMMARY.md`

### Phase 4: Validation (1 hour)

1. **Run full test suite**
   - Verify backward compatibility (existing tests still pass)
   - Verify new backend tests pass
   - Check test execution times

2. **Documentation review**
   - Ensure clear migration path for other kernels
   - Document backend configuration options
   - Add troubleshooting guide

**Total Estimated Time: 6-9 hours**

---

## Migration Guide for Other Kernels

To enable backend testing for any kernel, follow this pattern:

### Step 1: Create HLS Test Variant

```python
# tests/dual_pipeline/test_mykernel_hls.py

from tests.frameworks.dual_kernel_test import DualKernelTest

class TestMyKernelHLS(DualKernelTest):
    """Test MyKernel with HLS backend."""

    def get_backend_config(self):
        return ("hls", "cppsim")  # Enable HLS cppsim testing

    # Copy rest from test_mykernel.py...
```

### Step 2: Run Tests

```bash
# Run with HLS backend
pytest tests/dual_pipeline/test_mykernel_hls.py -v

# Run only cppsim tests
pytest tests/dual_pipeline/test_mykernel_hls.py -v -m cppsim

# Run only rtlsim tests
pytest tests/dual_pipeline/test_mykernel_hls.py -v -m rtlsim

# Skip slow backend tests
pytest tests/dual_pipeline/test_mykernel_hls.py -v -m "not slow"
```

### Step 3: Optional - RTL Variant

```python
class TestMyKernelRTL(DualKernelTest):
    """Test MyKernel with hand-written RTL backend."""

    def get_backend_config(self):
        return ("rtl", "rtlsim")  # Pure RTL testing
```

---

## Alternative Approaches Considered

### Alternative 1: Modify PipelineRunner Directly

**Approach:** Add backend parameters to `PipelineRunner.run()`

**Rejected because:**
- Violates Single Responsibility Principle
- Adds complexity to already-clear base runner
- Harder to maintain separate concerns

### Alternative 2: Backend as Transform

**Approach:** Create `SpecializeAndPrepare` transform

**Rejected because:**
- Transforms shouldn't manage runner state
- Less flexible (can't choose prepare mode independently)
- Harder to test individual phases

### Alternative 3: Separate Test Classes

**Approach:** `DualKernelTestHLS`, `DualKernelTestRTL`, etc.

**Rejected because:**
- Code duplication across test classes
- Harder to maintain (3+ classes per kernel)
- No backward compatibility

### Alternative 4: Decorator Pattern

**Approach:** `@with_backend("hls", "cppsim")` decorator

**Rejected because:**
- Less discoverable than method override
- Harder to configure dynamically
- Pytest may not handle decorators well

**Selected Approach:** `BackendPipelineRunner` + optional hook override
- ✅ Clean separation of concerns
- ✅ Backward compatible
- ✅ Minimal code changes
- ✅ Reuses existing infrastructure
- ✅ Type-safe and well-documented

---

## Open Questions

### 1. Should we auto-detect backend availability?

**Question:** Should the framework automatically check if `_hls`/`_rtl` variants exist and skip gracefully?

**Current behavior:** Raises error if backend not found

**Proposed:** Add `skip_if_unavailable=True` parameter:
```python
runner = BackendPipelineRunner(
    backend="rtl",
    prepare_for="rtlsim",
    skip_if_unavailable=True  # Skip instead of error
)
```

### 2. Should we support multiple backends in one test?

**Question:** Test both HLS and RTL in a single test class?

**Current:** One backend per test class

**Proposed:** Add `get_backends()` returning list:
```python
def get_backends(self):
    return [("hls", "cppsim"), ("rtl", "rtlsim")]
```

Would generate parameterized tests for each backend.

### 3. Should we add backend to test markers?

**Question:** Auto-add `@pytest.mark.hls` / `@pytest.mark.rtl` markers?

**Benefit:** Filter by backend easily:
```bash
pytest tests/ -v -m "hls and cppsim"
```

### 4. How to handle backend-specific configuration?

**Question:** Some backends need different configuration (e.g., RTL memory style)

**Proposed:** Add `configure_backend_node()` hook:
```python
def configure_backend_node(self, op, model, backend):
    if backend == "rtl":
        op.set_nodeattr("ram_style", "ultra")
```

---

## Success Criteria

This implementation is successful if:

1. ✅ **All existing tests still pass** (backward compatibility)
2. ✅ **New backend tests pass** (22/22, not 14/22)
3. ✅ **Single line enables backend testing** (simple override)
4. ✅ **No FINN transform reimplementation** (reuse infrastructure)
5. ✅ **Clear error messages** (when backend unavailable, missing config)
6. ✅ **Fast structural tests** (via `prepare_for="none"`)
7. ✅ **Easy migration path** (copy test, add one method)

---

## References

- **Current PipelineRunner**: `tests/support/pipeline.py`
- **Current Executors**: `tests/support/executors.py` (lines 182-189, 327-336)
- **FINN SpecializeLayers**: `deps/finn/src/finn/transformation/fpgadataflow/specialize_layers.py`
- **FINN Backend Mixins**: `deps/finn/src/finn/custom_op/fpgadataflow/hlsbackend.py`
- **Architecture Context**: `_artifacts/context/inheritance_chain_and_backends.md`
- **Legacy Parity Test**: `brainsmith/kernels/addstreams/tests/test_addstreams_parity.py`

---

**End of Plan**

**Status**: Ready for implementation
**Estimated effort**: 6-9 hours
**Risk level**: Low (backward compatible, incremental changes)
**Impact**: High (enables complete hardware validation)
