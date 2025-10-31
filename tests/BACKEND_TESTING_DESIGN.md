# Backend Testing Integration Design

**Date**: 2025-10-31
**Status**: Design Phase
**Context**: Adding cppsim/rtlsim verification to test framework

---

## Problem Statement

Current test framework only verifies **Python execution** of base kernels (AddStreams). Backend execution (cppsim/rtlsim) is completely skipped because:

1. PipelineRunner creates base nodes (AddStreams) without backend inheritance
2. Executors check `isinstance(op, HLSBackend)` and skip if False
3. No specialization step: AddStreams → AddStreams_hls

**Result**: 6 of 22 tests skip (all cppsim/rtlsim tests)

---

## FINN's Testing Pattern (Reference)

From `deps/finn/tests/fpgadataflow/test_fpgadataflow_addstreams.py`:

```python
def test_fpgadataflow_addstreams(idts, ch, fold, exec_mode):
    # 1. Create ONNX model
    model = make_addstreams_modelwrapper(ch, idts)

    # 2. Verify Python execution (base ONNX Add)
    y_expected = x1 + x2
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()

    # 3. Lower to HW layer (creates AddStreams base kernel)
    model = model.transform(to_hw.InferAddStreamsLayer())
    addstreams_node.set_nodeattr("PE", pe)

    # 4. Specialize to backend (AddStreams → AddStreams_hls)
    model = model.transform(SpecializeLayers("xc7z020clg400-1"))

    # 5. Prepare execution infrastructure
    if exec_mode == "cppsim":
        model = model.transform(PrepareCppSim())      # Generate C++
        model = model.transform(CompileCppSim())      # Compile
        model = model.transform(SetExecMode("cppsim"))
    elif exec_mode == "rtlsim":
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(fpgapart, clk))  # Generate RTL
        model = model.transform(HLSSynthIP())              # Synthesize
        model = model.transform(PrepareRTLSim())           # Compile xsim

    # 6. Execute backend and verify
    y_produced = oxe.execute_onnx(model, input_dict)["outp"]
    assert (y_produced == y_expected).all()
```

**Key observations**:
- Single test verifies BOTH Python and backend execution
- SpecializeLayers is the bridge from base → backend
- Preparation transforms set up infrastructure (code gen, compilation)
- SetExecMode tells nodes which simulation mode to use
- execute_onnx() handles dispatch based on exec_mode attribute

---

## Proposed Architecture

### Three-Phase Test Flow

```
Phase 1: Base Kernel (Python)
  ↓
  PipelineRunner creates base kernel (AddStreams)
  Python Executor runs execute_node()
  Validate vs golden
  ↓
Phase 2: Backend Specialization (Optional)
  ↓
  SpecializeLayers(fpgapart) → AddStreams_hls
  ↓
Phase 3: Backend Execution (Optional)
  ↓
  Prepare simulation (cppsim or rtlsim)
  Backend Executor runs execute_node()
  Validate vs golden
```

### Component Design

#### 1. BackendConfig (Data Class)

```python
@dataclass
class BackendConfig:
    """Configuration for backend testing."""
    backend_type: Literal["hls", "rtl"]
    exec_mode: Literal["cppsim", "rtlsim"]
    fpgapart: str = "xc7z020clg400-1"
    clk_ns: float = 5.0
```

#### 2. BackendSpecializer (Utility)

```python
class BackendSpecializer:
    """Handles backend specialization and preparation."""

    def specialize(
        self,
        op,
        model: ModelWrapper,
        config: BackendConfig
    ) -> Tuple[Any, ModelWrapper]:
        """Convert base kernel to backend variant.

        AddStreams → AddStreams_hls (if backend_type="hls")
        AddStreams → AddStreams_rtl (if backend_type="rtl")
        """
        model = model.transform(SpecializeLayers(config.fpgapart))

        # Find specialized node
        specialized_node = self._find_specialized_node(
            model,
            op.onnx_node.name,
            config.backend_type
        )
        specialized_op = get_hw_custom_op(specialized_node, model)

        return specialized_op, model

    def prepare_execution(
        self,
        model: ModelWrapper,
        config: BackendConfig
    ) -> ModelWrapper:
        """Prepare backend for execution."""
        if config.exec_mode == "cppsim":
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
            model = model.transform(SetExecMode("cppsim"))

        elif config.exec_mode == "rtlsim":
            model = model.transform(SetExecMode("rtlsim"))
            model = model.transform(PrepareIP(config.fpgapart, config.clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(PrepareRTLSim())

        return model
```

#### 3. DualKernelTest Integration

```python
class DualKernelTest(KernelTestConfig):
    """Base class for dual-kernel parity testing."""

    def get_backend_config(self) -> Optional[BackendConfig]:
        """Override to enable backend testing.

        Returns:
            None: Only test base kernel (Python execution)
            BackendConfig: Test base + backend execution

        Example:
            def get_backend_config(self):
                return BackendConfig(
                    backend_type="hls",
                    exec_mode="cppsim",
                    fpgapart="xc7z020clg400-1"
                )
        """
        return None

    def _run_parity_test(self, ...):
        """Run parity test with optional backend verification."""

        # Phase 1: Base kernel test (ALWAYS RUNS)
        brn_op, brn_model = self._setup_brainsmith_pipeline(...)
        finn_op, finn_model = self._setup_finn_pipeline(...)

        # Verify Python execution
        brn_output = self._execute_brainsmith(brn_model, inputs)
        finn_output = self._execute_finn(finn_model, inputs)

        self._validate_parity(brn_op, finn_op, brn_model, finn_model)
        assert_values_match(brn_output, finn_output)

        # Phase 2 & 3: Backend testing (OPTIONAL)
        backend_config = self.get_backend_config()
        if backend_config:
            self._run_backend_verification(
                brn_op, brn_model,
                finn_op, finn_model,
                inputs,
                backend_config
            )

    def _run_backend_verification(
        self,
        brn_op, brn_model,
        finn_op, finn_model,
        inputs,
        config: BackendConfig
    ):
        """Run backend specialization and execution verification."""
        specializer = BackendSpecializer()

        # Specialize both models
        brn_op, brn_model = specializer.specialize(brn_op, brn_model, config)
        finn_op, finn_model = specializer.specialize(finn_op, finn_model, config)

        # Prepare execution
        brn_model = specializer.prepare_execution(brn_model, config)
        finn_model = specializer.prepare_execution(finn_model, config)

        # Execute backend
        brn_output = self._execute_brainsmith(brn_model, inputs)
        finn_output = self._execute_finn(finn_model, inputs)

        # Verify parity still holds
        assert_values_match(
            brn_output, finn_output,
            msg=f"Backend parity failed ({config.backend_type}/{config.exec_mode})"
        )
```

#### 4. Test Usage Example

```python
class TestAddStreamsHLS(DualKernelTest):
    """AddStreams with HLS backend verification."""

    @staticmethod
    def get_kernel_pair() -> KernelPair:
        return KernelPair(
            brainsmith_kernel="brainsmith:AddStreams",
            finn_kernel="finn:AddStreams"
        )

    def get_backend_config(self) -> BackendConfig:
        """Enable HLS cppsim testing."""
        return BackendConfig(
            backend_type="hls",
            exec_mode="cppsim",
            fpgapart="xc7z020clg400-1",
            clk_ns=5.0
        )

    # Inherits 20 tests from DualKernelTest
    # Each test now runs:
    #   1. Python execution (base kernel)
    #   2. cppsim execution (HLS backend)

    @pytest.mark.slow
    def test_rtlsim(self, shape_2d, datatype_pair):
        """Additional test for rtlsim verification."""
        config = BackendConfig(
            backend_type="hls",
            exec_mode="rtlsim"
        )
        # ... test implementation using config ...
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (2-3 hours)
1. Create `tests/support/backend_testing.py`:
   - `BackendConfig` dataclass
   - `BackendSpecializer` class
   - Helper functions

2. Add imports from FINN:
   ```python
   from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
   from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
   from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
   from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
   from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
   from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
   from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
   ```

### Phase 2: Framework Integration (1-2 hours)
1. Update `tests/frameworks/dual_kernel_test.py`:
   - Add `get_backend_config()` hook (default None)
   - Add `_run_backend_verification()` method
   - Integrate into `_run_parity_test()`

2. Update `tests/support/__init__.py`:
   - Export `BackendConfig`, `BackendSpecializer`

### Phase 3: Example Implementation (1-2 hours)
1. Create `tests/dual_pipeline/test_addstreams_hls.py`:
   - Implement `TestAddStreamsHLS` with cppsim
   - Add `test_rtlsim()` for rtlsim verification

2. Add pytest markers:
   ```python
   @pytest.mark.fpgadataflow  # Requires FPGA tools
   @pytest.mark.vivado        # Requires Vivado
   @pytest.mark.slow          # Takes time (compilation)
   ```

### Phase 4: Validation (1 hour)
1. Run base tests (should still pass):
   ```bash
   pytest tests/dual_pipeline/test_addstreams_v2.py -v
   ```

2. Run backend tests:
   ```bash
   pytest tests/dual_pipeline/test_addstreams_hls.py -v -m cppsim
   pytest tests/dual_pipeline/test_addstreams_hls.py -v -m rtlsim --slow
   ```

---

## Benefits

### 1. Complete Coverage
- **Before**: Only Python execution tested (base kernel)
- **After**: Python + cppsim + rtlsim tested (full verification)

### 2. Opt-In Design
- Default: `get_backend_config() → None` (no backend testing)
- Enable: Override to return `BackendConfig`
- No breaking changes to existing tests

### 3. FINN Compatibility
- Uses FINN's transforms directly (SpecializeLayers, etc.)
- Follows FINN's testing pattern
- Reuses FINN's backend infrastructure

### 4. Flexible Configuration
- Per-test backend selection (hls vs rtl)
- Per-test execution mode (cppsim vs rtlsim)
- Configurable FPGA part and clock

---

## Test Execution Flow

### Without Backend Config (Current State)
```
test_basic_shapes()
  → Create base kernel (AddStreams)
  → Execute Python
  → Verify vs golden
  → PASS ✓
```

### With Backend Config (New)
```
test_basic_shapes()
  → Create base kernel (AddStreams)
  → Execute Python
  → Verify vs golden
  → PASS ✓ (Phase 1)

  → Specialize to backend (AddStreams_hls)
  → Prepare cppsim (generate C++, compile)
  → Execute cppsim
  → Verify vs golden
  → PASS ✓ (Phase 2)
```

---

## Addressing Original Problem

### Before
```
TestAddStreamsV2
  ✓ test_basic_shapes[python]        # Passes
  ⏭ test_basic_shapes[cppsim]        # SKIPPED (not an HLS backend)
  ⏭ test_basic_shapes[rtlsim]        # SKIPPED (not an HLS backend)
```

### After
```
TestAddStreamsHLS
  ✓ test_basic_shapes[python]        # Passes (base kernel)
  ✓ test_basic_shapes[cppsim]        # Passes (HLS backend)
  ✓ test_basic_shapes[rtlsim]        # Passes (HLS backend)
```

**Coverage Impact**:
- Before: 16/22 tests run (6 skip)
- After: 22/22 tests run (0 skip)

---

## Next Steps

1. ✅ Design complete
2. ⏭ Implement `backend_testing.py`
3. ⏭ Integrate into `DualKernelTest`
4. ⏭ Create `test_addstreams_hls.py` example
5. ⏭ Validate with pytest

**Estimated Time**: 5-7 hours total
