# Pipeline Integration Tests

End-to-end integration testing for the complete ONNX → Hardware → Execution pipeline with golden reference validation.

## Overview

Pipeline integration tests validate the **complete transformation flow** from standard ONNX nodes to hardware-accelerated kernels, ensuring correctness against golden reference implementations (NumPy/PyTorch).

### Key Features

- **Pipeline Validation**: Tests complete ONNX → Shapes → Datatypes → Kernel → Backend flow
- **Golden Reference**: All backends must match NumPy/PyTorch ground truth
- **Multi-Backend**: Validates Python, HLS cppsim, RTL rtlsim consistency
- **Property-Based**: Validates mathematical properties (e.g., commutativity for addition)
- **Phased Implementation**: Three phases of increasing sophistication

### Compared to Other Test Types

| Test Type | Purpose | Validates | Reference |
|-----------|---------|-----------|-----------|
| **Pipeline Tests** (this) | Kernel correctness | ONNX → HW pipeline vs NumPy golden reference | Single source of truth (golden) |
| **Parity Tests** (`tests/parity/`) | Implementation equivalence | Manual HWCustomOp vs KernelOp implementations | Each other (no golden reference) |
| **DSE Integration** (`tests/integration/`) | DSE framework | Design space exploration logic | DSE invariants |
| **Unit Tests** (`tests/unit/`) | Component isolation | Individual functions/classes | Expected outputs |

**Key Insight**: Pipeline tests answer "Is the kernel correct?" while parity tests answer "Do two implementations match?"

---

## Quick Start

### Run All Fast Tests
```bash
# Run all pipeline tests (excludes slow cppsim/rtlsim)
pytest tests/pipeline/ -v

# Run specific kernel
pytest tests/pipeline/test_addstreams_integration.py -v

# Run only golden reference tests
pytest tests/pipeline/ -v -m golden
```

### Run Slow Tests (HLS/RTL Simulation)

Slow tests require environment variables (LD_LIBRARY_PATH, XILINX_VIVADO, etc.) to be set before Python starts.

#### Recommended: Use direnv (Automatic)

```bash
# Setup once
brainsmith project allow-direnv

# Daily use - automatic environment loading
cd ~/work/project     # direnv auto-loads on cd
pytest tests/pipeline/ -v --run-slow
pytest tests/pipeline/ -v -m rtlsim

# Config changes - automatic
vim brainsmith.yaml
cd .  # Auto-regenerates and reloads
```

**How it works:**
- direnv detects when you cd into the project directory
- Automatically activates virtualenv and loads environment variables
- Uses `watch_file` to detect config changes and auto-regenerate
- Automatically cleans up when you cd out

#### Alternative: Manual Activation

```bash
# Activate environment (once per session)
source .brainsmith/env.sh

# Run tests
pytest tests/pipeline/ -v --run-slow
pytest tests/pipeline/ -v -m rtlsim

# Config changes - manual regeneration
vim brainsmith.yaml
brainsmith project init  # Regenerate scripts (won't overwrite config.yaml)
source .brainsmith/env.sh     # Reload environment
```

**How it works:**
- Shell script sets all environment variables
- Must be re-sourced after config changes
- Use `source .brainsmith/deactivate.sh` to clean up when done

### Run Phase-Specific Tests
```bash
# Phase 1: Pipeline + golden reference (fast)
pytest tests/pipeline/ -v -m phase1

# Phase 2: Cross-backend + parametric (medium)
pytest tests/pipeline/ -v -m phase2

# Phase 3: Snapshots + properties (advanced)
pytest tests/pipeline/ -v -m phase3
```

---

## Phase 1: Core Pipeline + Golden Reference ✅ IMPLEMENTED

**Status**: Implemented and validated for AddStreams

### What It Tests
- ✅ ONNX node → Hardware node transformation
- ✅ Shape/datatype preservation through pipeline
- ✅ Python execution vs golden reference
- ✅ HLS C++ simulation vs golden reference

### Test Organization

| Test File | Kernel | Tests | Coverage |
|-----------|--------|-------|----------|
| `test_addstreams_integration.py` | AddStreams | 10 | Pipeline, golden reference, properties |

### Example: AddStreams

```python
class TestAddStreamsIntegration(IntegratedPipelineTest):
    """Complete pipeline integration test for AddStreams."""

    def make_test_model(self):
        # Create ONNX Add node (standard ONNX)
        ...
        return model, "Add_test"

    def get_kernel_inference_transform(self):
        return InferKernelList  # Converts Add → AddStreams

    def get_kernel_class(self):
        return AddStreams  # For golden reference
```

### Inherited Tests (from IntegratedPipelineTest)

All subclasses automatically get these tests:

1. **test_pipeline_creates_hw_node()** - Validates kernel inference
2. **test_shapes_preserved_through_pipeline()** - Shape correctness
3. **test_datatypes_preserved_through_pipeline()** - Datatype correctness
4. **test_python_execution_vs_golden()** - Python execution validation
5. **test_cppsim_execution_vs_golden()** - HLS simulation validation (slow)

### Golden Reference Pattern

Each kernel implements:
```python
class MyKernel(KernelOp):
    @staticmethod
    def compute_golden_reference(inputs: Dict) -> Dict:
        """NumPy reference implementation."""
        return {"output": np.some_operation(inputs["input0"])}

    @staticmethod
    def validate_golden_properties(inputs: Dict, outputs: Dict):
        """Validate mathematical properties."""
        # Check invariants, properties, constraints
        ...
```

---

## Phase 2: Multi-Backend + Parametric ✅ IMPLEMENTED

**Status**: Implemented with manual backend specification

### Key Features
- ✅ Manual backend specification via `get_target_backend()`
- ✅ RTL simulation support (`execute_rtlsim()`)
- ✅ Automatic backend detection and routing
- ✅ Parametric testing across shapes, datatypes, PE/SIMD
- ✅ Independent backend validation against golden reference

### Backend Testing Strategy

**Core principle**: Each backend is validated **independently** against the golden reference. No explicit cross-backend parity tests needed.

**Transitive property**: If HLS matches golden AND RTL matches golden, then HLS ≈ RTL (by transitivity).

### Writing Multi-Backend Tests

#### Approach 1: Separate Test Classes per Backend

Create one test class for each backend you want to test:

```python
class TestThresholdingHLS(IntegratedPipelineTest):
    """Test HLS backend with HLS-appropriate configurations."""

    def get_target_backend(self):
        return "hls"  # Prefer HLS backend

    def get_test_datatype(self):
        return DataType.INT16  # HLS optimized for wide bitwidths

    def configure_kernel_node(self, op, model):
        op.set_nodeattr("PE", 7)  # Non-power-of-2 (HLS supports)

    # Inherited tests:
    # - test_python_execution_vs_golden()
    # - test_cppsim_execution_vs_golden()  (actually runs HLS cppsim)


class TestThresholdingRTL(IntegratedPipelineTest):
    """Test RTL backend with RTL-appropriate configurations."""

    def get_target_backend(self):
        return "rtl"  # Prefer RTL backend

    def get_test_datatype(self):
        return DataType.INT8  # RTL optimized for narrow bitwidths

    def configure_kernel_node(self, op, model):
        op.set_nodeattr("PE", 4)  # Power-of-2 (RTL requirement)

    # Inherited tests:
    # - test_python_execution_vs_golden()
    # - test_cppsim_execution_vs_golden()  (actually runs RTL rtlsim)
```

**When to use**: When backends have different constraints or optimal configurations.

#### Approach 2: Parametric Tests Within Backend Class

Sweep parameter space for a specific backend:

```python
class TestThresholdingHLSParametric(TestThresholdingHLS):
    """Parametric tests for HLS backend."""

    @pytest.mark.phase2
    @pytest.mark.parametrize("datatype", [
        DataType.INT16,
        DataType.INT32,
    ])
    def test_various_datatypes(self, datatype):
        """Test HLS backend with various datatypes."""
        self.get_test_datatype = lambda: datatype
        self.test_python_execution_vs_golden()

    @pytest.mark.phase2
    @pytest.mark.parametrize("shape", [
        (1, 32),   # Small
        (1, 64),   # Medium
        (1, 128),  # Large
        (8, 64),   # Multi-batch
    ])
    def test_various_shapes(self, shape):
        """Test HLS backend with various shapes."""
        self.get_test_shape = lambda: shape
        self.test_python_execution_vs_golden()
```

**When to use**: Test configuration space for a single backend.

### Backend Selection Notes

**Important**: Setting `get_target_backend()` expresses a *preference* but doesn't guarantee that backend will be selected. SpecializeLayers may choose a different backend based on:
- Backend availability
- Kernel configuration compatibility
- Fallback logic

**Test author responsibilities**:
1. Configure kernel parameters appropriate for target backend
2. Understand backend constraints (e.g., RTL requires power-of-2 PE)
3. Verify tests exercise the intended backend

**Future enhancement**: Automated backend capability detection (not in Phase 2).

### Example: Multi-Backend Test Organization

```bash
tests/pipeline/test_thresholding_integration.py
├── TestThresholdingHLS              # INT16/32, arbitrary PE
├── TestThresholdingRTL              # INT4/8, power-of-2 PE
├── TestThresholdingHLSParametric    # Sweep HLS design space
└── TestThresholdingRTLParametric    # Sweep RTL design space
```

### Running Backend-Specific Tests

```bash
# Test HLS backend only
pytest tests/pipeline/ -v -k "HLS"

# Test RTL backend only (slow, requires Vivado)
pytest tests/pipeline/ -v -k "RTL" --run-slow

# Test all backends for a kernel
pytest tests/pipeline/test_thresholding_integration.py -v

# Test Phase 2 parametric tests
pytest tests/pipeline/ -v -m "phase2"
```

---

## Phase 3: Advanced Features **[PLANNED]**

**Status**: Planned for future implementation

### Planned Features
- Pipeline stage validation (explicit stage modeling)
- Graph state snapshots (debugging aid)
- Property validation framework (mathematical invariants)

### Planned Infrastructure
```python
class PipelineStage:
    """Explicit pipeline stage modeling."""
    name: str
    transforms: List[Transformation]
    validators: List[Callable]

def test_pipeline_stages_sequential():
    """Test each stage with validators."""
    for stage in self.get_pipeline_stages():
        # Apply transforms
        # Run validators
        # Capture snapshot
```

---

## Writing a New Pipeline Test

### Step 1: Add Golden Reference to Kernel

```python
# In brainsmith/kernels/mykernel/mykernel.py

class MyKernel(KernelOp):
    # ... existing code ...

    @staticmethod
    def compute_golden_reference(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """NumPy reference implementation.

        Args:
            inputs: Dict with input arrays (e.g., {"input": np.array(...)})

        Returns:
            Dict with output arrays (e.g., {"output": np.array(...)})
        """
        # Implement using NumPy
        return {"output": np.my_operation(inputs["input"])}

    @staticmethod
    def validate_golden_properties(inputs: Dict, outputs: Dict) -> None:
        """Validate mathematical properties.

        Args:
            inputs: Input arrays
            outputs: Output arrays from golden reference

        Raises:
            AssertionError: If properties violated
        """
        # Check properties (e.g., monotonicity, bounds, etc.)
        assert np.all(outputs["output"] >= 0), "Output must be non-negative"
```

### Step 2: Create Integration Test File

```python
# In tests/pipeline/test_mykernel_integration.py

from tests.pipeline.base_integration_test import IntegratedPipelineTest

class TestMyKernelIntegration(IntegratedPipelineTest):
    """Complete pipeline integration test for MyKernel."""

    # Required methods
    def make_test_model(self):
        """Create standard ONNX model (not hardware node)."""
        # Create ONNX node (e.g., Softmax, LayerNorm, etc.)
        ...
        return model, node_name

    def get_kernel_inference_transform(self):
        """Return transform that creates HW node."""
        from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList
        return InferKernelList

    def get_kernel_class(self):
        """Return kernel class for golden reference."""
        from brainsmith.kernels.mykernel import MyKernel
        return MyKernel

    # Optional configuration
    def configure_kernel_node(self, op, model):
        """Configure node after inference (e.g., set PE/SIMD)."""
        op.set_nodeattr("PE", 8)

    def get_num_inputs(self):
        """Override if > 1 input."""
        return 1

    def get_num_outputs(self):
        """Override if > 1 output."""
        return 1
```

### Step 3: Run Tests

```bash
# Run your new tests
pytest tests/pipeline/test_mykernel_integration.py -v

# Run fast tests only
pytest tests/pipeline/test_mykernel_integration.py -v -m "not slow"

# Run with HLS simulation
pytest tests/pipeline/test_mykernel_integration.py -v --run-slow
```

---

## Test Execution

### Fast Tests (< 1 min)
- Python execution vs golden
- Pipeline validation (shapes, datatypes)
- Property validation
- Mathematical invariant checks

**Runs on**: Every commit, pre-merge CI

### Slow Tests (1-10 min)
- HLS cppsim compilation and execution
- RTL rtlsim compilation and execution
- Cross-backend validation

**Runs on**: Manual trigger, nightly CI

### Very Slow Tests (10+ min)
- Full parametric sweep
- All backend combinations
- Comprehensive property validation

**Runs on**: Weekly, pre-major-release

### Execution Patterns

```bash
# Fast feedback loop (< 1 min)
pytest tests/pipeline/ -v -m "phase1 and not slow"

# Full validation before commit (< 5 min)
pytest tests/pipeline/ -v -m phase1

# Comprehensive validation (10+ min)
pytest tests/pipeline/ -v --run-slow
```

---

## Pytest Markers

### Test Types
- `@pytest.mark.pipeline` - Pipeline integration test
- `@pytest.mark.golden` - Tests golden reference validation
- `@pytest.mark.cppsim` - Requires HLS C++ simulation
- `@pytest.mark.rtlsim` - Requires RTL simulation

### Implementation Phases
- `@pytest.mark.phase1` - Phase 1 tests (pipeline + golden)
- `@pytest.mark.phase2` - Phase 2 tests (cross-backend + parametric)
- `@pytest.mark.phase3` - Phase 3 tests (snapshots + properties)

### Execution Time
- `@pytest.mark.slow` - Slow tests (> 1 min)

### Example Usage

```python
@pytest.mark.pipeline
@pytest.mark.golden
@pytest.mark.phase1
def test_python_execution_vs_golden(self):
    """Test Python execution matches golden reference."""
    ...

@pytest.mark.pipeline
@pytest.mark.cppsim
@pytest.mark.slow
@pytest.mark.phase1
def test_cppsim_execution_vs_golden(self):
    """Test HLS simulation matches golden reference."""
    ...
```

---

## Test Infrastructure

### Base Class: IntegratedPipelineTest

Located in `tests/pipeline/base_integration_test.py`

**Key Methods**:
- `run_inference_pipeline()` - Run ONNX → Hardware transformation
- `run_hls_specialization()` - Specialize to HLS backend
- `compute_golden_reference()` - Delegate to kernel's golden reference
- `execute_python()` - Execute via Python simulation
- `execute_cppsim()` - Execute via HLS C++ simulation
- `validate_against_golden()` - Compare outputs to golden reference

### Helper Infrastructure

**Reused from Parity Tests**:
- `tests/parity/assertions.py` - `assert_arrays_close()` for numerical comparison
- `tests/parity/executors.py` - `CppSimExecutor`, `RTLSimExecutor` for backend execution
- `tests/parity/test_fixtures.py` - `make_execution_context()` for input generation
- `tests/parity/backend_helpers.py` - `setup_hls_backend_via_specialize()` for backend setup

**Pipeline-Specific**:
- `tests/pipeline/conftest.py` - Pytest fixtures (FPGA parts, tolerances, etc.)
- `tests/pipeline/base_integration_test.py` - Core framework

---

## Requirements

### For Fast Tests (Phase 1 without cppsim)
- Python 3.9+
- QONNX, FINN dependencies
- NumPy, PyTorch (for golden references)

### For HLS Simulation (cppsim)
- Vitis HLS 2020.1+ or Vivado HLS
- `VITIS_PATH` or `HLS_PATH` environment variable
- C++ compiler (g++ 7+)

### For RTL Simulation (rtlsim)
- Xilinx Vivado (with XSI support)
- XSI Python bindings built
- `FINN_ROOT` environment variable

---

## Troubleshooting

### Tests Skip (cppsim)
```
SKIPPED [1] ... VITIS_PATH not set
```

**Solution**: Set environment variable
```bash
export VITIS_PATH=/tools/Xilinx/Vitis_HLS/2020.1
```

### Import Errors
```
ImportError: cannot import name 'IntegratedPipelineTest'
```

**Solution**: Check PYTHONPATH includes tests directory
```bash
export PYTHONPATH=/home/tafk/dev/brainsmith-1:$PYTHONPATH
```

### Golden Reference Not Found
```
NotImplementedError: AddStreams does not implement compute_golden_reference()
```

**Solution**: Add golden reference to kernel class (see "Writing a New Pipeline Test")

### Numerical Mismatch
```
AssertionError: Python execution vs golden reference for 'output'
  Max absolute difference: 1.5e-6
```

**Solution**: Adjust tolerance in test class:
```python
def get_golden_tolerance_python(self):
    return {"rtol": 1e-5, "atol": 1e-5}  # More relaxed
```

---

## Migration from Parity Tests

### When to Use Pipeline Tests vs Parity Tests

| Use Pipeline Tests When | Use Parity Tests When |
|------------------------|----------------------|
| Validating kernel correctness | Comparing manual vs auto implementations |
| Testing new kernels | Migrating FINN kernels to KernelOp |
| Ensuring golden reference match | Ensuring implementation equivalence |
| Single implementation exists | Two implementations to compare |

### Migration Path

1. **Keep existing parity tests** during transition
2. **Add golden reference** to kernel class
3. **Create pipeline test** for kernel
4. **Validate both pass** before deprecating parity test
5. **Deprecate parity test** once confidence established

---

## Current Status

### Phase 1: ✅ IMPLEMENTED
- [x] Core framework (`base_integration_test.py`)
- [x] AddStreams example with 10 tests
- [x] Golden reference pattern established
- [x] pytest integration complete
- [x] Documentation complete

### Phase 2: ✅ IMPLEMENTED
- [x] Manual backend specification (`get_target_backend()`)
- [x] RTL simulation support (`execute_rtlsim()`)
- [x] Automatic backend detection and routing
- [x] Parametric testing infrastructure (AddStreams example)
- [x] Documentation complete

### Phase 3: **[PLANNED]**
- [ ] Pipeline stage validation
- [ ] Graph state snapshots
- [ ] Property validation framework
- [ ] Advanced debugging tools

---

## Next Steps

### Immediate (Expand Coverage)
1. Add golden references to more kernels (ElementwiseBinary, Thresholding)
2. Create multi-backend tests for kernels with multiple backends
3. Add parametric tests for existing kernels

### Short Term (Multi-Kernel Coverage)
1. ElementwiseBinary integration test (17 operations)
2. Thresholding integration test (HLS + RTL backends)
3. Expand parametric test coverage
4. Document patterns and best practices

### Long Term (Phase 3)
1. Pipeline snapshot infrastructure
2. Property validation framework
3. Full kernel coverage
4. Deprecate parity tests

---

## Related Documentation

- `tests/parity/base_parity_test.py` - Parity testing framework (comparison approach)
- `tests/integration/README.md` - DSE integration tests (framework validation)
- `brainsmith/kernels/*/README.md` - Kernel-specific documentation
- `docs/TESTING.md` - Overall testing strategy

---

**Questions?** Open an issue or see `tests/pipeline/base_integration_test.py` for detailed API documentation.
