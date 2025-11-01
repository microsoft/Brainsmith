# Test Framework Quick Reference

**TL;DR**: Use `SingleKernelTest` or `DualKernelTest`. Everything else is utilities.

---

## Quick Links

**Full Architecture Report**: `TEST_DIRECTORY_ARCHITECTURE_REPORT.md`  
**For Details On**: Single kernel testing, dual parity, golden reference validation, pipeline architecture

---

## Test Framework Checklist

### For Testing ONE Kernel (SingleKernelTest)

```python
from tests.frameworks.single_kernel_test import SingleKernelTest

class TestMyKernel(SingleKernelTest):
    ✅ def make_test_model(self) -> Tuple[ModelWrapper, str]
    ✅ def get_kernel_inference_transform(self) -> Type[Transformation]
    ✅ def compute_golden_reference(self, inputs) -> Dict[str, np.ndarray]
    ✅ def get_num_inputs(self) -> int
    ✅ def get_num_outputs(self) -> int
    
    ⏺ def configure_kernel_node(self, op, model) -> None  # Optional
    ⏺ def get_backend_fpgapart(self) -> str  # Optional: enable backend tests
    ⏺ def get_tolerance_*() -> Dict  # Optional: customize tolerances
```

**Gets Automatically**: 6 inherited tests
- test_pipeline_creates_hw_node
- test_shapes_preserved_through_pipeline
- test_datatypes_preserved_through_pipeline
- test_python_execution_vs_golden
- test_cppsim_execution_vs_golden (slow)
- test_rtlsim_execution_vs_golden (slow)

---

### For Testing DUAL Implementations (DualKernelTest)

```python
from tests.frameworks.dual_kernel_test import DualKernelTest

class TestMyKernelParity(DualKernelTest):
    ✅ def make_test_model(self) -> Tuple[ModelWrapper, str]
    ✅ def get_manual_transform(self) -> Type[Transformation]  # FINN
    ✅ def get_auto_transform(self) -> Type[Transformation]  # Brainsmith
    ✅ def compute_golden_reference(self, inputs) -> Dict[str, np.ndarray]
    ✅ def get_num_inputs(self) -> int
    ✅ def get_num_outputs(self) -> int
    
    ⏺ def configure_kernel_node(self, op, model) -> None  # Optional
```

**Gets Automatically**: 20 inherited tests
- 7 core parity tests (shapes, widths, datatypes)
- 5 HW estimation tests (cycles, resources)
- 8 golden execution tests (manual/auto vs golden)

---

## Support Utilities (For Advanced Use)

| Utility | Use When |
|---------|----------|
| `PipelineRunner` | Need custom ONNX → Hardware transformation |
| `GoldenValidator` | Need standalone validation (outside frameworks) |
| `PythonExecutor` | Need to execute Python backend manually |
| `CppSimExecutor` | Need to execute HLS C++ backend manually |
| `RTLSimExecutor` | Need to execute RTL backend manually |
| `make_execution_context` | Need to generate test data manually |
| `OnnxModelBuilder` | Need to build custom ONNX test models |

---

## Test Organization by Category

| Category | Purpose | Use When |
|----------|---------|----------|
| `frameworks/` | Test framework classes | Writing new kernel tests |
| `support/` | Reusable utilities | Building custom test utilities |
| `fixtures/` | Test data and models | Building test models, fixtures |
| `pipeline/` | Kernel pipeline tests | Testing complete ONNX → HW flow |
| `integration/` | DSE framework tests | Testing DSE framework, not kernels |
| `dual_pipeline/` | Legacy dual testing | Migrating to DualKernelTest |
| `unit/` | Component tests | Testing individual utilities |
| `kernels/` | Kernel-specific tests | Testing kernel behavior (not via framework) |

---

## Running Tests

### All tests
```bash
pytest tests/ -v
```

### Single kernel tests only
```bash
pytest tests/ -m "single_kernel" -v
```

### Dual kernel tests only
```bash
pytest tests/ -m "dual_kernel" -v
```

### Skip slow tests (cppsim, rtlsim)
```bash
pytest tests/ -m "not slow" -v
```

### Only golden reference tests
```bash
pytest tests/ -m "golden" -v
```

### Only HLS C++ simulation tests
```bash
pytest tests/ -m "cppsim" -v --run-slow
```

### Only RTL simulation tests
```bash
pytest tests/ -m "rtlsim" -v --run-slow
```

### Run pipeline tests
```bash
pytest tests/pipeline/ -v
```

---

## Key Concepts

### Test Ownership Model
- Tests own golden references (TEST LOGIC)
- Kernels contain production code only
- No coupling between kernels and test infrastructure

### Pipeline Stages
1. **Stage 1**: ONNX node (standard ONNX: Add, Mul, etc.)
2. **Stage 2**: Base kernel (no backend: AddStreams, MVAU, etc.)
3. **Stage 3**: Backend kernel (with backend: AddStreams_hls, etc.)

### Composition Pattern
```
SingleKernelTest/DualKernelTest
    ↓ uses
PipelineRunner (Stage 1 → Stage 2)
    + specialize_to_backend() (Stage 2 → Stage 3)
    ↓ with
GoldenValidator (pure validation)
    ↓ using
Executors (PythonExecutor, CppSimExecutor, RTLSimExecutor)
    ↓ and
make_execution_context (test data generation)
```

---

## Configuration Examples

### Minimum (Python execution only)
```python
class TestMyKernel(SingleKernelTest):
    def make_test_model(self):
        return model, node_name
    def get_kernel_inference_transform(self):
        return InferMyKernel
    def compute_golden_reference(self, inputs):
        return {"output": np.my_op(inputs["input"])}
    def get_num_inputs(self):
        return 1
    def get_num_outputs(self):
        return 1
```

### With Backend Testing
```python
class TestMyKernel(SingleKernelTest):
    # ... required methods ...
    
    def configure_kernel_node(self, op, model):
        op.set_nodeattr("PE", 8)
        op.set_nodeattr("SIMD", 16)
    
    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"  # Enable cppsim/rtlsim tests
```

### With Custom Tolerances
```python
class TestMyKernel(SingleKernelTest):
    # ... required methods ...
    
    def get_tolerance_python(self):
        return {"rtol": 1e-5, "atol": 1e-6}  # Looser than default
    
    def get_tolerance_cppsim(self):
        return {"rtol": 1e-4, "atol": 1e-5}  # Looser for HLS
```

---

## File Locations

```
tests/
├── frameworks/                      # ← Start here for new tests
│   ├── kernel_test_base.py         # Minimal abstract base
│   ├── single_kernel_test.py       # Use this ✅
│   ├── dual_kernel_test.py         # Or this ✅
│   └── test_addstreams_validation.py # Example
│
├── support/                         # Support utilities
│   ├── context.py
│   ├── executors.py
│   ├── validator.py
│   ├── pipeline.py
│   ├── assertions.py
│   └── ...
│
├── fixtures/                        # Test fixtures
│   ├── kernel_test_helpers.py
│   ├── components/
│   └── ...
│
└── TEST_DIRECTORY_ARCHITECTURE_REPORT.md  # Full details
```

---

## Common Questions

**Q: How do I test my kernel?**
A: Create a class inheriting from `SingleKernelTest`, implement 5 required methods, inherit 6 tests.

**Q: How do I compare manual vs auto implementations?**
A: Use `DualKernelTest` instead. Implement `get_manual_transform()` and `get_auto_transform()`.

**Q: What's the difference between SingleKernelTest and DualKernelTest?**
A: Single tests ONE implementation (6 tests). Dual tests TWO implementations against each other AND both vs golden (20 tests).

**Q: What's the "golden reference"?**
A: NumPy implementation of what the kernel should compute. Tests define this, not kernels.

**Q: Can I run just Python tests?**
A: Yes: `pytest tests/ -m "not slow" -v`

**Q: What if my kernel has 2 inputs?**
A: Implement `get_num_inputs()` to return 2. Framework handles the rest.

**Q: How do I enable HLS/RTL testing?**
A: Override `get_backend_fpgapart()` to return FPGA part string.

**Q: Can I customize tolerances?**
A: Yes, override `get_tolerance_python()`, `get_tolerance_cppsim()`, `get_tolerance_rtlsim()`.

---

## Pytest Markers

```bash
@pytest.mark.pipeline       # Pipeline integration test
@pytest.mark.golden        # Tests golden reference
@pytest.mark.parity        # Tests manual vs auto parity
@pytest.mark.single_kernel # SingleKernelTest tests
@pytest.mark.dual_kernel   # DualKernelTest tests
@pytest.mark.cppsim        # HLS C++ simulation
@pytest.mark.rtlsim        # RTL simulation
@pytest.mark.slow          # Slow test (> 1 min)
@pytest.mark.fast          # Fast test (< 1 min)
@pytest.mark.phase1/2/3    # Implementation phase
```

---

## See Also

- **Full Report**: `TEST_DIRECTORY_ARCHITECTURE_REPORT.md`
- **SingleKernelTest API**: `frameworks/single_kernel_test.py`
- **DualKernelTest API**: `frameworks/dual_kernel_test.py`
- **Working Example**: `frameworks/test_addstreams_validation.py`
- **Pipeline Details**: `pipeline/README.md`
