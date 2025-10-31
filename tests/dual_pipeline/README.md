# Dual Pipeline Parity Testing Framework

## Overview

The **DualPipelineParityTest** framework provides comprehensive testing for kernel implementations by combining:

1. **Golden Reference Validation** - Each implementation tested against NumPy ground truth (absolute correctness)
2. **Hardware Parity Validation** - Manual vs Auto implementations compared (migration safety)

This unified approach gives you the best of both worlds in a single test class.

## Philosophy

```
┌─────────────────────────────────────────────────────────────┐
│                  DualPipelineParityTest                      │
│                                                              │
│  Manual FINN              Auto Brainsmith                    │
│       │                         │                            │
│       ├─ ONNX Transform        ├─ ONNX Transform            │
│       ├─ Shapes Inference      ├─ Shapes Inference          │
│       ├─ Datatype Inference    ├─ Datatype Inference        │
│       ├─ HLS Specialization    ├─ HLS Specialization        │
│       │                         │                            │
│       v                         v                            │
│  ✓ vs Golden Reference    ✓ vs Golden Reference             │
│       │                         │                            │
│       └────── Compare Hardware ─┘                           │
│         (stream widths, folded shapes,                       │
│          resource estimates, cycles)                         │
└─────────────────────────────────────────────────────────────┘
```

**Key Insight**: Hardware specs (stream widths, folded shapes, resource estimates) have no NumPy equivalent. We validate them by comparing manual vs auto implementations.

## Quick Start

```python
from tests.dual_pipeline import DualPipelineParityTest
from brainsmith.kernels.mykernel import MyKernel
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferMyKernelLayer
from brainsmith.primitives.transforms.infer_kernel_list import InferKernelList

class TestMyKernelDualParity(DualPipelineParityTest):
    """Complete testing for MyKernel: golden + parity."""

    def make_test_model(self):
        """Create standard ONNX node (NOT hardware node)."""
        model = create_onnx_model()  # Standard ONNX
        return model, "MyOp_0"

    def get_manual_transform(self):
        """FINN's manual transform."""
        return InferMyKernelLayer

    def get_auto_transform(self):
        """Brainsmith's unified transform."""
        return InferKernelList

    def get_kernel_class(self):
        """Kernel class for golden reference."""
        return MyKernel

    def configure_kernel_node(self, op, model, is_manual):
        """Configure both implementations identically."""
        op.set_nodeattr("PE", 8)
```

That's it! You now have **~20 tests automatically**:
- ✅ Manual Python execution vs golden
- ✅ Auto Python execution vs golden
- ✅ Manual HLS cppsim vs golden
- ✅ Auto HLS cppsim vs golden
- ✅ Stream widths parity (manual vs auto)
- ✅ Folded shapes parity
- ✅ Resource estimates parity
- ✅ Cycles parity
- ✅ Datatypes parity
- ✅ And 11 more...

## Test Categories

### 1. Golden Reference Tests (4 tests)

Validate each implementation against NumPy ground truth:

```python
@pytest.mark.golden
def test_manual_python_execution_vs_golden(self):
    """FINN manual implementation must match NumPy."""

@pytest.mark.golden
def test_auto_python_execution_vs_golden(self):
    """Brainsmith auto implementation must match NumPy."""

@pytest.mark.golden
@pytest.mark.cppsim
@pytest.mark.slow
def test_manual_cppsim_execution_vs_golden(self):
    """FINN HLS code generation must match NumPy."""

@pytest.mark.golden
@pytest.mark.cppsim
@pytest.mark.slow
def test_auto_cppsim_execution_vs_golden(self):
    """Brainsmith HLS code generation must match NumPy."""
```

**What this proves**: Both implementations produce correct results.

### 2. Hardware Parity Tests (12 tests)

Compare manual vs auto hardware specs:

```python
@pytest.mark.parity
def test_stream_widths_parity(self):
    """Stream widths must match between implementations."""

@pytest.mark.parity
def test_folded_shapes_parity(self):
    """Folded shapes must match (PE parallelization)."""

@pytest.mark.parity
def test_resource_estimates_parity(self):
    """LUT/BRAM/URAM estimates must match."""

@pytest.mark.parity
def test_expected_cycles_parity(self):
    """Cycle counts must match."""
```

**What this proves**: Migration from manual → auto is safe (produces identical hardware).

### 3. Integration Tests (4 tests)

Validate pipeline execution:

```python
@pytest.mark.integration
def test_both_pipelines_create_hw_nodes(self):
    """Both transforms successfully create HW nodes."""

@pytest.mark.integration
def test_both_specializations_succeed(self):
    """Both implementations specialize to HLS."""
```

**What this proves**: Complete pipeline works end-to-end.

## Complete Test List

| # | Test Name | Category | Validates |
|---|-----------|----------|-----------|
| 1 | `test_manual_python_execution_vs_golden` | Golden | Manual Python → NumPy |
| 2 | `test_auto_python_execution_vs_golden` | Golden | Auto Python → NumPy |
| 3 | `test_manual_cppsim_execution_vs_golden` | Golden | Manual HLS → NumPy |
| 4 | `test_auto_cppsim_execution_vs_golden` | Golden | Auto HLS → NumPy |
| 5 | `test_normal_shapes_parity` | Parity | Input/output shapes match |
| 6 | `test_folded_shapes_parity` | Parity | Folded shapes match |
| 7 | `test_stream_widths_parity` | Parity | Stream widths match |
| 8 | `test_stream_widths_padded_parity` | Parity | Padded widths match (AXI) |
| 9 | `test_datatypes_parity` | Parity | Input/output datatypes match |
| 10 | `test_datatype_inference_parity` | Parity | Inference logic matches |
| 11 | `test_expected_cycles_parity` | Parity | Cycle counts match |
| 12 | `test_number_output_values_parity` | Parity | FIFO sizing matches |
| 13 | `test_resource_estimates_parity` | Parity | LUT/BRAM/URAM match |
| 14 | `test_efficiency_metrics_parity` | Parity | Efficiency estimates match |
| 15 | `test_operation_counts_parity` | Parity | Op/param counts match |
| 16 | `test_make_shape_compatible_op_parity` | Parity | Shape inference helper |
| 17 | `test_both_pipelines_create_hw_nodes` | Integration | Transforms succeed |
| 18 | `test_both_hw_nodes_have_same_type` | Integration | Same base op_type |
| 19 | `test_both_specializations_succeed` | Integration | HLS specialization |
| 20 | `test_golden_reference_properties` | Golden | Mathematical properties |

**Total: 20 comprehensive tests**

## Running Tests

```bash
# Run all dual pipeline tests
pytest tests/dual_pipeline/test_addstreams_v2.py -v

# Run only golden reference tests (correctness)
pytest tests/dual_pipeline/test_addstreams_v2.py -v -m golden

# Run only parity tests (equivalence)
pytest tests/dual_pipeline/test_addstreams_v2.py -v -m parity

# Run fast tests only (skip slow cppsim)
pytest tests/dual_pipeline/test_addstreams_v2.py -v -m "not slow"

# Run all dual pipeline tests across all kernels
pytest tests/dual_pipeline/ -v
```

## Configuration Hooks

### Required Methods

```python
def make_test_model(self) -> Tuple[ModelWrapper, str]:
    """Create standard ONNX model (NOT hardware node)."""

def get_manual_transform(self) -> Type[Transformation]:
    """Return FINN's manual transform class."""

def get_auto_transform(self) -> Type[Transformation]:
    """Return Brainsmith's auto transform class."""

def get_kernel_class(self) -> Type[KernelOp]:
    """Return kernel class with compute_golden_reference()."""
```

### Optional Methods

```python
def get_num_inputs(self) -> int:
    """Number of inputs. Default: 1"""
    return 2  # For multi-input kernels

def get_num_outputs(self) -> int:
    """Number of outputs. Default: 1"""
    return 1

def configure_kernel_node(self, op, model, is_manual):
    """Configure both implementations identically."""
    op.set_nodeattr("PE", 8)
    op.set_nodeattr("SIMD", 16)

def get_golden_tolerance_python(self):
    """Python execution tolerance."""
    return {"rtol": 1e-7, "atol": 1e-9}

def get_golden_tolerance_cppsim(self):
    """C++ simulation tolerance."""
    return {"rtol": 1e-5, "atol": 1e-6}
```

## Adding Kernel-Specific Tests

You can add kernel-specific tests alongside the inherited ones:

```python
class TestAddStreamsDualParity(DualPipelineParityTest):
    # ... configuration ...

    @pytest.mark.dual_pipeline
    @pytest.mark.parity
    def test_overflow_prevention_both_implementations(self):
        """Both must widen INT8 + INT8 → INT9."""
        manual_op, _ = self.run_manual_pipeline()
        auto_op, _ = self.run_auto_pipeline()

        assert manual_op.get_output_datatype(0) == DataType["INT9"]
        assert auto_op.get_output_datatype(0) == DataType["INT9"]

    @pytest.mark.dual_pipeline
    @pytest.mark.golden
    def test_commutativity_both_implementations(self):
        """Both must satisfy a + b == b + a."""
        # Test mathematical property
```

## Golden Reference Requirements

Your kernel class must implement:

```python
class MyKernel(KernelOp):
    @staticmethod
    def compute_golden_reference(inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """NumPy reference implementation.

        Args:
            inputs: Dict mapping input names → numpy arrays

        Returns:
            Dict mapping output names → expected numpy arrays
        """
        return {"output": inputs["input0"] + inputs["input1"]}

    @staticmethod
    def validate_golden_properties(inputs: Dict, outputs: Dict) -> None:
        """Optional: Validate mathematical properties."""
        # Test commutativity, associativity, etc.
```

## Comparison: DualPipelineParityTest vs Alternatives

### vs ParityTestBase (Old Framework)

| Feature | ParityTestBase | DualPipelineParityTest |
|---------|---------------|----------------------|
| **Golden Reference** | ❌ No | ✅ Yes - all 4 tests |
| **Hardware Parity** | ✅ Yes - 26 tests | ✅ Yes - 12 critical tests |
| **HLS Code Gen Tests** | ✅ 7 template tests | ❌ Not needed (cppsim proves it) |
| **Test Count** | 26 tests | 20 tests |
| **Absolute Correctness** | ❌ Relative only | ✅ Golden reference |
| **Migration Safety** | ✅ Yes | ✅ Yes |

**Advantage**: DualPipelineParityTest validates absolute correctness (vs NumPy), not just equivalence.

### vs IntegratedPipelineTest (Old Framework)

| Feature | IntegratedPipelineTest | DualPipelineParityTest |
|---------|----------------------|----------------------|
| **Golden Reference** | ✅ 2 tests | ✅ 4 tests (manual + auto) |
| **Hardware Validation** | ❌ No parity | ✅ 12 parity tests |
| **Tests Both Impls** | ❌ One only | ✅ Both manual + auto |
| **Migration Safety** | ❌ No | ✅ Yes |

**Advantage**: DualPipelineParityTest tests BOTH implementations in parallel.

## Migration Guide

### From ParityTestBase

**Before**:
```python
class TestAddStreamsHLSParity(ParityTestBase, HLSCodegenParityMixin):
    @property
    def manual_op_class(self):
        return AddStreams_hls

    @property
    def auto_op_class(self):
        return AddStreams_hls

    # Custom setup_manual_op(), setup_auto_op()
    # 26 + 7 = 33 inherited tests
```

**After**:
```python
class TestAddStreamsDualParity(DualPipelineParityTest):
    def get_manual_transform(self):
        return InferAddStreamsLayer

    def get_auto_transform(self):
        return InferKernelList

    def get_kernel_class(self):
        return AddStreams

    # 20 inherited tests + golden reference validation
```

**Benefits**:
- ✅ Simpler configuration (no custom setup methods)
- ✅ Golden reference validation included
- ✅ Reuses transform-based workflow (more realistic)

### From IntegratedPipelineTest

**Before**:
```python
class TestAddStreamsIntegration(IntegratedPipelineTest):
    # Tests ONE implementation (auto only)
    # 5 inherited tests
```

**After**:
```python
class TestAddStreamsDualParity(DualPipelineParityTest):
    # Tests BOTH implementations (manual + auto)
    # 20 inherited tests
```

**Benefits**:
- ✅ Tests both manual and auto in parallel
- ✅ 4x more comprehensive (20 tests vs 5)
- ✅ Migration safety validation included

## Why DualPipelineParityTest?

**For Small Teams**:
- ✅ One test framework to maintain
- ✅ Comprehensive coverage (~20 tests automatically)
- ✅ Tests absolute correctness AND migration safety

**For Compiler Teams**:
- ✅ Validates complete pipeline (ONNX → Hardware → Execution)
- ✅ Catches both correctness bugs AND equivalence regressions
- ✅ Hardware specs validated (no NumPy equivalent)

**For Migration**:
- ✅ Proves manual → auto transition is safe
- ✅ Each implementation validated against ground truth
- ✅ Three-way validation: manual ↔ golden ↔ auto

## Troubleshooting

### Test Failure: "Manual implementation wrong"
- Check: `test_manual_python_execution_vs_golden` failed
- **Fix**: Update manual FINN implementation or golden reference

### Test Failure: "Auto implementation wrong"
- Check: `test_auto_python_execution_vs_golden` failed
- **Fix**: Update auto Brainsmith implementation or golden reference

### Test Failure: "Implementations differ"
- Check: Parity test failed (e.g., `test_stream_widths_parity`)
- **Fix**: Investigate why manual and auto produce different hardware
- **Note**: Both may be "correct" but need to produce identical hardware

### Test Failure: "Golden reference not implemented"
- Error: `NotImplementedError: MyKernel does not implement compute_golden_reference()`
- **Fix**: Add `@staticmethod compute_golden_reference()` to kernel class

## Examples

See:
- `tests/dual_pipeline/test_addstreams_v2.py` - Complete AddStreams example with test-owned golden reference
- `tests/parity/core_parity_test.py` - Core parity testing (shapes, widths, datatypes)
- `tests/parity/hw_estimation_parity_test.py` - HW estimation parity testing (resources, cycles)

## FAQ

**Q: Do I need both manual and auto implementations?**
A: During migration, yes. After migration, you can use the framework with only auto implementation (just set manual = auto).

**Q: Can I skip cppsim tests (they're slow)?**
A: Yes! Use `pytest -m "not slow"` to skip them during development.

**Q: What if I don't have a golden reference yet?**
A: Start with ParityTestBase, migrate to DualPipelineParityTest once you have golden reference.

**Q: Can I test RTL backends?**
A: Not yet (Phase 2 feature). Current framework focuses on HLS cppsim.

**Q: How do I test just one implementation?**
A: Use markers: `pytest -m "manual"` (coming soon) or use IntegratedPipelineTest for single-impl testing.

---

**Framework Version**: 1.0
**Created**: 2025-01-XX
**Maintainer**: Brainsmith Team
