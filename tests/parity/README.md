# Parity Testing Framework

This directory contains the parity testing framework for validating equivalence between FINN's manual HWCustomOp implementations and Brainsmith's AutoHWCustomOp implementations.

## Overview

The parity testing framework provides a reusable base class (`ParityTestBase`) that automates testing of:

- **Shape methods**: `get_normal_input/output_shape()`, `get_folded_input/output_shape()`
- **Stream widths**: `get_instream_width()`, `get_outstream_width()`
- **Datatypes**: `get_input_datatype()`, `get_output_datatype()`
- **Expected cycles**: `get_exp_cycles()`
- **Execution parity**: `execute_node()` with random inputs

### Key Features

- **Transform-based by default**: Tests the complete production workflow (ONNX → Infer → HW node)
- **Minimal boilerplate**: ~20 lines of code for complete test suite
- **80% code reuse**: Inherit from `ParityTestBase` to get 15 generic test methods
- **Handles initialization differences**: Automatic setup via Infer transforms
- **Multi-input/output support**: Automatically tests all inputs and outputs
- **Deterministic testing**: Uses fixed random seed for reproducible results
- **Comprehensive assertions**: Clear error messages showing both manual and auto values

## Quick Start

### Transform-Based Testing (Recommended)

The simplest and most production-realistic approach uses Infer transforms:

```python
from tests.parity import ParityTestBase
from brainsmith.kernels.mykernel import InferHWMyKernel, InferAutoMyKernel, HWMyKernel, AutoMyKernel

class TestMyKernelParity(ParityTestBase):
    """Parity tests for MyKernel using transforms."""

    @property
    def manual_op_class(self):
        return HWMyKernel

    @property
    def auto_op_class(self):
        return AutoMyKernel

    def make_test_model(self):
        """Create standard ONNX model (not HWMyKernel)."""
        from tests.fixtures.model_utils import create_mykernel_model
        model = create_mykernel_model(channels=768)
        return model, "MyKernel_0"

    def get_manual_transform(self):
        """Return the manual Infer transform class."""
        return InferHWMyKernel

    def get_auto_transform(self):
        """Return the auto Infer transform class."""
        return InferAutoMyKernel

    def configure_test_op(self, op, model, is_auto):
        """Configure op after transform (e.g., override SIMD)."""
        op.set_nodeattr("SIMD", 16)
        if is_auto:
            op.refresh_tensor_context(model)
```

That's it! You now have **15 parity tests** that validate the complete production workflow:
- ONNX node → InferTransform → HW node → Execution

### Legacy Direct Node Creation

For cases without Infer transforms, override `setup_manual_op()` and `setup_auto_op()`:

```python
class TestLegacyKernelParity(ParityTestBase):
    # ... abstract properties ...

    def make_test_model(self):
        """Create ONNX model with HWMyKernel node."""
        model = ModelWrapper(...)
        return model, "HWMyKernel_0"

    def get_shared_nodeattrs(self):
        """Nodeattrs needed by both manual and auto ops."""
        return {
            "SIMD": 16,
            "PE": 8,
        }
```

### Running Tests

```bash
# Run all parity tests
./smithy pytest -m parity

# Run parity tests for specific kernel
./smithy pytest tests/parity/test_mykernel_parity.py

# Run with verbose output
./smithy pytest -m parity -v

# Run only execution parity test (slowest but most comprehensive)
./smithy pytest -m parity -k "execute_node"
```

## Usage Examples by Kernel Category

### 1. Shape-Preserving Operations (Simple)

Example: LayerNorm, ReLU

**Transform-Based Testing (Default)**

```python
from tests.parity import ParityTestBase
from brainsmith.kernels.layernorm import LayerNorm_Batch, AutoLayerNorm, InferLayerNorm, InferAutoLayerNorm
from tests.fixtures.model_utils import create_layernorm_model

class TestLayerNormParity(ParityTestBase):
    """Parity tests using Infer transforms (production workflow)."""

    @property
    def manual_op_class(self):
        return LayerNorm_Batch

    @property
    def auto_op_class(self):
        return AutoLayerNorm

    def make_test_model(self):
        """Create standard ONNX LayerNormalization node."""
        model = create_layernorm_model(
            batch_size=1, seq_len=128, channels=768,
            input_dtype="INT8", output_dtype="FLOAT32"
        )
        return model, "LayerNormalization_0"

    def get_manual_transform(self):
        return InferLayerNorm

    def get_auto_transform(self):
        return InferAutoLayerNorm

    def configure_test_op(self, op, model, is_auto):
        """Override SIMD for testing."""
        op.set_nodeattr("SIMD", 16)
        if is_auto:
            op.refresh_tensor_context(model)
```

**Direct Node Creation (Legacy)**

```python
class TestLayerNormParity(ParityTestBase):
    @property
    def manual_op_class(self):
        return LayerNorm_Batch

    @property
    def auto_op_class(self):
        return AutoLayerNorm

    def make_test_model(self):
        """Create test model with LayerNorm node."""
        # Input: [1, 768]
        # Output: [1, 768] (same shape)
        model = create_layernorm_model(seq_len=768)
        return model, "LayerNorm_0"

    def get_shared_nodeattrs(self):
        return {
            "SIMD": 16,
            "PE": 1,
            "numInputVectors": 768,
        }

    def get_manual_only_nodeattrs(self):
        """Manual op needs explicit shape params."""
        return {
            "Dim": [768],  # Feature dimension
        }
```

### 2. Multi-Input Operations

Example: AddStreams, MVAU (data + weights)

```python
class TestAddStreamsParity(ParityTestBase):
    @property
    def manual_op_class(self):
        return AddStreams

    @property
    def auto_op_class(self):
        return AutoAddStreams

    def get_num_inputs(self):
        """AddStreams has 2 inputs."""
        return 2

    def make_test_model(self):
        """Create model with two input streams."""
        # Input0: [1, 32]
        # Input1: [1, 32]
        # Output: [1, 32]
        model = create_addstreams_model(channels=32)
        return model, "AddStreams_0"

    def get_shared_nodeattrs(self):
        return {
            "NumChannels": 32,
            "PE": 1,
        }
```

### 3. Shape-Transforming Operations

Example: Pool, MVAU (matrix multiply)

```python
class TestPoolParity(ParityTestBase):
    @property
    def manual_op_class(self):
        return Pool

    @property
    def auto_op_class(self):
        return AutoPool

    def make_test_model(self):
        """Create model with pooling node."""
        # Input: [1, 56, 56, 64]
        # Output: [1, 28, 28, 64] (spatial reduction)
        model = create_pool_model(
            input_shape=(1, 56, 56, 64),
            kernel_size=2,
            stride=2
        )
        return model, "Pool_0"

    def get_shared_nodeattrs(self):
        return {
            "Channels": 64,
            "PE": 1,
            "Kernel": [2, 2],
            "Stride": [2, 2],
            "PoolType": "max",
        }

    def get_manual_only_nodeattrs(self):
        return {
            "ImgDim": [56, 56],  # Manual op needs explicit spatial dims
        }
```

### 4. Operations with Optional Inputs

Example: LayerNorm with optional bias

```python
class TestLayerNormWithBiasParity(ParityTestBase):
    @property
    def manual_op_class(self):
        return LayerNorm_Batch

    @property
    def auto_op_class(self):
        return AutoLayerNorm

    def get_num_inputs(self):
        """3 inputs: data, weight (gamma), bias (beta)."""
        return 3

    def make_test_model(self):
        """Create model with weight and bias."""
        model = create_layernorm_model(
            seq_len=768,
            with_weight=True,
            with_bias=True
        )
        return model, "LayerNorm_0"

    def get_shared_nodeattrs(self):
        return {
            "SIMD": 16,
            "PE": 1,
            "numInputVectors": 768,
        }
```

## Customization Guide

### When to Override Setup Methods

The transform-based pattern handles most cases. Override `setup_manual_op()` or `setup_auto_op()` only when:

1. **No Infer transform exists**: Legacy kernels without transforms
2. **Non-standard transform workflow**: Custom initialization required
3. **Testing transform variations**: Multiple transform configurations

Example of custom setup:

```python
class TestCustomWorkflowParity(ParityTestBase):
    # ... abstract properties ...

    def setup_manual_op(self):
        """Custom initialization for manual op."""
        model, node_name = self.make_test_model()
        # Custom workflow here
        node = model.graph.node[0]
        op = getCustomOp(node)
        # Custom configuration
        return op, model

    def setup_auto_op(self):
        """Custom initialization for auto op."""
        model, node_name = self.make_test_model()
        # Custom workflow here
        node = model.graph.node[0]
        op = getCustomOp(node)
        op.refresh_tensor_context(model)
        return op, model
```

### Override Execution Context Generation

For special input requirements (e.g., specific value ranges, non-uniform distributions):

```python
class TestMyKernelParity(ParityTestBase):
    # ... abstract methods ...

    def _make_execution_context(self, model, op):
        """Custom input generation."""
        context = {}
        node = op.onnx_node

        # Custom input generation
        inp_shape = op.get_normal_input_shape(0)
        inp_dtype = op.get_input_datatype(0)

        # Generate specific distribution (e.g., softmax needs sum-to-1)
        data = np.random.uniform(0.0, 1.0, size=inp_shape)
        data = data / data.sum()  # Normalize

        context[node.input[0]] = data.astype(np.float32)

        # Pre-allocate output
        out_shape = op.get_normal_output_shape(0)
        context[node.output[0]] = np.zeros(out_shape, dtype=np.float32)

        return context
```

### Add Custom Validation Tests

Extend with kernel-specific tests:

```python
class TestMyKernelParity(ParityTestBase):
    # ... abstract methods ...

    @pytest.mark.parity
    def test_custom_property_parity(self):
        """Test kernel-specific property."""
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_value = manual_op.calc_wmem()
        auto_value = auto_op.calc_wmem()

        assert manual_value == auto_value, (
            f"Weight memory calculation mismatch:\n"
            f"  Manual: {manual_value}\n"
            f"  Auto:   {auto_value}"
        )
```

### Handle Multiple Test Configurations

Use `pytest.mark.parametrize` for testing multiple configurations:

```python
import pytest

class TestLayerNormParity(ParityTestBase):
    # ... abstract methods ...

    @pytest.mark.parity
    @pytest.mark.parametrize("simd,pe", [
        (1, 1),
        (16, 1),
        (32, 1),
        (16, 2),
    ])
    def test_folded_shapes_parametrized(self, simd, pe):
        """Test folded shapes with various SIMD/PE configurations."""
        # Override nodeattrs for this test
        self._test_simd = simd
        self._test_pe = pe

        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_shape = manual_op.get_folded_input_shape(0)
        auto_shape = auto_op.get_folded_input_shape(0)

        assert manual_shape == auto_shape

    def get_shared_nodeattrs(self):
        """Use test-specific values if set."""
        return {
            "SIMD": getattr(self, "_test_simd", 16),
            "PE": getattr(self, "_test_pe", 1),
            "numInputVectors": 768,
        }
```

## When NOT to Use Parity Testing

Parity tests are for validating AutoHWCustomOp migrations. **Do not use** for:

1. **No manual implementation exists**: Just write regular unit tests
2. **Manual implementation is known broken**: Fix it first or skip parity testing
3. **Fundamentally different approaches**: If auto version intentionally differs (e.g., better algorithm)
4. **Infrastructure operations**: Operations like StreamingFIFO where shapes are externally determined

## Troubleshooting

### Problem: Shape mismatch in tests

**Symptom**: `test_normal_input_shape_parity` fails

**Cause**: Manual op needs explicit shape nodeattrs (MW, MH, ImgDim, etc.) that auto op infers

**Fix**: Add manual-only nodeattrs via `get_manual_only_nodeattrs()`:

```python
def get_manual_only_nodeattrs(self):
    return {
        "MW": 16,
        "MH": 16,
        "NumChannels": 64,
    }
```

### Problem: Datatype mismatch

**Symptom**: `test_input_datatype_parity` or `test_output_datatype_parity` fails

**Cause**: Datatypes not set on ONNX tensors, or manual op's `infer_node_datatype` produces different result

**Fix**:
1. Ensure `make_test_model()` sets value_info with datatypes
2. Verify manual op's datatype inference logic
3. Check that auto op's schema has correct `datatype_attr` mappings

### Problem: Execution results differ

**Symptom**: `test_execute_node_parity` fails with numerical differences

**Possible causes**:
1. **Floating-point precision**: Adjust tolerances in test
2. **Different computation order**: Manual and auto may compute in different order (expected for some ops)
3. **Weight layout differences**: Manual and auto may expect different weight tensor formats
4. **Random initialization**: Ensure deterministic seed is set

**Fix for tolerance**:
```python
def test_execute_node_parity(self):
    """Override with custom tolerance."""
    manual_op, manual_model = self.setup_manual_op()
    auto_op, auto_model = self.setup_auto_op()

    np.random.seed(42)
    manual_context = self._make_execution_context(manual_model, manual_op)
    np.random.seed(42)
    auto_context = self._make_execution_context(auto_model, auto_op)

    manual_op.execute_node(manual_context, manual_model.graph)
    auto_op.execute_node(auto_context, auto_model.graph)

    manual_output = manual_context[manual_op.onnx_node.output[0]]
    auto_output = auto_context[auto_op.onnx_node.output[0]]

    # Custom tolerance for this kernel
    np.testing.assert_allclose(
        manual_output,
        auto_output,
        rtol=1e-3,  # Increased tolerance
        atol=1e-5,
    )
```

### Problem: Test skipped with "Cannot generate input"

**Symptom**: Test shows "SKIPPED" with message about input generation

**Cause**: `_make_execution_context()` couldn't generate input (shape/datatype retrieval failed)

**Fix**: Override `_make_execution_context()` with custom input generation for your kernel

### Problem: Manual op initialization fails

**Symptom**: Error during `setup_manual_op()` about missing nodeattrs

**Cause**: Manual op requires nodeattrs that weren't provided

**Fix**: Add missing nodeattrs to `get_shared_nodeattrs()` or `get_manual_only_nodeattrs()`

## Architecture Notes

### Transform-Based Testing Workflow

**Default workflow** (via `_setup_via_transform()`):
1. Create standard ONNX model (e.g., Softmax node)
2. Apply shape and datatype inference
3. Apply kernel-specific Infer transform (e.g., InferHWSoftmax, InferAutoSoftmax)
4. Find transformed hardware node
5. Create custom op instance
6. Configure op (e.g., override SIMD via `configure_test_op()`)
7. Auto ops: Refresh tensor context with new configuration

**Key benefits**:
- Tests the actual production workflow
- Validates Infer transforms (not just kernel implementations)
- Ensures transforms produce equivalent results
- Minimal test code (~20 lines for complete test suite)

### Initialization Sequence Differences

**Manual ops** (transform-based):
1. Infer transform creates HW node with nodeattrs
2. Create op instance from transformed node
3. Override nodeattrs if needed (e.g., SIMD)

**Auto ops** (transform-based):
1. Infer transform creates HW node with nodeattrs
2. Create op instance from transformed node
3. Transform calls `refresh_tensor_context()` → caches shapes/types
4. Override nodeattrs if needed (e.g., SIMD)
5. Call `refresh_tensor_context()` again with new nodeattrs

The ParityTestBase framework handles these differences automatically via `configure_test_op()`.

### What Gets Tested

Each test method validates one aspect of the HWCustomOp interface:

| Test Method | What it checks |
|-------------|----------------|
| `test_normal_input_shape_parity` | `get_normal_input_shape(ind)` for all inputs |
| `test_normal_output_shape_parity` | `get_normal_output_shape(ind)` for all outputs |
| `test_folded_input_shape_parity` | `get_folded_input_shape(ind)` with SIMD/PE folding |
| `test_folded_output_shape_parity` | `get_folded_output_shape(ind)` with SIMD/PE folding |
| `test_instream_width_parity` | `get_instream_width(ind)` in bits |
| `test_outstream_width_parity` | `get_outstream_width(ind)` in bits |
| **`test_instream_width_padded_parity`** | **`get_instream_width_padded(ind)` AXI Stream alignment** |
| **`test_outstream_width_padded_parity`** | **`get_outstream_width_padded(ind)` AXI Stream alignment** |
| `test_input_datatype_parity` | `get_input_datatype(ind)` QONNX DataType |
| `test_output_datatype_parity` | `get_output_datatype(ind)` QONNX DataType |
| **`test_infer_node_datatype_parity`** | **`infer_node_datatype(model)` inference logic (CRITICAL)** |
| `test_exp_cycles_parity` | `get_exp_cycles()` execution latency |
| **`test_number_output_values_parity`** | **`get_number_output_values()` for FIFO sizing** |
| **`test_make_shape_compatible_op_parity`** | **`make_shape_compatible_op(model)` shape inference** |
| `test_execute_node_parity` | `execute_node()` with random inputs (numerical correctness) |

**Total: 15 comprehensive parity tests**

### Critical Test: Datatype Inference Logic

The `test_infer_node_datatype_parity()` test is particularly important because:

1. **Tests logic, not just state**: Unlike other tests that compare getter method results, this test validates that the datatype inference **process** works correctly
2. **Validates side effects**: Checks that both implementations correctly update the ONNX model's tensor datatypes
3. **Different implementations**: Manual and Auto use fundamentally different approaches:
   - **Manual**: Custom logic with explicit nodeattr updates
   - **Auto**: KernelSchema-driven with `datatype_attr` mappings
4. **Critical for transforms**: Datatype inference is called during graph transformations and must produce identical results

This test ensures that both the getter methods (`get_input_datatype()`) AND the inference logic (`infer_node_datatype()`) are equivalent.

## References

- **Migration Guide**: `docs/AutoHWCustomOp_Migration_Guide.md`
- **Design Document**: `docs/parity_testing_design.md`
- **AutoHWCustomOp**: `brainsmith/core/finn/auto_hw_custom_op.py`
- **Dataflow System**: `brainsmith/core/dataflow/`

## Example: Complete Test File

See `tests/parity/test_softmax_parity.py` for a complete working example of transform-based parity testing (NOTE: Softmax has been migrated to unified naming; this test file is retained as a reference example).

This example demonstrates:
- Transform-based setup with Infer transforms
- Testing from standard ONNX nodes
- Custom tolerance for numerical parity (floating-point ops)
- Probability distribution validation (sums to 1.0)
- Only ~115 lines for 21 comprehensive tests (19 passed, 2 skipped)
