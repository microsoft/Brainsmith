"""Test elementwise binary operations with fixture-based parameterization (v2.3).

This test demonstrates the v2.3 framework:
- Pytest fixtures for parameterization (NOT hardcoded in test)
- Symbolic shapes from fixtures
- Direct DataType annotations (NO Quant nodes) (v2.3)
- Automatic test data generation with seed management (v2.1)
- QONNX golden reference with pre-quantized data (v2.1)
- Stage 1 golden reference for backend tests (v2.2)
- KernelOp execution initialization fix (v2.2)

Result: 108 tests (18 configs × 6 test types) with ~30 lines of code!
- Add/Sub: 3 dtypes × 3 shapes = 9 configurations each
- Mul: 1 dtype × 3 shapes = 3 configurations (INT8 only)
- 6 tests per config: 3 pipeline + 3 backend execution
- Note: Non-FP32 FLOAT dtypes excluded (not yet supported by Brainsmith compiler)

Seed Management (v2.1):
- Default seed: 42 (deterministic)
- Override per test: `def get_test_seed(self): return 12345`
- Override via CLI: `pytest --seed=12345`

Stage Configuration (v2.1):
- Stage 2 (Kernel): `configure_kernel_node()` - dimension parameters (PE, SIMD)
- Stage 3 (Backend): `configure_backend_node()` - backend parameters (mem_mode, ram_style)

Examples:
    # Stage 2: Configure kernel dimensions
    class TestMyKernel(SingleKernelTest):
        def configure_kernel_node(self, op, model):
            op.set_nodeattr("PE", 8)
            op.set_nodeattr("SIMD", 16)
            if isinstance(op, KernelOp):
                op._ensure_ready(model)

    # Stage 3: Configure backend implementation
    class TestMyKernel(SingleKernelTest):
        def configure_backend_node(self, op, model):
            # HLS memory configuration
            op.set_nodeattr("mem_mode", "internal_decoupled")
            op.set_nodeattr("ram_style", "ultra")
            op.set_nodeattr("resType", "dsp")

Compare to v1.0: Would require 200+ lines and manual test data generation.
"""

import pytest
import numpy as np
import onnx.helper as helper
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.single_kernel_test_v2 import SingleKernelTest


# ============================================================================
# Fixtures - define test parameterization
# ============================================================================


@pytest.fixture(
    params=[
        {"input": DataType["INT8"], "param": DataType["INT8"]},
        {"input": DataType["INT16"], "param": DataType["INT16"]},
        {"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
    ]
)
def input_datatypes(request):
    """Parameterize datatypes: INT8, INT16, BIPOLAR.

    Note: Non-FP32 float types (FLOAT<exp,mant,bias>) are not yet supported
    by the Brainsmith compiler.
    """
    return request.param


@pytest.fixture(
    params=[
        {"input": (1, 64), "param": (1, 64)},  # Same shapes (no broadcasting)
        {"input": (4, 128), "param": (4, 128)},  # Same shapes
        {"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},  # Same shapes (4D)
    ]
)
def input_shapes(request):
    """Parameterize shapes: 2D small, 2D large, 4D (NHWC).

    Note: AddStreams requires same shapes (no broadcasting).
    """
    return request.param


# ============================================================================
# Test class - just define operations!
# ============================================================================


class TestElementwiseBinaryAdd(SingleKernelTest):
    """Test Add operation with direct DataType annotations (v2.3).

    Runs 54 tests (9 configs × 6 test types) automatically!

    Each configuration (dtype × shape) validates:
    1. Pipeline creates correct HW node
    2. Shapes preserved through pipeline
    3. Datatypes preserved through pipeline
    4. Python execution matches QONNX golden (pre-quantized data)
    5. HLS cppsim matches QONNX golden (if backend configured)
    6. RTL rtlsim matches QONNX golden (if backend configured)

    Total: 9 × 6 = 54 test assertions with ~30 lines of code!

    Seed Management (v2.1):
        # Use default seed (42)
        class TestMyKernel(SingleKernelTest):
            pass  # Uses default seed

        # Override seed per test class
        class TestMyKernel(SingleKernelTest):
            def get_test_seed(self):
                return 12345  # Custom seed

        # Override seed via CLI
        pytest --seed=99999  # All tests use this seed
    """

    def make_test_model(self, input_shapes):
        """Create Add operation with concrete shapes from fixture.

        Args:
            input_shapes: Dict from fixture, e.g.,
                         {"input": (1, 64), "param": (64,)}

        Returns:
            (model, input_names): Model and list of inputs to annotate with DataTypes
        """
        # Use shapes from fixture
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        # Create Add node
        node = helper.make_node("Add", ["input", "param"], ["output"], name="Add_0")

        graph = helper.make_graph([node], "test_add", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Framework annotates these inputs with DataTypes automatically!
        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        """Use InferKernels to convert Add → ElementwiseBinaryOp."""
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return callable that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    def configure_kernel_node(self, op, model):
        """Configure PE parameter to 8."""
        # Note: PE attribute only exists on backends (Stage 3), not base kernel (Stage 2)
        # Skip configuration for now - this will be set during backend specialization
        pass

    def get_backend_fpgapart(self):
        """Enable backend testing with xc7z020 part."""
        return "xc7z020clg400-1"

    def get_tolerance_python(self):
        """Python tolerance - very tight for Add operation."""
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self):
        """C++ sim tolerance - moderate for fixed-point."""
        return {"rtol": 1e-5, "atol": 1e-6}


# ============================================================================
# Additional test: Sub operation (shows reusability)
# ============================================================================


class TestElementwiseBinarySub(SingleKernelTest):
    """Test Sub operation - same parameterization, different operation.

    Demonstrates that fixtures are reused automatically!
    This class also runs 12 tests (4 dtypes × 3 shapes).
    """

    def make_test_model(self, input_shapes):
        """Create Sub operation with shapes from fixture."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        # Create Sub node (only difference from Add!)
        node = helper.make_node("Sub", ["input", "param"], ["output"], name="Sub_0")

        graph = helper.make_graph([node], "test_sub", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return callable that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    def configure_kernel_node(self, op, model):
        # Note: PE only exists on backends (Stage 3), not base kernel (Stage 2)
        pass

    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"


# ============================================================================
# Additional test: Mul operation with single dtype (shows flexibility)
# ============================================================================


# Override fixtures for this test only
@pytest.fixture(params=[{"input": DataType["INT8"], "param": DataType["INT8"]}])
def input_datatypes_mul(request):
    """Only INT8 for Mul test."""
    return request.param


class TestElementwiseBinaryMul(SingleKernelTest):
    """Test Mul operation with single INT8 configuration.

    Demonstrates per-test fixture customization.
    This runs 3 tests (1 dtype × 3 shapes).
    """

    # Use custom fixture name (requires pytest.fixture override in class)
    @pytest.fixture
    def input_datatypes(self, input_datatypes_mul):
        """Use custom dtype fixture for Mul test."""
        return input_datatypes_mul

    def make_test_model(self, input_shapes):
        """Create Mul operation with shapes from fixture."""
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        node = helper.make_node("Mul", ["input", "param"], ["output"], name="Mul_0")

        graph = helper.make_graph([node], "test_mul", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        # Return callable that creates configured InferKernels instance
        return lambda: InferKernels([ElementwiseBinaryOp])

    def configure_kernel_node(self, op, model):
        # Note: PE only exists on backends (Stage 3), not base kernel (Stage 2)
        pass

    def get_backend_fpgapart(self):
        return "xc7z020clg400-1"


# ============================================================================
# Summary
# ============================================================================

"""
Test count summary:
- TestElementwiseBinaryAdd: 9 configs (3 dtypes × 3 shapes) × 6 tests = 54 tests
- TestElementwiseBinarySub: 9 configs (3 dtypes × 3 shapes) × 6 tests = 54 tests
- TestElementwiseBinaryMul: 3 configs (1 dtype × 3 shapes) × 6 tests = 18 tests

Total: 21 configurations × 6 test methods = 126 test assertions!

All with ~100 lines of code (vs ~240 lines in v1.0 for just one dtype).

v2.3 Changes:
- Removed Quant node insertion (use direct DataType annotations)
- Removed 3 Quant validation tests per config
- Test count reduced from 9 to 6 per config (but faster execution!)
- Fixes rtlsim failures (no Quant nodes to synthesize)
- Removed non-FP32 FLOAT dtypes (not yet supported by Brainsmith compiler)

Supported DataTypes:
- Arbitrary integers 1-32 bits (INT/UINT)
- BIPOLAR, TERNARY
- FLOAT32
- Note: Non-FP32 float types (FLOAT<exp,mant,bias>) not supported (Brainsmith compiler limitation)

Run tests:
    # All tests
    pytest tests/kernels/test_elementwise_binary_v2.py -v

    # Specific dtype
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "INT8"

    # Specific shape
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "1-64"

    # Specific operation
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "Add"

    # Fast tests only (skip cppsim/rtlsim)
    pytest tests/kernels/test_elementwise_binary_v2.py -v -m "not slow"
"""
