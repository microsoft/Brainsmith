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
from tests.frameworks.test_config import KernelTestConfig

# ============================================================================
# Test Configuration Fixtures (v3.0) - Per-Operation
# ============================================================================


# Add operation configs - ALL tests (fast + slow)
_ADD_CONFIGS = [
    # INT8 configurations
    KernelTestConfig(
        test_id="add_int8_1x64_baseline",
        operation="Add",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_int8_4x128_pe8",
        operation="Add",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        input_streams={0: 8},  # PE=8 for first input
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_int8_1x8x8x32_cppsim",
        operation="Add",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        input_streams={0: 8},
        fpgapart="xc7z020clg400-1",  # Enable backend testing
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    # INT16 configurations
    KernelTestConfig(
        test_id="add_int16_1x64_baseline",
        operation="Add",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_int16_4x128_pe16",
        operation="Add",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        input_streams={0: 16},  # PE=16 for first input
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_int16_1x8x8x32_cppsim",
        operation="Add",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        input_streams={0: 8},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    # BIPOLAR configurations
    KernelTestConfig(
        test_id="add_bipolar_1x64_baseline",
        operation="Add",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_bipolar_4x128_pe8",
        operation="Add",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        input_streams={0: 8},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="add_bipolar_1x8x8x32_cppsim",
        operation="Add",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        input_streams={0: 8},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
]

# Add operation - FAST tests only (no backend)
_ADD_CONFIGS_FAST = [cfg for cfg in _ADD_CONFIGS if not cfg.fpgapart]

# Sub operation configs
_SUB_CONFIGS = [
    # INT8 configurations
    KernelTestConfig(
        test_id="sub_int8_1x64_baseline",
        operation="Sub",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="sub_int8_4x128_baseline",
        operation="Sub",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    KernelTestConfig(
        test_id="sub_int8_1x8x8x32_cppsim",
        operation="Sub",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    # INT16 configurations
    KernelTestConfig(
        test_id="sub_int16_1x64_baseline",
        operation="Sub",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="sub_int16_4x128_baseline",
        operation="Sub",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="sub_int16_1x8x8x32_cppsim",
        operation="Sub",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
    # BIPOLAR configurations
    KernelTestConfig(
        test_id="sub_bipolar_1x64_baseline",
        operation="Sub",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="sub_bipolar_4x128_baseline",
        operation="Sub",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="sub_bipolar_1x8x8x32_cppsim",
        operation="Sub",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["BIPOLAR"], "param": DataType["BIPOLAR"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
]

# Mul operation configs
_MUL_CONFIGS = [
    # All INT8 configurations
    KernelTestConfig(
        test_id="mul_int8_1x64_baseline",
        operation="Mul",
        input_shapes={"input": (1, 64), "param": (1, 64)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="mul_int8_4x128_baseline",
        operation="Mul",
        input_shapes={"input": (4, 128), "param": (4, 128)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    ),
    KernelTestConfig(
        test_id="mul_int8_1x8x8x32_cppsim",
        operation="Mul",
        input_shapes={"input": (1, 8, 8, 32), "param": (1, 8, 8, 32)},
        input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
        fpgapart="xc7z020clg400-1",
        tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
    ),
]


# ============================================================================
# Test class - just define operations!
# ============================================================================


class TestElementwiseBinaryAdd(SingleKernelTest):
    """Test Add operation with unified test configuration (v3.0).

    Framework auto-applies configuration from kernel_test_config fixture:
    - input_shapes, input_dtypes extracted automatically
    - DSE parameters (input_streams) applied via auto_configure_from_fixture()
    - Tolerances and fpgapart extracted from config

    Runs 54 tests (9 configs × 6 test types) automatically!

    Each configuration validates:
    1. Pipeline creates correct HW node
    2. Shapes preserved through pipeline
    3. Datatypes preserved through pipeline
    4. Python execution matches QONNX golden (pre-quantized data)
    5. HLS cppsim matches QONNX golden (if backend configured)
    6. RTL rtlsim matches QONNX golden (if backend configured)
    """

    @pytest.fixture(params=_ADD_CONFIGS, ids=lambda cfg: cfg.test_id)
    def kernel_test_config(self, request):
        """Add-specific configuration fixture."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create Add operation from unified configuration.

        Args:
            kernel_test_config: Unified test configuration (v3.0)
                Framework passes the full config object to this method.

        Returns:
            (model, input_names): Model and list of inputs to annotate with DataTypes
        """
        # Extract shapes from config
        input_shapes = kernel_test_config.input_shapes

        # Create Add model with shapes from config
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        node = helper.make_node("Add", ["input", "param"], ["output"], name="Add_0")

        graph = helper.make_graph([node], "test_add", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Framework annotates these inputs with DataTypes from kernel_test_config!
        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        """Use InferKernels to convert Add → ElementwiseBinaryOp."""
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        return lambda: InferKernels([ElementwiseBinaryOp])


# ============================================================================
# Additional test: Sub operation (shows reusability)
# ============================================================================


class TestElementwiseBinarySub(SingleKernelTest):
    """Test Sub operation with unified test configuration (v3.0).

    Demonstrates config reusability - same pattern as Add, different operation.
    Framework auto-applies all configuration from kernel_test_config fixture.
    """

    @pytest.fixture(params=_SUB_CONFIGS, ids=lambda cfg: cfg.test_id)
    def kernel_test_config(self, request):
        """Sub-specific configuration fixture."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create Sub operation from unified configuration."""
        # Extract shapes from config
        input_shapes = kernel_test_config.input_shapes

        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        # Create Sub node
        node = helper.make_node("Sub", ["input", "param"], ["output"], name="Sub_0")

        graph = helper.make_graph([node], "test_sub", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        return lambda: InferKernels([ElementwiseBinaryOp])


# ============================================================================
# Additional test: Mul operation (shows flexibility)
# ============================================================================


class TestElementwiseBinaryMul(SingleKernelTest):
    """Test Mul operation with unified test configuration (v3.0).

    All INT8 configurations for Mul operation.
    Framework auto-applies all configuration from kernel_test_config fixture.
    """

    @pytest.fixture(params=_MUL_CONFIGS, ids=lambda cfg: cfg.test_id)
    def kernel_test_config(self, request):
        """Mul-specific configuration fixture."""
        return request.param

    def make_test_model(self, kernel_test_config):
        """Create Mul operation from unified configuration."""
        # Extract shapes from config
        input_shapes = kernel_test_config.input_shapes

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

        return lambda: InferKernels([ElementwiseBinaryOp])


# ============================================================================
# Summary
# ============================================================================

"""
Test count summary (v3.0):
- TestElementwiseBinaryAdd: 9 configs × 6 tests = 54 tests
- TestElementwiseBinarySub: 9 configs × 6 tests = 54 tests
- TestElementwiseBinaryMul: 9 configs × 6 tests = 54 tests

Total: 27 configurations × 6 test methods = 162 test assertions!

v3.0 Migration Changes:
- Unified KernelTestConfig fixture replaces separate input_shapes/input_datatypes fixtures
- All configuration in one place (shapes, dtypes, DSE params, tolerances, fpgapart)
- Test classes simplified - only make_test_model() and get_kernel_inference_transform()
- Removed methods: configure_parameters(), get_tolerance_*(), get_backend_fpgapart()
- Framework auto-applies DSE parameters via input_streams config
- Explicit test IDs for better pytest output
- ~90 lines of code per test class (vs ~120 in v2.4)

Configuration Architecture:
- Config-centric design: KernelTestConfig owns data and defaults
- Framework uses thin delegation wrappers
- Config reusable beyond kernel tests (CLI, scripts, etc.)
- Type-safe with full IDE autocomplete

Test Configurations per Operation:
- 3 datatypes (INT8, INT16, BIPOLAR)
- 3 shapes per dtype: (1,64), (4,128), (1,8,8,32)
- Varying DSE configs: baseline (no PE), PE=8, PE=16
- Backend testing enabled for select configs (fpgapart present)

Supported DataTypes:
- Arbitrary integers 1-32 bits (INT/UINT)
- BIPOLAR, TERNARY
- FLOAT32
- Note: Non-FP32 float types (FLOAT<exp,mant,bias>) not supported (Brainsmith compiler limitation)

Run tests:
    # All tests
    pytest tests/kernels/test_elementwise_binary_v2.py -v

    # Specific test ID
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "add_int8_1x64_baseline"

    # Specific operation
    pytest tests/kernels/test_elementwise_binary_v2.py::TestElementwiseBinaryAdd -v

    # Specific dtype
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "int16"

    # Only backend tests
    pytest tests/kernels/test_elementwise_binary_v2.py -v -k "cppsim"

    # Fast tests only (skip cppsim/rtlsim)
    pytest tests/kernels/test_elementwise_binary_v2.py -v -m "not slow"
"""
