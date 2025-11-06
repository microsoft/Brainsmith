"""Integration test example using v2.5 unified configuration.

This test demonstrates the v2.5 framework with KernelTestConfig:
- Single unified configuration object (KernelTestConfig)
- Declarative test parameterization via fixtures
- Semantic DSE API (with_input_stream for PE)
- Stage 2 stream dimension configuration
- Type-safe with full IDE autocomplete

Compare to v2.4 (test_elementwise_binary_v2.py):
- v2.4: 6+ scattered fixtures/methods per test case
- v2.5: 1 unified KernelTestConfig per test case
- v2.4: Generic with_dimension("PE", 8) API
- v2.5: Semantic with_input_stream(0, 8) API
- v2.4: Stage 3 only for stream dims (WRONG)
- v2.5: Stage 2+ for stream dims (CORRECT)

Result: Cleaner, more maintainable tests with better type safety!
"""

import pytest
import onnx.helper as helper
from onnx import TensorProto
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model

from tests.frameworks.single_kernel_test_v2 import SingleKernelTest
from tests.frameworks.test_config import KernelTestConfig


# ============================================================================
# v2.5 Unified Configuration Fixture
# ============================================================================


@pytest.fixture(
    params=[
        # Minimal config (Python-only testing)
        KernelTestConfig(
            test_id="add_int8_small",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        ),
        # Config with stream dimension (Stage 2+)
        KernelTestConfig(
            test_id="add_int8_small_pe8",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            input_streams={0: 8},  # PE=8 via semantic API (Stage 2+)
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        ),
        # Full config with backend testing
        KernelTestConfig(
            test_id="add_int16_large_pe16_backend",
            operation="Add",
            input_shapes={"input": (4, 128), "param": (4, 128)},
            input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
            input_streams={0: 16},  # PE=16 via semantic API
            fpgapart="xc7z020clg400-1",  # Enable backend testing
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
            tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
        ),
    ],
    ids=lambda cfg: cfg.test_id,
)
def kernel_test_config(request):
    """Unified test configuration (v2.5).

    Returns:
        KernelTestConfig: All test parameters in single object

    Type Safety:
        IDE provides autocomplete for all fields:
        - config.input_shapes
        - config.input_streams
        - config.fpgapart
        - config.tolerance_python
        - etc.

    Parameterization:
        Each config becomes a separate test case.
        Test IDs auto-generated from config (or explicit test_id field).
    """
    return request.param


# ============================================================================
# v2.5 Test Class - Minimal Implementation!
# ============================================================================


class TestElementwiseBinaryAddV2_5(SingleKernelTest):
    """Test Add operation with v2.5 unified configuration.

    Runs 18 tests (3 configs × 6 test types) automatically!

    Each configuration validates:
    1. Pipeline creates correct HW node
    2. Shapes preserved through pipeline
    3. Datatypes preserved through pipeline
    4. Python execution matches QONNX golden
    5. HLS cppsim matches QONNX golden (if backend configured)
    6. RTL rtlsim matches QONNX golden (if backend configured)

    v2.5 Features Demonstrated:
    - Single KernelTestConfig consolidates all parameters
    - Semantic with_input_stream(0, 8) API for PE
    - Stage 2 stream dimension configuration
    - Declarative fixture-based configuration
    - Framework auto-applies config (no manual configure_parameters!)

    Compare to v2.4:
    - v2.4: ~100 lines with scattered fixtures
    - v2.5: ~40 lines with unified config
    - v2.4: Manual configure_parameters() required
    - v2.5: Auto-configuration from fixture!
    """

    def make_test_model(self, input_shapes):
        """Create Add operation with shapes from fixture.

        Args:
            input_shapes: Dict extracted from kernel_test_config via pytest composition

        Returns:
            (model, input_names): Model and list of inputs to annotate

        Note:
            Pytest composition extracts input_shapes from kernel_test_config automatically.
            No manual unpacking needed - just use input_shapes directly!
        """
        # Create ONNX model (input_shapes comes from pytest fixture composition)
        inp = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shapes["input"])
        param = helper.make_tensor_value_info("param", TensorProto.FLOAT, input_shapes["param"])
        out = helper.make_tensor_value_info("output", TensorProto.FLOAT, input_shapes["input"])

        node = helper.make_node("Add", ["input", "param"], ["output"], name="Add_0")

        graph = helper.make_graph([node], "test_add", [inp, param], [out])
        model = ModelWrapper(qonnx_make_model(graph))

        # Framework auto-annotates inputs with DataTypes from kernel_test_config!
        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        """Use InferKernels to convert Add → ElementwiseBinaryOp."""
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        return lambda: InferKernels([ElementwiseBinaryOp])

    # ========================================================================
    # NO configure_parameters() needed! (v2.5 auto-configuration)
    # ========================================================================
    # The framework auto-applies kernel_test_config:
    # - input_streams → with_input_stream(0, 8) at Stage 2
    # - fpgapart → backend testing enabled
    # - tolerances → used by validation methods
    #
    # Compare to v2.4 where you needed:
    # def configure_parameters(self, op, model, stage):
    #     if stage == 3:  # WRONG! Should be Stage 2
    #         self.set_dse_param(op, model, "PE", 8)  # Wrong API
    # ========================================================================

    # ========================================================================
    # Optional: Override helper methods to use config values
    # ========================================================================
    # With v2.5, tolerances and backend config come from kernel_test_config.
    # Override these methods to extract from the config via pytest composition.

    def get_tolerance_python(self, kernel_test_config):
        """Extract Python tolerance from config.

        Args:
            kernel_test_config: Injected by pytest fixture composition

        Returns:
            Tolerance dict from config, or default if not specified
        """
        return kernel_test_config.tolerance_python or {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self, kernel_test_config):
        """Extract C++ sim tolerance from config.

        Args:
            kernel_test_config: Injected by pytest fixture composition

        Returns:
            Tolerance dict from config, or default if not specified
        """
        return kernel_test_config.tolerance_cppsim or {"rtol": 1e-5, "atol": 1e-6}

    def get_backend_fpgapart(self, kernel_test_config):
        """Extract backend FPGA part from config.

        Args:
            kernel_test_config: Injected by pytest fixture composition

        Returns:
            FPGA part string from config, or None if not specified
        """
        return kernel_test_config.fpgapart


# ============================================================================
# Summary
# ============================================================================

"""
Test count summary:
- TestElementwiseBinaryAddV2_5: 3 configs × 6 tests = 18 tests

All with ~40 lines of code (vs ~100 lines in v2.4)!

v2.5 Improvements:
- Single unified KernelTestConfig (vs 6+ scattered fixtures/methods)
- Semantic with_input_stream() API (vs generic with_dimension())
- Stage 2 stream configuration (vs incorrect Stage 3 only)
- Declarative fixture-based config (vs imperative configure_parameters())
- Type-safe with IDE autocomplete (vs dict-based fixtures)
- Auto-configuration by framework (vs manual application)

Run tests:
    # All tests
    pytest tests/kernels/test_elementwise_binary_v2_5_example.py -v

    # Specific config
    pytest tests/kernels/test_elementwise_binary_v2_5_example.py -v -k "pe8"

    # Fast tests only (skip cppsim/rtlsim)
    pytest tests/kernels/test_elementwise_binary_v2_5_example.py -v -m "not slow"

    # Unit tests for KernelTestConfig
    pytest tests/frameworks/test_kernel_test_config.py -v
"""
