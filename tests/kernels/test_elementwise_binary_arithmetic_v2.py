"""ElementwiseBinaryOp arithmetic operations test (v2.5).

Tests arithmetic operations (Add, Sub, Mul, Div) using v2.5 unified configuration.

Phase 1a: Add operation only (3 configs, 18 tests)
  - Baseline: Python-only, no PE
  - PE=8: Stream parallelism at Stage 2
  - PE=16 + cppsim: Backend testing

Phase 1b: Expand to Sub, Mul, Div (~10 configs, ~60 tests)

Design:
  - Polymorphic make_test_model() uses config.operation
  - Single test class handles all arithmetic operations
  - Pytest fixture composition extracts parameters
  - Semantic DSE API (with_input_stream)
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


# =============================================================================
# Phase 1a: Add Operation Only (3 configs)
# =============================================================================

@pytest.fixture(
    params=[
        # Level 1: Baseline (Python-only, no PE)
        KernelTestConfig(
            test_id="add_int8_1x64_baseline",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        ),

        # Level 2: Add PE (Stage 2 stream config)
        KernelTestConfig(
            test_id="add_int8_1x64_pe8",
            operation="Add",
            input_shapes={"input": (1, 64), "param": (1, 64)},
            input_dtypes={"input": DataType["INT8"], "param": DataType["INT8"]},
            input_streams={0: 8},  # PE=8 via semantic API (Stage 2+)
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
        ),

        # Level 3: Backend testing (cppsim)
        KernelTestConfig(
            test_id="add_int16_4x128_pe16_cppsim",
            operation="Add",
            input_shapes={"input": (4, 128), "param": (4, 128)},
            input_dtypes={"input": DataType["INT16"], "param": DataType["INT16"]},
            input_streams={0: 16},  # PE=16
            fpgapart="xc7z020clg400-1",  # Enable cppsim
            tolerance_python={"rtol": 1e-7, "atol": 1e-9},
            tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
        ),
    ],
    ids=lambda cfg: cfg.test_id,
)
def kernel_test_config(request):
    """Unified test configuration for arithmetic operations (v2.5).

    Returns:
        KernelTestConfig: All test parameters in single object

    Test Generation:
        Each config generates 6 tests via SingleKernelTest:
        1. test_pipeline_creates_correct_hw_node
        2. test_pipeline_preserves_shapes
        3. test_pipeline_preserves_datatypes
        4. test_python_execution_matches_golden
        5. test_cppsim_matches_golden (if fpgapart)
        6. test_rtlsim_matches_golden (if fpgapart)

    Pytest Composition:
        Framework auto-extracts input_shapes and input_datatypes
        via fixture composition. Tests declare dependencies explicitly.
    """
    return request.param


# =============================================================================
# Test Class: Arithmetic Operations
# =============================================================================

class TestElementwiseBinaryArithmetic(SingleKernelTest):
    """Test arithmetic operations: Add, Sub, Mul, Div.

    Phase 1a Status:
        ✅ Add operation (3 configs, 18 tests)
        ⏸ Sub operation (Phase 1b)
        ⏸ Mul operation (Phase 1b)
        ⏸ Div operation (Phase 1b)

    Architecture:
        - Polymorphic make_test_model() uses config.operation
        - Single code path for all arithmetic operations
        - v2.5 pytest fixture composition (Arete)
        - Semantic DSE API (with_input_stream)

    Coverage (Phase 1a):
        • Operations: Add only
        • Data types: INT8, INT16
        • Shapes: 2D (1×64, 4×128)
        • PE values: None, 8, 16
        • Backends: Python, cppsim
    """

    # =========================================================================
    # Required Method Implementations
    # =========================================================================

    def make_test_model(self, input_shapes):
        """Create ONNX Add model with shapes from fixture.

        Phase 1a Implementation:
            Hardcoded to "Add" operation following v2.5 pattern.
            Framework doesn't pass kernel_test_config to make_test_model(),
            so polymorphic dispatch isn't possible without framework changes.

        Phase 1b Plan:
            For Sub/Mul/Div, options:
            1. Create separate test classes (TestAdd, TestSub, etc.)
            2. Enhance framework to pass config to make_test_model()
            3. Use class-level parametrization

        Args:
            input_shapes: Dict extracted from kernel_test_config via pytest composition
                {"input": (1, 64), "param": (1, 64)}

        Returns:
            (model, input_names): Tuple of ModelWrapper and list of inputs to annotate

        Pattern: dynamic_static
            - "input": Dynamic (streaming) tensor
            - "param": Static (initializer) parameter

        Note:
            Framework auto-annotates inputs with DataTypes from kernel_test_config.
            No manual annotation needed here.
        """
        # Create ONNX tensors
        inp = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,
            input_shapes["input"]
        )
        param = helper.make_tensor_value_info(
            "param",
            TensorProto.FLOAT,
            input_shapes["param"]
        )
        out = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            input_shapes["input"]  # Output shape matches input
        )

        # Create Add node (hardcoded for Phase 1a)
        node = helper.make_node(
            "Add",
            inputs=["input", "param"],
            outputs=["output"],
            name="Add_0"
        )

        # Build ONNX graph
        graph = helper.make_graph(
            nodes=[node],
            name="test_add",
            inputs=[inp, param],
            outputs=[out]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and list of inputs to annotate with datatypes
        return model, ["input", "param"]

    def get_kernel_inference_transform(self):
        """Transform to convert ONNX operation → ElementwiseBinaryOp.

        Uses InferKernels with ElementwiseBinaryOp to handle all
        arithmetic operations (Add, Sub, Mul, Div) in single transform.

        Returns:
            Transform factory function (lambda: InferKernels([...]))
        """
        from brainsmith.primitives.transforms.infer_kernels import InferKernels
        from brainsmith.kernels.elementwise_binary import ElementwiseBinaryOp

        return lambda: InferKernels([ElementwiseBinaryOp])

    # =========================================================================
    # Optional Overrides (Extract from Config)
    # =========================================================================

    def get_tolerance_python(self, kernel_test_config=None):
        """Extract Python execution tolerance from config.

        Args:
            kernel_test_config: Optional unified config (v2.5)
                - If provided: extract from config
                - If None: use default (backward compatible with v2.4)

        Returns:
            Tolerance dict from config, or default if not specified
        """
        if kernel_test_config and kernel_test_config.tolerance_python:
            return kernel_test_config.tolerance_python
        return {"rtol": 1e-7, "atol": 1e-9}

    def get_tolerance_cppsim(self, kernel_test_config=None):
        """Extract C++ simulation tolerance from config.

        Args:
            kernel_test_config: Optional unified config (v2.5)
                - If provided: extract from config
                - If None: use default (backward compatible with v2.4)

        Returns:
            Tolerance dict from config, or default if not specified
        """
        if kernel_test_config and kernel_test_config.tolerance_cppsim:
            return kernel_test_config.tolerance_cppsim
        return {"rtol": 1e-5, "atol": 1e-6}

    def get_backend_fpgapart(self, kernel_test_config=None):
        """Extract backend FPGA part from config.

        Controls whether cppsim/rtlsim tests run:
        - None: Python-only tests (skip backend)
        - "xc7z020clg400-1": Enable backend tests

        Args:
            kernel_test_config: Optional unified config (v2.5)
                - If provided: extract from config
                - If None: return None (backward compatible with v2.4)

        Returns:
            FPGA part string from config, or None if not specified
        """
        if kernel_test_config:
            return kernel_test_config.fpgapart
        return None


# =============================================================================
# Summary
# =============================================================================

"""
Phase 1a Test Count:
  • 3 configs × 6 tests = 18 tests total

  Breakdown per config:
    add_int8_1x64_baseline (6 tests):
      1. test_pipeline_creates_correct_hw_node ✓
      2. test_pipeline_preserves_shapes ✓
      3. test_pipeline_preserves_datatypes ✓
      4. test_python_execution_matches_golden ✓
      5. test_cppsim_matches_golden (skipped - no fpgapart)
      6. test_rtlsim_matches_golden (skipped - no fpgapart)

    add_int8_1x64_pe8 (6 tests):
      1-4: Same as baseline ✓
      5-6: Skipped (no backend)

    add_int16_4x128_pe16_cppsim (6 tests):
      1-4: Same as baseline ✓
      5. test_cppsim_matches_golden ✓ (fpgapart configured)
      6. test_rtlsim_matches_golden ✓ (fpgapart configured)

Run tests:
  # All Phase 1a tests
  pytest tests/kernels/test_elementwise_binary_arithmetic_v2.py -v

  # Specific config
  pytest tests/kernels/test_elementwise_binary_arithmetic_v2.py -v -k "baseline"
  pytest tests/kernels/test_elementwise_binary_arithmetic_v2.py -v -k "pe8"
  pytest tests/kernels/test_elementwise_binary_arithmetic_v2.py -v -k "cppsim"

  # Fast tests only (skip cppsim/rtlsim)
  pytest tests/kernels/test_elementwise_binary_arithmetic_v2.py -v -m "not slow"

Next Steps (Phase 1b):
  1. Add Sub configs (2 configs, ~12 tests)
  2. Add Mul configs (2 configs, ~12 tests)
  3. Add Div configs (3 configs, ~18 tests)
  4. Total Phase 1b: ~10 configs, ~60 tests
"""
