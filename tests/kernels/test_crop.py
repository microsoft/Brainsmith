"""Validation tests for Crop kernel operation.

Tactical validation tests covering representative cases for CI/CD:
- Core datatypes (INT4, UINT4, INT8, UINT8, INT16, UINT16)
- SIMD parallelization (SIMD=1, 8, 16)
- Edge cases (zero crop, minimal output, maximal crop)
- Spatial dimension variations (square, wide, tall)

Test count: ~15 cases × 6 inherited tests = ~90 tests

Usage:
    pytest tests/kernels/crop/test_crop_validation.py -v
    pytest -m validation -k crop
"""

import numpy as np
import pytest
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model
from tests.frameworks.kernel_test import KernelTest
from tests.frameworks.test_config import (
    DesignParameters,
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
    ValidationConfig,
)

# ============================================================================
# Shared Constants
# ============================================================================

# Platform configuration
PLATFORM_ZYNQ7020 = PlatformConfig(fpgapart="xc7z020clg400-1")

# Design parameters
DESIGN_BASELINE = DesignParameters()  # SIMD=1, channel_fold=1
DESIGN_SIMD8 = DesignParameters(dimensions={"SIMD": 8})
DESIGN_SIMD16 = DesignParameters(dimensions={"SIMD": 16})


# ============================================================================
# Test Case Definitions
# ============================================================================

# Format: (test_id, input_shape, crop_params, dtype, design, platform)
# crop_params: (axis, start_idx, end_idx) -> will generate Gather indices
#   axis=1 (height): computes crop_north, crop_south
#   axis=2 (width): computes crop_east, crop_west

VALIDATION_CASES = [
    # ========================================================================
    # Core Datatype Coverage - Height Axis Cropping
    # ========================================================================
    ("crop_height_int4", (1, 16, 16, 32), (1, 2, 14), DataType["INT4"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_height_uint4", (1, 16, 16, 32), (1, 2, 14), DataType["UINT4"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_height_int8", (1, 28, 28, 64), (1, 4, 24), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_height_uint8", (1, 28, 28, 64), (1, 4, 24), DataType["UINT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_height_int16", (1, 32, 32, 128), (1, 6, 26), DataType["INT16"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_height_uint16", (1, 32, 32, 128), (1, 6, 26), DataType["UINT16"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),

    # ========================================================================
    # Core Datatype Coverage - Width Axis Cropping
    # ========================================================================
    ("crop_width_int8", (1, 28, 28, 64), (2, 4, 24), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("crop_width_uint8", (1, 28, 28, 64), (2, 4, 24), DataType["UINT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),

    # ========================================================================
    # SIMD Parallelization - Height Cropping with Different SIMD Values
    # ========================================================================
    ("crop_height_simd8", (1, 28, 28, 64), (1, 4, 24), DataType["INT8"], DESIGN_SIMD8, PLATFORM_ZYNQ7020),
    ("crop_height_simd16", (1, 32, 32, 64), (1, 4, 28), DataType["INT8"], DESIGN_SIMD16, PLATFORM_ZYNQ7020),

    # ========================================================================
    # Edge Cases
    # ========================================================================
    # Zero crop (passthrough)
    ("crop_zero", (1, 16, 16, 32), (1, 0, 16), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # Minimal output (crop to 2×W)
    ("crop_minimal_height", (1, 16, 16, 32), (1, 7, 9), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # Maximal crop (one edge only)
    ("crop_maximal_north", (1, 28, 28, 64), (1, 20, 28), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),

    # ========================================================================
    # Spatial Dimension Variations
    # ========================================================================
    # Wide aspect ratio (H < W)
    ("crop_wide", (1, 16, 64, 64), (2, 8, 56), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # Tall aspect ratio (H > W)
    ("crop_tall", (1, 64, 16, 64), (1, 8, 56), DataType["INT8"], DESIGN_BASELINE, PLATFORM_ZYNQ7020),
]


# ============================================================================
# Helper Functions
# ============================================================================

def make_crop_test_config(
    test_id: str,
    input_shape: tuple,
    crop_params: tuple,
    input_dtype: DataType,
    design: DesignParameters,
    platform: PlatformConfig,
) -> KernelTestConfig:
    """Build a Crop test configuration.

    Args:
        test_id: Unique test identifier
        input_shape: NHWC input shape (N, H, W, C)
        crop_params: (axis, start_idx, end_idx) for Gather indices
        input_dtype: Input datatype
        design: Design parameters (SIMD, etc.)
        platform: Platform configuration

    Returns:
        KernelTestConfig with Gather operation (will be transformed to Crop)
    """
    axis, start_idx, end_idx = crop_params

    # Validate axis
    if axis not in [1, 2]:
        raise ValueError(f"axis must be 1 (height) or 2 (width), got {axis}")

    # Store crop_params in config for use in make_test_model
    # Note: indices is an initializer (static), not a graph input, so it's not in ModelStructure
    return KernelTestConfig(
        test_id=test_id,
        model=ModelStructure(
            operation="Gather",  # Will be transformed to Crop by Crop.infer_from()
            input_shapes={"input": input_shape},
            input_dtypes={"input": input_dtype}
        ),
        design=design,
        platform=platform,
        validation=ValidationConfig(),
        marks=[pytest.mark.validation],
    )


# Build test configurations
TEST_CONFIGS = [
    make_crop_test_config(test_id, shape, crop_params, dtype, design, platform)
    for test_id, shape, crop_params, dtype, design, platform in VALIDATION_CASES
]


# ============================================================================
# Test Configuration Fixture
# ============================================================================

@pytest.fixture(
    params=[
        pytest.param(config, marks=config.marks, id=config.test_id)
        for config in TEST_CONFIGS
    ]
)
def kernel_test_config(request) -> KernelTestConfig:
    """Parameterized test configuration for Crop validation cases.

    Yields one test configuration per validation case.
    All cases marked with @pytest.mark.validation.
    """
    return request.param


# ============================================================================
# Test Class
# ============================================================================

class TestCropValidation(KernelTest):
    """Validation tests for Crop kernel operation.

    Inherits implementation from KernelTest base class.

    Inherits 6 tests automatically from KernelTest:
    1. test_stage1_model_structure - Validates ONNX Gather model structure
    2. test_stage2_kernel_inference - Validates Gather → Crop transformation
    3. test_python_execution_vs_golden - Python execution parity
    4. test_stage3_backend_specialization - Backend specialization validation
    5. test_cppsim_execution_vs_golden - HLS C++ simulation parity
    6. test_rtlsim_execution_vs_golden - RTL simulation parity

    All tests use session-scoped model_cache for computational reuse.
    """

    # ========================================================================
    # Required KernelTest Methods
    # ========================================================================

    def make_test_model(self, kernel_test_config: KernelTestConfig):
        """Create ONNX Gather model that will be transformed to Crop.

        The Crop kernel infers from Gather nodes with consecutive indices.
        This method creates:
        1. Input tensor (dynamic)
        2. Indices initializer (static, consecutive range)
        3. Gather node with axis attribute

        Crop.infer_from() will transform this to Crop node with computed
        crop_north/south/east/west parameters.

        Args:
            kernel_test_config: Test configuration with shapes, dtypes, crop params

        Returns:
            (model, input_names): ONNX model and list of input tensor names
        """
        # Extract configuration
        input_shape = kernel_test_config.input_shapes["input"]

        # Find the matching test case to get crop_params
        # Convention: axis and crop range are encoded in VALIDATION_CASES
        test_id = kernel_test_config.test_id
        crop_params = None
        for tid, shape, cparams, dtype, design, platform in VALIDATION_CASES:
            if tid == test_id:
                crop_params = cparams
                break

        if crop_params is None:
            raise RuntimeError(f"Could not find crop_params for test_id={test_id}")

        axis, start_idx, end_idx = crop_params

        # Create consecutive indices array
        indices_array = np.arange(start_idx, end_idx, dtype=np.int64)
        indices_shape = (end_idx - start_idx,)

        # Create ONNX graph
        # Input tensor (dynamic)
        input_tensor = helper.make_tensor_value_info(
            "input",
            TensorProto.FLOAT,  # QONNX convention: container type is FLOAT
            list(input_shape)
        )

        # Output tensor (shape will be inferred)
        output_shape = list(input_shape)
        output_shape[axis] = end_idx - start_idx  # Cropped dimension
        output_tensor = helper.make_tensor_value_info(
            "output",
            TensorProto.FLOAT,
            output_shape
        )

        # Indices initializer (static)
        indices_init = helper.make_tensor(
            "indices",
            TensorProto.INT64,
            list(indices_shape),
            indices_array.flatten().tolist()
        )

        # Gather node
        gather_node = helper.make_node(
            "Gather",
            inputs=["input", "indices"],
            outputs=["output"],
            name="Gather_0",
            axis=axis
        )

        # Create graph
        graph = helper.make_graph(
            nodes=[gather_node],
            name="test_crop_gather",
            inputs=[input_tensor],
            outputs=[output_tensor],
            initializer=[indices_init]
        )

        model = ModelWrapper(qonnx_make_model(graph))

        # Return model and input names (only "input" - "indices" is initializer)
        return model, ["input"]

    def get_kernel_op(self):
        """Return Crop kernel operator class.

        Returns:
            Crop class (KernelOp)
        """
        from brainsmith.kernels.crop import Crop
        return Crop

    def get_num_inputs(self):
        """Crop has 1 input (data tensor).

        Note: Gather has 2 inputs (data + indices), but Crop.infer_from()
        transforms this to 1 input (indices are consumed during transformation).
        """
        return 1

    def get_num_outputs(self):
        """Crop has 1 output."""
        return 1

    # ========================================================================
    # Optional Overrides (if needed for Crop-specific behavior)
    # ========================================================================

    # No overrides needed - default KernelTest behavior works for Crop
