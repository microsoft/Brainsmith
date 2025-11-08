"""Test infrastructure for ElementwiseBinary operations.

This package provides test base classes and shared test case definitions
for all elementwise binary operations (Add, Sub, Mul, Div).

Architecture:
- test_elementwise_binary.py: Base test class (ElementwiseBinaryTestBase)
- test_cases.py: Shared test case definitions and constants

Usage:
    from brainsmith.kernels.elementwise_binary.tests import (
        ElementwiseBinaryTestBase,
        VALIDATION_CASES_BASE,
        make_elementwise_case,
    )

    class TestAddValidation(ElementwiseBinaryTestBase):
        # Automatically inherits make_test_model() and get_kernel_op()
        pass
"""

# Re-export base test class
from .test_elementwise_binary import ElementwiseBinaryTestBase

# Re-export test case utilities
from .test_cases import (
    DESIGN_BASELINE,
    DESIGN_MEM_DECOUPLED,
    DESIGN_PE8,
    DESIGN_PE8_RAM_BLOCK,
    DESIGN_PE8_RAM_DISTRIBUTED,
    DESIGN_PE16,
    DESIGN_RAM_BLOCK,
    DESIGN_RAM_DISTRIBUTED,
    DESIGN_RAM_ULTRA,
    DTYPES_BINARY,
    DTYPES_BIPOLAR,
    DTYPES_FLOAT32,
    DTYPES_INT4,
    DTYPES_INT8,
    DTYPES_INT16,
    DTYPES_INT32,
    DTYPES_MIXED_SIGN,
    DTYPES_MIXED_WIDTH,
    DTYPES_UINT4,
    DTYPES_UINT8,
    DTYPES_UINT16,
    DTYPES_UINT32,
    PLATFORM_ZYNQ7020,
    SHAPE_2D_1x16,
    SHAPE_2D_1x64,
    SHAPE_2D_4x128,
    SHAPE_3D_1x16x64,
    SHAPE_4D_1x8x8x32,
    SHAPE_BROADCAST_BIDIR,
    SHAPE_BROADCAST_CHANNEL,
    SHAPE_BROADCAST_RANK,
    SHAPE_BROADCAST_SCALAR,
    SHAPE_BROADCAST_SPATIAL,
    VALIDATION_CASES_BASE,
    VALIDATION_STANDARD,
    make_elementwise_case,
)

__all__ = [
    # Base test class
    "ElementwiseBinaryTestBase",
    # Test case builder
    "make_elementwise_case",
    # Validation cases
    "VALIDATION_CASES_BASE",
    # Shape constants
    "SHAPE_2D_1x16",
    "SHAPE_2D_1x64",
    "SHAPE_2D_4x128",
    "SHAPE_3D_1x16x64",
    "SHAPE_4D_1x8x8x32",
    "SHAPE_BROADCAST_CHANNEL",
    "SHAPE_BROADCAST_SCALAR",
    "SHAPE_BROADCAST_SPATIAL",
    "SHAPE_BROADCAST_BIDIR",
    "SHAPE_BROADCAST_RANK",
    # Dtype constants
    "DTYPES_INT4",
    "DTYPES_UINT4",
    "DTYPES_INT8",
    "DTYPES_INT16",
    "DTYPES_INT32",
    "DTYPES_UINT8",
    "DTYPES_UINT16",
    "DTYPES_UINT32",
    "DTYPES_BINARY",
    "DTYPES_BIPOLAR",
    "DTYPES_FLOAT32",
    "DTYPES_MIXED_SIGN",
    "DTYPES_MIXED_WIDTH",
    # Platform constants
    "PLATFORM_ZYNQ7020",
    # Design constants
    "DESIGN_BASELINE",
    "DESIGN_PE8",
    "DESIGN_PE16",
    "DESIGN_RAM_DISTRIBUTED",
    "DESIGN_RAM_BLOCK",
    "DESIGN_RAM_ULTRA",
    "DESIGN_MEM_DECOUPLED",
    "DESIGN_PE8_RAM_DISTRIBUTED",
    "DESIGN_PE8_RAM_BLOCK",
    # Validation constants
    "VALIDATION_STANDARD",
]
