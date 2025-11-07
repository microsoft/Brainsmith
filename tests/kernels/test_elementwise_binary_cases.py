"""Test cases for ElementwiseBinary kernel.

This file contains ONLY test data - the actual test logic lives in test_elementwise_binary_v2.py.

Architecture:
1. Reusable constants define dimensions (shapes, dtypes, designs)
2. Loops generate regular combinations programmatically (~27 cases)
3. Explicit functions handle special cases (broadcasting, mixed dtypes) (~7 cases)
4. @pytest.mark.basic_validation marks representative cases

Benefits:
- 90% less code duplication (constants + loops)
- Easy to add new dimensions (just add to list)
- Special cases remain clear and debuggable
- Two selection modes: basic_validation vs full_sweep

Selection:
- pytest -m basic_validation     # Quick validation (~108 tests)
- pytest                          # Full sweep (612 tests)
"""

import pytest
from qonnx.core.datatype import DataType

from tests.frameworks.test_config import (
    DesignParameters,
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
    ValidationConfig,
)

# ============================================================================
# Constants: Shapes
# ============================================================================

# Basic shapes (matching inputs)
SHAPE_2D_1x64 = {"lhs": (1, 64), "rhs": (1, 64)}
SHAPE_2D_4x128 = {"lhs": (4, 128), "rhs": (4, 128)}
SHAPE_4D_1x8x8x32 = {"lhs": (1, 8, 8, 32), "rhs": (1, 8, 8, 32)}

# Broadcasting shapes (special cases - not in regular grid)
SHAPE_BROADCAST_CHANNEL = {"lhs": (1, 8, 8, 32), "rhs": (32,)}  # 4D → 1D
SHAPE_BROADCAST_SCALAR = {"lhs": (1, 8, 8, 32), "rhs": (1,)}  # 4D → scalar
SHAPE_BROADCAST_SPATIAL = {"lhs": (1, 8, 8, 32), "rhs": (1, 1, 1, 32)}  # 4D → 4D with 1s
SHAPE_BROADCAST_BIDIR = {"lhs": (1, 8, 1, 32), "rhs": (1, 1, 8, 1)}  # Bidirectional
SHAPE_BROADCAST_RANK = {"lhs": (4, 8, 16), "rhs": (16,)}  # Rank mismatch

# ============================================================================
# Constants: DataTypes
# ============================================================================

# Regular dtypes (all supported by HLS for basic arithmetic)
DTYPES_INT8 = {"lhs": DataType["INT8"], "rhs": DataType["INT8"]}
DTYPES_INT16 = {"lhs": DataType["INT16"], "rhs": DataType["INT16"]}
DTYPES_INT32 = {"lhs": DataType["INT32"], "rhs": DataType["INT32"]}
DTYPES_UINT8 = {"lhs": DataType["UINT8"], "rhs": DataType["UINT8"]}
DTYPES_UINT16 = {"lhs": DataType["UINT16"], "rhs": DataType["UINT16"]}
DTYPES_UINT32 = {"lhs": DataType["UINT32"], "rhs": DataType["UINT32"]}
DTYPES_BIPOLAR = {"lhs": DataType["BIPOLAR"], "rhs": DataType["BIPOLAR"]}
DTYPES_FLOAT32 = {"lhs": DataType["FLOAT32"], "rhs": DataType["FLOAT32"]}

# Mixed dtypes (special cases - not in regular grid)
DTYPES_MIXED_SIGN = {"lhs": DataType["INT8"], "rhs": DataType["UINT8"]}
DTYPES_MIXED_WIDTH = {"lhs": DataType["INT8"], "rhs": DataType["INT16"]}
DTYPES_MIXED_WIDTH_UNSIGNED = {"lhs": DataType["UINT8"], "rhs": DataType["UINT16"]}
DTYPES_MIXED_BOTH = {"lhs": DataType["INT8"], "rhs": DataType["UINT16"]}

# ============================================================================
# Constants: Platform & Validation
# ============================================================================

ZYNQ_7020 = PlatformConfig(fpgapart="xc7z020clg400-1")

# Single validation config with ALL tolerances
# Framework checks platform.fpgapart to determine if cppsim should run
VALIDATION_CONFIG = ValidationConfig(
    tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
)

# ============================================================================
# Constants: Design Parameters
# ============================================================================

DESIGN_BASELINE = DesignParameters()
DESIGN_PE8 = DesignParameters(input_streams={0: 8})
DESIGN_PE16 = DesignParameters(input_streams={0: 16})


# ============================================================================
# Helper Function: Composable Case Builder
# ============================================================================


def make_case(
    test_id: str,
    shapes: dict,
    dtypes: dict,
    design: DesignParameters = DESIGN_BASELINE,
    platform: PlatformConfig = ZYNQ_7020,
) -> KernelTestConfig:
    """Build a test case from reusable components.

    Args:
        test_id: Unique test identifier
        shapes: Input shapes dict (lhs/rhs)
        dtypes: Input dtypes dict (lhs/rhs)
        design: Design parameters (default: baseline)
        platform: Platform config (default: ZYNQ 7020, enables cppsim)

    Returns:
        KernelTestConfig ready for testing
    """
    return KernelTestConfig(
        test_id=test_id,
        model=ModelStructure(
            operation="placeholder",  # Set by test fixture
            input_shapes=shapes,
            input_dtypes=dtypes,
        ),
        design=design,
        platform=platform,
        validation=VALIDATION_CONFIG,
    )


# ============================================================================
# Programmatic Generation: Regular Combinations
# ============================================================================
# Generate regular grid of (dtype × shape × design) combinations
#
# Dimensions:
# - Dtypes: INT8, INT16, INT32, UINT8, UINT16, UINT32, BIPOLAR, FLOAT32 (8)
# - Shapes: 1x64, 4x128, 1x8x8x32 (3)
# - Designs: baseline, pe8, pe16 (3)
#
# Total regular cases: 8 dtypes × 3 shapes + 2 pe16 variants = ~26 cases
# All regular cases get ZYNQ_7020 platform → cppsim tests will run
# Only special cases (broadcasting, mixed dtypes) skip cppsim

# All dtype combinations - HLS supports all basic integer and float types
ALL_DTYPE_COMBINATIONS = [
    ("int8", DTYPES_INT8),
    ("int16", DTYPES_INT16),
    ("int32", DTYPES_INT32),
    ("uint8", DTYPES_UINT8),
    ("uint16", DTYPES_UINT16),
    ("uint32", DTYPES_UINT32),
    ("bipolar", DTYPES_BIPOLAR),
    ("float32", DTYPES_FLOAT32),
]

# Shape variations
SHAPE_VARIATIONS = [
    ("1x64_baseline", SHAPE_2D_1x64, DESIGN_BASELINE),
    ("4x128_pe8", SHAPE_2D_4x128, DESIGN_PE8),
    ("1x8x8x32_cppsim", SHAPE_4D_1x8x8x32, DESIGN_PE8),
]

# Also generate pe16 variant for int16/uint16
PE16_SHAPE = ("4x128_pe16", SHAPE_2D_4x128, DESIGN_PE16)


# Generate all regular dtype × shape combinations
# All get ZYNQ_7020 platform → cppsim will run for all
for dtype_name, dtype_const in ALL_DTYPE_COMBINATIONS:
    for shape_name, shape_const, design in SHAPE_VARIATIONS:
        test_id = f"{dtype_name}_{shape_name}"

        # Create the case function dynamically with proper closure
        def _make_case_func(tid, sc, dc, des, dn, sn, p=ZYNQ_7020):
            def case_func():
                return make_case(tid, sc, dc, des, p)

            case_func.__name__ = f"case_{tid}"
            case_func.__doc__ = f"{dn.upper()} with shape {sn}"
            return case_func

        func = _make_case_func(test_id, shape_const, dtype_const, design, dtype_name, shape_name)

        # Mark representative cases as basic_validation
        if dtype_name == "int8" and shape_name == "1x64_baseline":
            func = pytest.mark.basic_validation(func)
        elif dtype_name == "uint8" and shape_name == "1x64_baseline":
            func = pytest.mark.basic_validation(func)
        elif dtype_name == "float32" and shape_name == "1x64_baseline":
            func = pytest.mark.basic_validation(func)

        # Add to module globals so pytest can discover it
        globals()[f"case_{test_id}"] = func

    # Add pe16 variant for int16/uint16 (types that commonly use 16-bit parallelization)
    if dtype_name in ["int16", "uint16"]:
        test_id = f"{dtype_name}_4x128_pe16"

        def _make_pe16_func(tid, sc, dc, des, dn, p=ZYNQ_7020):
            def case_func():
                return make_case(tid, sc, dc, des, p)

            case_func.__name__ = f"case_{tid}"
            case_func.__doc__ = f"{dn.upper()} with PE=16"
            return case_func

        func = _make_pe16_func(
            test_id, PE16_SHAPE[1], dtype_const, DESIGN_PE16, dtype_name, ZYNQ_7020
        )
        globals()[f"case_{test_id}"] = func


# ============================================================================
# Explicit Case Functions: Special Cases (Broadcasting)
# ============================================================================


@pytest.mark.basic_validation
def case_int8_1x64x64x128_channel_broadcast():
    """INT8 channel broadcast - representative broadcast pattern."""
    return make_case(
        "int8_1x64x64x128_channel_broadcast",
        SHAPE_BROADCAST_CHANNEL,
        DTYPES_INT8,
        platform=PlatformConfig(),  # Broadcasting not backend-ready yet
    )


def case_int8_1x64x64x128_scalar_broadcast():
    """INT8 scalar broadcast."""
    return make_case(
        "int8_1x64x64x128_scalar_broadcast",
        SHAPE_BROADCAST_SCALAR,
        DTYPES_INT8,
        platform=PlatformConfig(),
    )


def case_int8_1x64x64x128_spatial_broadcast():
    """INT8 spatial broadcast."""
    return make_case(
        "int8_1x64x64x128_spatial_broadcast",
        SHAPE_BROADCAST_SPATIAL,
        DTYPES_INT8,
        design=DESIGN_PE8,
        platform=PlatformConfig(),
    )


def case_int8_1x64x1x128_bidirectional_broadcast():
    """INT8 bidirectional broadcast."""
    return make_case(
        "int8_1x64x1x128_bidirectional_broadcast",
        SHAPE_BROADCAST_BIDIR,
        DTYPES_INT8,
        platform=PlatformConfig(),
    )


def case_int8_4x8x16_rank_mismatch():
    """INT8 rank mismatch broadcast."""
    return make_case(
        "int8_4x8x16_rank_mismatch",
        SHAPE_BROADCAST_RANK,
        DTYPES_INT8,
        platform=PlatformConfig(),
    )


def case_float32_1x64x64x128_channel_broadcast():
    """FLOAT32 channel broadcast."""
    return make_case(
        "float32_1x64x64x128_channel_broadcast",
        SHAPE_BROADCAST_CHANNEL,
        DTYPES_FLOAT32,
        platform=PlatformConfig(),
    )


def case_float32_1x64x64x128_scalar_broadcast():
    """FLOAT32 scalar broadcast."""
    return make_case(
        "float32_1x64x64x128_scalar_broadcast",
        SHAPE_BROADCAST_SCALAR,
        DTYPES_FLOAT32,
        platform=PlatformConfig(),
    )


def case_uint8_1x64x64x128_channel_broadcast():
    """UINT8 channel broadcast (unsigned overflow behavior)."""
    return make_case(
        "uint8_1x64x64x128_channel_broadcast",
        SHAPE_BROADCAST_CHANNEL,
        DTYPES_UINT8,
        platform=PlatformConfig(),
    )


# ============================================================================
# Explicit Case Functions: Special Cases (Mixed DataTypes)
# ============================================================================


def case_mixed_mixed_sign():
    """Mixed signedness: INT8 + UINT8."""
    return make_case(
        "mixed_mixed_sign",
        SHAPE_2D_1x64,
        DTYPES_MIXED_SIGN,
        platform=PlatformConfig(),
    )


def case_mixed_mixed_width():
    """Mixed width: INT8 + INT16."""
    return make_case(
        "mixed_mixed_width",
        SHAPE_2D_1x64,
        DTYPES_MIXED_WIDTH,
        platform=PlatformConfig(),
    )


def case_mixed_mixed_width_unsigned():
    """Mixed width unsigned: UINT8 + UINT16."""
    return make_case(
        "mixed_mixed_width_unsigned",
        SHAPE_2D_1x64,
        DTYPES_MIXED_WIDTH_UNSIGNED,
        platform=PlatformConfig(),
    )


def case_mixed_mixed_both():
    """Mixed sign and width: INT8 + UINT16."""
    return make_case(
        "mixed_mixed_both",
        SHAPE_2D_1x64,
        DTYPES_MIXED_BOTH,
        platform=PlatformConfig(),
    )
