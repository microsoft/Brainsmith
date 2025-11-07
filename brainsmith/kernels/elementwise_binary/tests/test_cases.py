"""Shared test case definitions for ElementwiseBinary operations.

This module contains reusable constants and case builder functions for
elementwise binary operations (Add, Sub, Mul, Div).

Architecture:
1. Reusable constants for shapes, dtypes, designs, platforms
2. Explicit case lists (no programmatic generation)
3. Shared builder function for all operations
4. Clear separation: validation vs certification cases

Usage:
    from brainsmith.kernels.elementwise_binary.tests import (
        VALIDATION_CASES_BASE,
        make_elementwise_case
    )

    # Build operation-specific configs
    validation_cases = [
        make_elementwise_case(operation="Add", **case)
        for case in VALIDATION_CASES_BASE
    ]
"""

from qonnx.core.datatype import DataType

from tests.frameworks.test_config import (
    DesignParameters,
    KernelTestConfig,
    ModelStructure,
    PlatformConfig,
    ValidationConfig,
)

# ============================================================================
# Constants: Shapes (Matching Inputs)
# ============================================================================

SHAPE_2D_1x16 = {"lhs": (1, 16), "rhs": (1, 16)}  # Small for output width tests
SHAPE_2D_1x64 = {"lhs": (1, 64), "rhs": (1, 64)}
SHAPE_2D_4x128 = {"lhs": (4, 128), "rhs": (4, 128)}
SHAPE_3D_1x16x64 = {"lhs": (1, 16, 64), "rhs": (1, 16, 64)}  # Tests 3D path in defines()
SHAPE_4D_1x8x8x32 = {"lhs": (1, 8, 8, 32), "rhs": (1, 8, 8, 32)}

# ============================================================================
# Constants: Shapes (Broadcasting)
# ============================================================================

SHAPE_BROADCAST_CHANNEL = {"lhs": (1, 8, 8, 32), "rhs": (32,)}  # 4D → 1D
SHAPE_BROADCAST_SCALAR = {"lhs": (1, 8, 8, 32), "rhs": (1,)}  # 4D → scalar
SHAPE_BROADCAST_SPATIAL = {"lhs": (1, 8, 8, 32), "rhs": (1, 1, 1, 32)}  # 4D → 4D with 1s
SHAPE_BROADCAST_BIDIR = {"lhs": (1, 8, 1, 32), "rhs": (1, 1, 8, 1)}  # Bidirectional
SHAPE_BROADCAST_RANK = {"lhs": (4, 8, 16), "rhs": (16,)}  # Rank mismatch

# ============================================================================
# Constants: DataTypes (Regular)
# ============================================================================

# Narrow integer types (output width validation)
DTYPES_INT4 = {"lhs": DataType["INT4"], "rhs": DataType["INT4"]}
DTYPES_UINT4 = {"lhs": DataType["UINT4"], "rhs": DataType["UINT4"]}

# Standard integer types
DTYPES_INT8 = {"lhs": DataType["INT8"], "rhs": DataType["INT8"]}
DTYPES_INT16 = {"lhs": DataType["INT16"], "rhs": DataType["INT16"]}
DTYPES_INT32 = {"lhs": DataType["INT32"], "rhs": DataType["INT32"]}
DTYPES_UINT8 = {"lhs": DataType["UINT8"], "rhs": DataType["UINT8"]}
DTYPES_UINT16 = {"lhs": DataType["UINT16"], "rhs": DataType["UINT16"]}
DTYPES_UINT32 = {"lhs": DataType["UINT32"], "rhs": DataType["UINT32"]}

# Binary types
DTYPES_BINARY = {"lhs": DataType["BINARY"], "rhs": DataType["BINARY"]}
DTYPES_BIPOLAR = {"lhs": DataType["BIPOLAR"], "rhs": DataType["BIPOLAR"]}

# Float types
DTYPES_FLOAT32 = {"lhs": DataType["FLOAT32"], "rhs": DataType["FLOAT32"]}

# ============================================================================
# Constants: DataTypes (Mixed - Special Cases)
# ============================================================================

DTYPES_MIXED_SIGN = {"lhs": DataType["INT8"], "rhs": DataType["UINT8"]}
DTYPES_MIXED_WIDTH = {"lhs": DataType["INT8"], "rhs": DataType["INT16"]}

# ============================================================================
# Constants: Platform Configurations
# ============================================================================

# Use v5.0 PlatformConfig structure
PLATFORM_ZYNQ7020 = PlatformConfig(fpgapart="xc7z020clg400-1")
PLATFORM_SOFTWARE_ONLY = PlatformConfig()  # No fpgapart = software-only (no cppsim)

# ============================================================================
# Constants: Design Parameters
# ============================================================================

DESIGN_BASELINE = DesignParameters()
DESIGN_PE8 = DesignParameters(input_streams={0: 8})
DESIGN_PE16 = DesignParameters(input_streams={0: 16})

# DSE Dimension Variants (ram_style)
DESIGN_RAM_DISTRIBUTED = DesignParameters(dimensions={"ram_style": "distributed"})  # LUT-based
DESIGN_RAM_BLOCK = DesignParameters(dimensions={"ram_style": "block"})  # BRAM-based
DESIGN_RAM_ULTRA = DesignParameters(dimensions={"ram_style": "ultra"})  # UltraRAM (US+)

# DSE Dimension Variants (mem_mode)
DESIGN_MEM_DECOUPLED = DesignParameters(dimensions={"mem_mode": "internal_decoupled"})

# Combined DSE Variants
DESIGN_PE8_RAM_DISTRIBUTED = DesignParameters(
    input_streams={0: 8},
    dimensions={"ram_style": "distributed"}
)
DESIGN_PE8_RAM_BLOCK = DesignParameters(
    input_streams={0: 8},
    dimensions={"ram_style": "block"}
)

# ============================================================================
# Constants: Validation Configuration
# ============================================================================

VALIDATION_STANDARD = ValidationConfig(
    tolerance_python={"rtol": 1e-7, "atol": 1e-9},
    tolerance_cppsim={"rtol": 1e-5, "atol": 1e-6},
)

# ============================================================================
# Case Builder Function
# ============================================================================


def make_elementwise_case(
    test_id: str,
    operation: str,
    input_shapes: dict,
    input_dtypes: dict,
    design: DesignParameters = None,
    platform: PlatformConfig = None,
    validation: ValidationConfig = None,
) -> KernelTestConfig:
    """Build an elementwise binary test case configuration.

    Args:
        test_id: Unique test identifier
        operation: ONNX operation name (Add, Sub, Mul, Div)
        input_shapes: Input shapes dict (lhs/rhs)
        input_dtypes: Input dtypes dict (lhs/rhs)
        design: Design parameters (default: baseline)
        platform: Platform config (default: ZYNQ_7020)
        validation: Validation config (default: VALIDATION_STANDARD)

    Returns:
        KernelTestConfig ready for testing
    """
    if design is None:
        design = DESIGN_BASELINE
    if platform is None:
        platform = PLATFORM_ZYNQ7020
    if validation is None:
        validation = VALIDATION_STANDARD

    return KernelTestConfig(
        test_id=test_id,
        model=ModelStructure(
            operation=operation,
            input_shapes=input_shapes,
            input_dtypes=input_dtypes,
        ),
        design=design,
        platform=platform,
        validation=validation,
    )


# ============================================================================
# Validation Cases (Base - Operation-agnostic)
# ============================================================================
# These cases will be instantiated per operation (Add, Sub, Mul)
# Format: (test_id_suffix, shapes, dtypes, design, platform)
# test_id will be prefixed with operation name: "add_{suffix}"

VALIDATION_CASES_BASE = [
    # ========================================================================
    # Regular Cases: Basic Dtypes × Shapes (Hardware-Ready)
    # ========================================================================
    # INT8 cases (most common)
    ("int8_1x64_baseline", SHAPE_2D_1x64, DTYPES_INT8, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("int8_4x128_pe8", SHAPE_2D_4x128, DTYPES_INT8, DESIGN_PE8, PLATFORM_ZYNQ7020),
    ("int8_1x8x8x32_pe8", SHAPE_4D_1x8x8x32, DTYPES_INT8, DESIGN_PE8, PLATFORM_ZYNQ7020),
    # UINT8 cases
    ("uint8_1x64_baseline", SHAPE_2D_1x64, DTYPES_UINT8, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("uint8_4x128_pe8", SHAPE_2D_4x128, DTYPES_UINT8, DESIGN_PE8, PLATFORM_ZYNQ7020),
    # FLOAT32 cases
    ("float32_1x64_baseline", SHAPE_2D_1x64, DTYPES_FLOAT32, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("float32_4x128_pe8", SHAPE_2D_4x128, DTYPES_FLOAT32, DESIGN_PE8, PLATFORM_ZYNQ7020),
    # INT16 cases (wider types)
    ("int16_1x64_baseline", SHAPE_2D_1x64, DTYPES_INT16, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("int16_4x128_pe16", SHAPE_2D_4x128, DTYPES_INT16, DESIGN_PE16, PLATFORM_ZYNQ7020),
    # UINT16 cases
    ("uint16_1x64_baseline", SHAPE_2D_1x64, DTYPES_UINT16, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("uint16_4x128_pe16", SHAPE_2D_4x128, DTYPES_UINT16, DESIGN_PE16, PLATFORM_ZYNQ7020),
    # INT32 cases (largest width)
    ("int32_1x64_baseline", SHAPE_2D_1x64, DTYPES_INT32, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # UINT32 cases
    ("uint32_1x64_baseline", SHAPE_2D_1x64, DTYPES_UINT32, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # BIPOLAR cases (binary arithmetic)
    ("bipolar_1x64_baseline", SHAPE_2D_1x64, DTYPES_BIPOLAR, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # ========================================================================
    # Narrow/Binary DataTypes (Output Width Validation)
    # ========================================================================
    ("int4_1x16_baseline", SHAPE_2D_1x16, DTYPES_INT4, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("uint4_1x16_baseline", SHAPE_2D_1x16, DTYPES_UINT4, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    ("binary_1x64_baseline", SHAPE_2D_1x64, DTYPES_BINARY, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # ========================================================================
    # 3D Shape Coverage (Tests 3D Path in defines())
    # ========================================================================
    ("int8_1x16x64_baseline", SHAPE_3D_1x16x64, DTYPES_INT8, DESIGN_BASELINE, PLATFORM_ZYNQ7020),
    # ========================================================================
    # DSE Dimension Coverage: RAM Style Variants
    # ========================================================================
    ("int8_1x64_ram_distributed", SHAPE_2D_1x64, DTYPES_INT8, DESIGN_RAM_DISTRIBUTED, PLATFORM_ZYNQ7020),
    ("int8_1x64_ram_block", SHAPE_2D_1x64, DTYPES_INT8, DESIGN_RAM_BLOCK, PLATFORM_ZYNQ7020),
    ("int8_1x64_ram_ultra", SHAPE_2D_1x64, DTYPES_INT8, DESIGN_RAM_ULTRA, PLATFORM_ZYNQ7020),
    # ========================================================================
    # DSE Dimension Coverage: Memory Mode Variants
    # ========================================================================
    ("int8_1x64_mem_decoupled", SHAPE_2D_1x64, DTYPES_INT8, DESIGN_MEM_DECOUPLED, PLATFORM_ZYNQ7020),
    # ========================================================================
    # DSE Dimension Coverage: Combined Variants (PE + RAM Style)
    # ========================================================================
    ("int8_4x128_pe8_ram_distributed", SHAPE_2D_4x128, DTYPES_INT8, DESIGN_PE8_RAM_DISTRIBUTED, PLATFORM_ZYNQ7020),
    ("int8_4x128_pe8_ram_block", SHAPE_2D_4x128, DTYPES_INT8, DESIGN_PE8_RAM_BLOCK, PLATFORM_ZYNQ7020),
    # ========================================================================
    # Broadcasting Cases (Software-Only - Not Backend-Ready Yet)
    # ========================================================================
    # All 5 broadcasting patterns tested with INT8
    (
        "int8_channel_broadcast",
        SHAPE_BROADCAST_CHANNEL,
        DTYPES_INT8,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
    (
        "int8_scalar_broadcast",
        SHAPE_BROADCAST_SCALAR,
        DTYPES_INT8,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
    (
        "int8_spatial_broadcast",
        SHAPE_BROADCAST_SPATIAL,
        DTYPES_INT8,
        DESIGN_PE8,
        PLATFORM_SOFTWARE_ONLY,
    ),
    (
        "int8_bidirectional_broadcast",
        SHAPE_BROADCAST_BIDIR,
        DTYPES_INT8,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
    (
        "int8_rank_mismatch",
        SHAPE_BROADCAST_RANK,
        DTYPES_INT8,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
    # ========================================================================
    # Mixed DataType Cases (Software-Only - HLS Doesn't Support)
    # ========================================================================
    (
        "mixed_sign",
        SHAPE_2D_1x64,
        DTYPES_MIXED_SIGN,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
    (
        "mixed_width",
        SHAPE_2D_1x64,
        DTYPES_MIXED_WIDTH,
        DESIGN_BASELINE,
        PLATFORM_SOFTWARE_ONLY,
    ),
]
