"""Certification test suite generation utilities.

Provides helpers to generate comprehensive test sweeps across:
- DataTypes: All supported integer/float types
- Shapes: 2D, 3D, 4D tensor configurations
- Design Parameters: PE, SIMD, parallelization variants

Used by per-kernel certification_cases.py modules.
"""

from itertools import product
from typing import Iterator, List, Tuple

from qonnx.core.datatype import DataType

from tests.frameworks.test_config import DesignParameters, ModelStructure


# ============================================================================
# DataType Sweeps
# ============================================================================

# Integer types (signed)
DTYPES_SIGNED_INT = [
    DataType["INT4"],
    DataType["INT8"],
    DataType["INT16"],
    DataType["INT32"],
]

# Integer types (unsigned)
DTYPES_UNSIGNED_INT = [
    DataType["UINT4"],
    DataType["UINT8"],
    DataType["UINT16"],
    DataType["UINT32"],
]

# All integer types
DTYPES_INTEGER = DTYPES_SIGNED_INT + DTYPES_UNSIGNED_INT

# Binary/Bipolar
DTYPES_BINARY = [
    DataType["BINARY"],
    DataType["BIPOLAR"],
]

# Float types
DTYPES_FLOAT = [
    DataType["FLOAT32"],
]

# All types
DTYPES_ALL = DTYPES_INTEGER + DTYPES_BINARY + DTYPES_FLOAT


# ============================================================================
# Shape Sweeps
# ============================================================================

# 2D Shapes (typical for fully-connected layers)
SHAPES_2D = [
    (1, 64),  # Small
    (4, 128),  # Medium
    (8, 256),  # Large
]

# 3D Shapes (sequence processing)
SHAPES_3D = [
    (1, 16, 64),  # Short sequence
    (4, 32, 128),  # Medium sequence
]

# 4D Shapes (typical for conv/pooling)
SHAPES_4D = [
    (1, 8, 8, 32),  # Small feature map
    (1, 16, 16, 64),  # Medium feature map
    (1, 32, 32, 128),  # Large feature map
]

# All shapes
SHAPES_ALL = SHAPES_2D + SHAPES_3D + SHAPES_4D


# ============================================================================
# Design Parameter Sweeps
# ============================================================================

# PE (Parallelism Element) variants
PE_VARIANTS = [1, 2, 4, 8, 16, 32]

# SIMD (Single Instruction Multiple Data) variants
SIMD_VARIANTS = [1, 2, 4, 8, 16, 32]


# ============================================================================
# Sweep Generators
# ============================================================================


def sweep_dtypes(
    dtypes: List[DataType],
    shapes: List[Tuple[int, ...]],
    operation: str = "placeholder",
    input_names: Tuple[str, str] = ("lhs", "rhs"),
    design: DesignParameters = None,
) -> Iterator[Tuple[str, ModelStructure, DesignParameters]]:
    """Generate test cases across dtypes and shapes.

    Args:
        dtypes: List of DataTypes to sweep
        shapes: List of tensor shapes to sweep
        operation: ONNX operation name
        input_names: Input tensor names (default: lhs/rhs)
        design: Base DesignParameters (default: baseline)

    Yields:
        (test_id, ModelStructure, DesignParameters) tuples
    """
    if design is None:
        design = DesignParameters()

    for dtype, shape in product(dtypes, shapes):
        dtype_name = dtype.name.lower()
        shape_str = "x".join(map(str, shape))
        test_id = f"{dtype_name}_{shape_str}"

        model = ModelStructure(
            operation=operation,
            input_shapes={input_names[0]: shape, input_names[1]: shape},
            input_dtypes={input_names[0]: dtype, input_names[1]: dtype},
        )

        yield test_id, model, design


def sweep_pe_parallelization(
    base_cases: List[Tuple[str, ModelStructure, DesignParameters]], pe_values: List[int] = None
) -> Iterator[Tuple[str, ModelStructure, DesignParameters]]:
    """Generate PE parallelization variants for each base case.

    Args:
        base_cases: Base test cases to parameterize
        pe_values: PE values to sweep (default: [1, 8, 16])

    Yields:
        (test_id, ModelStructure, DesignParameters) tuples with PE variants
    """
    if pe_values is None:
        pe_values = [1, 8, 16]

    for test_id, model, design in base_cases:
        for pe in pe_values:
            # Check if PE divides last dimension
            last_dim = model.input_shapes[list(model.input_shapes.keys())[0]][-1]
            if last_dim % pe != 0:
                continue  # Skip invalid PE values

            pe_suffix = f"_pe{pe}" if pe > 1 else "_baseline"
            new_test_id = f"{test_id}{pe_suffix}"

            new_design = DesignParameters(
                input_streams={0: pe} if pe > 1 else {},
                output_streams=design.output_streams,
                dimensions=design.dimensions,
                backend_variants=design.backend_variants,
            )

            yield new_test_id, model, new_design


def sweep_simd_parallelization(
    base_cases: List[Tuple[str, ModelStructure, DesignParameters]], simd_values: List[int] = None
) -> Iterator[Tuple[str, ModelStructure, DesignParameters]]:
    """Generate SIMD parallelization variants for each base case.

    Args:
        base_cases: Base test cases to parameterize
        simd_values: SIMD values to sweep (default: [1, 8, 16])

    Yields:
        (test_id, ModelStructure, DesignParameters) tuples with SIMD variants
    """
    if simd_values is None:
        simd_values = [1, 8, 16]

    for test_id, model, design in base_cases:
        for simd in simd_values:
            simd_suffix = f"_simd{simd}" if simd > 1 else ""
            new_test_id = f"{test_id}{simd_suffix}"

            new_design = DesignParameters(
                input_streams=design.input_streams,
                output_streams=design.output_streams,
                dimensions={**design.dimensions, "SIMD": simd},
                backend_variants=design.backend_variants,
            )

            yield new_test_id, model, new_design


# ============================================================================
# Certification Suite Builder
# ============================================================================


def build_certification_suite(
    operation: str,
    dtypes: List[DataType] = None,
    shapes: List[Tuple[int, ...]] = None,
    pe_variants: List[int] = None,
    input_names: Tuple[str, str] = ("lhs", "rhs"),
) -> List[Tuple[str, ModelStructure, DesignParameters]]:
    """Build comprehensive certification suite for a kernel.

    Generates test cases covering:
    - All specified dtypes × shapes (baseline)
    - PE parallelization variants (if specified)

    Args:
        operation: ONNX operation name
        dtypes: DataTypes to test (default: all integer types)
        shapes: Shapes to test (default: representative 2D/4D)
        pe_variants: PE values to sweep (default: [1, 8])
        input_names: Input tensor names

    Returns:
        List of (test_id, ModelStructure, DesignParameters) tuples
    """
    if dtypes is None:
        dtypes = DTYPES_INTEGER  # Default: all integer types

    if shapes is None:
        shapes = [SHAPES_2D[0], SHAPES_4D[0]]  # Default: one 2D, one 4D

    if pe_variants is None:
        pe_variants = [1, 8]  # Default: baseline + PE=8

    # Generate base cases (dtype × shape)
    base_cases = list(sweep_dtypes(dtypes, shapes, operation, input_names))

    # Generate PE variants
    all_cases = list(sweep_pe_parallelization(base_cases, pe_variants))

    return all_cases
