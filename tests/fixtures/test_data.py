"""Test data generation for ONNX and QONNX models.

This module consolidates test data generation from multiple files:

**Consolidated from:**
- tests/support/data_generation.py (QONNX DataType generation with FP8)
- tests/support/onnx_utils.py (ONNX TensorProto generation)

**Provides two data generation approaches:**

1. **QONNX DataType Generation (recommended for kernel testing):**
   - generate_test_data() - Generate for any QONNX DataType
   - Supports arbitrary precision floats (FLOAT<exp,mant,bias>) for FP8
   - Always returns float32 containers (QONNX convention)
   - Uses QONNX gen_finn_dt_tensor() with FP8 extension

2. **ONNX TensorProto Generation (for Stage 1 golden reference):**
   - generate_onnx_test_data() - Generate for ONNX TensorProto types
   - Returns actual NumPy dtypes (int8, float32, etc.)
   - ONNX-native generation (not QONNX-dependent)
   - Use for pure ONNX golden reference validation

Usage:
    # QONNX DataType (kernel testing)
    from tests.fixtures.test_data import generate_test_data
    from qonnx.core.datatype import DataType

    data = generate_test_data(DataType["INT8"], (16, 9), seed=42)
    # → float32 array with values in [-128, 127]

    data = generate_test_data(DataType["FLOAT<5,10,15>"], (16, 9))
    # → float32 array quantized to FP8 E5M10 format

    # ONNX TensorProto (golden reference)
    from tests.fixtures.test_data import generate_onnx_test_data
    from onnx import TensorProto

    data = generate_onnx_test_data(TensorProto.INT8, (3, 4), seed=42)
    # → int8 array with values in [-128, 127]

Supported Formats:
- QONNX: INT/UINT (1-32 bits), BINARY, BIPOLAR, TERNARY, FLOAT32/16, FLOAT<e,m,b>
- ONNX: FLOAT, FLOAT16, INT8/16/32, UINT8/16/32
"""

import re
from typing import Optional, Tuple

import numpy as np
from onnx import TensorProto
from qonnx.core.datatype import ArbPrecFloatType, DataType
from qonnx.custom_op.general.floatquant import compute_max_val, float_quant
from qonnx.util.basic import gen_finn_dt_tensor


# ============================================================================
# QONNX DataType Generation (Kernel Testing)
# ============================================================================


def generate_test_data(
    datatype: DataType, shape: tuple, seed: Optional[int] = None
) -> np.ndarray:
    """Generate test data for any QONNX DataType, including FP8.

    Extends QONNX's gen_finn_dt_tensor() with support for FLOAT<exp,mant,bias> types.
    Always returns float32 containers (QONNX convention).

    Args:
        datatype: QONNX DataType (INT8, FLOAT<5,10,15>, BIPOLAR, etc.)
        shape: Tensor shape tuple
        seed: Optional random seed for reproducibility

    Returns:
        np.ndarray with dtype=float32, values in datatype's range

    Examples:
        >>> generate_test_data(DataType["INT8"], (16, 9))
        array([...], dtype=float32)  # Values in [-128, 127]

        >>> generate_test_data(DataType["FLOAT<5,10,15>"], (16, 9))
        array([...], dtype=float32)  # FP8 quantized values

        >>> # Mixed-type inputs
        >>> inputs = {
        ...     "int_input": generate_test_data(DataType["INT8"], (16, 9)),
        ...     "float_input": generate_test_data(DataType["FLOAT<5,10,15>"], (16, 9))
        ... }
    """
    if seed is not None:
        np.random.seed(seed)

    # Try QONNX utility first (handles INT, UINT, BINARY, BIPOLAR, FIXED, FLOAT32/16)
    try:
        return gen_finn_dt_tensor(datatype, shape)
    except ValueError as e:
        # Handle unsupported types (FLOAT<exp,mant,bias>)
        if isinstance(datatype, ArbPrecFloatType):
            return _generate_arbprec_float(datatype, shape)
        else:
            raise ValueError(f"Unsupported DataType: {datatype}. Original error: {e}")


def _generate_arbprec_float(datatype: ArbPrecFloatType, shape: tuple) -> np.ndarray:
    """Generate arbitrary precision float data (FP8, E4M3, E5M2, etc.).

    Uses FloatQuant to quantize random float32 values to the target format.

    Args:
        datatype: ArbPrecFloatType instance (e.g., FLOAT<5,10,15>)
        shape: Tensor shape tuple

    Returns:
        np.ndarray with dtype=float32, quantized to FP format

    Raises:
        ValueError: If datatype format is invalid
    """
    # Parse FLOAT<exp,mant,bias> format
    match = re.match(r"FLOAT<(\d+),(\d+),(\d+)>", datatype.name)
    if not match:
        raise ValueError(f"Invalid ArbPrecFloatType format: {datatype.name}")

    exp_bits = int(match.group(1))
    mant_bits = int(match.group(2))
    exp_bias = int(match.group(3))

    # Compute max representable value for this format
    max_val = compute_max_val(exp_bits, mant_bits, exp_bias)

    # Generate random float32 values in representable range
    # Use uniform distribution scaled to max_val
    values = np.random.uniform(-max_val, max_val, shape).astype(np.float32)

    # Apply float quantization to simulate FP8 format
    quantized = float_quant(
        values,
        scale=np.array(1.0, dtype=np.float32),
        exponent_bitwidth=np.array(exp_bits, dtype=np.float32),
        mantissa_bitwidth=np.array(mant_bits, dtype=np.float32),
        exponent_bias=np.array(exp_bias, dtype=np.float32),
        signed=True,
        max_val=np.array(max_val, dtype=np.float32),
        has_inf=False,
        has_nan=False,
        has_subnormal=False,
        rounding_mode="ROUND",
        saturation=True,
    )

    # Ensure float32 dtype (float_quant may return float64)
    return quantized.astype(np.float32)


# ============================================================================
# ONNX TensorProto Generation (Golden Reference)
# ============================================================================


def generate_onnx_test_data(
    tensor_type: int,
    shape: Tuple[int, ...],
    seed: Optional[int] = None,
    value_range: Optional[Tuple[float, float]] = None,
) -> np.ndarray:
    """Generate test data for ONNX TensorProto types.

    Generates random test data appropriate for the TensorProto type.
    This is ONNX-native generation (not QONNX-dependent).

    Args:
        tensor_type: ONNX TensorProto type constant
        shape: Tensor shape (tuple of dimensions)
        seed: Random seed for reproducibility (optional)
        value_range: Override default value range (min, max) (optional)

    Returns:
        NumPy array with appropriate dtype and values

    Example:
        >>> # Generate FLOAT data
        >>> data = generate_onnx_test_data(TensorProto.FLOAT, (3, 4), seed=42)
        >>> data.dtype
        dtype('float32')

        >>> # Generate INT8 data (returns int8 array)
        >>> data = generate_onnx_test_data(TensorProto.INT8, (3, 4), seed=42)
        >>> data.dtype
        dtype('int8')
    """
    if seed is not None:
        np.random.seed(seed)

    # FLOAT types - use normal distribution
    if tensor_type == TensorProto.FLOAT:
        if value_range:
            # Generate in specified range
            data = np.random.uniform(value_range[0], value_range[1], size=shape)
        else:
            # Standard normal distribution (mean=0, std=1)
            data = np.random.randn(*shape)
        return data.astype(np.float32)

    elif tensor_type == TensorProto.FLOAT16:
        if value_range:
            data = np.random.uniform(value_range[0], value_range[1], size=shape)
        else:
            data = np.random.randn(*shape)
        return data.astype(np.float16)

    # Integer types - use uniform integer distribution
    elif tensor_type == TensorProto.INT8:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = -128, 127
        return np.random.randint(low, high + 1, size=shape, dtype=np.int8)

    elif tensor_type == TensorProto.INT16:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = -32768, 32767
        return np.random.randint(low, high + 1, size=shape, dtype=np.int16)

    elif tensor_type == TensorProto.INT32:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = -2147483648, 2147483647
        return np.random.randint(low, high + 1, size=shape, dtype=np.int32)

    elif tensor_type == TensorProto.UINT8:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = 0, 255
        return np.random.randint(low, high + 1, size=shape, dtype=np.uint8)

    elif tensor_type == TensorProto.UINT16:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = 0, 65535
        return np.random.randint(low, high + 1, size=shape, dtype=np.uint16)

    elif tensor_type == TensorProto.UINT32:
        if value_range:
            low, high = int(value_range[0]), int(value_range[1])
        else:
            low, high = 0, 4294967295
        return np.random.randint(low, high + 1, size=shape, dtype=np.uint32)

    # Unsupported type
    else:
        raise ValueError(
            f"Unsupported TensorProto type: {tensor_type}. "
            f"Supported types: FLOAT, FLOAT16, INT8, INT16, INT32, UINT8, UINT16, UINT32"
        )


# Export public functions
__all__ = [
    # QONNX DataType generation
    "generate_test_data",
    # ONNX TensorProto generation
    "generate_onnx_test_data",
]
