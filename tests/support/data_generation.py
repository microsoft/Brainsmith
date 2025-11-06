"""Extended test data generation with FP8 support.

This module extends QONNX's gen_finn_dt_tensor() to support arbitrary precision
float types (FLOAT<exp,mant,bias>) for FP8 and custom floating-point formats.
"""

import re
from typing import Optional

import numpy as np
from qonnx.core.datatype import ArbPrecFloatType, DataType
from qonnx.custom_op.general.floatquant import compute_max_val, float_quant
from qonnx.util.basic import gen_finn_dt_tensor


def generate_test_data(
    datatype: DataType, shape: tuple, seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate test data for any QONNX DataType, including FP8.

    Extends gen_finn_dt_tensor() with support for FLOAT<exp,mant,bias> types.
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
            raise ValueError(
                f"Unsupported DataType: {datatype}. " f"Original error: {e}"
            )


def _generate_arbprec_float(datatype: ArbPrecFloatType, shape: tuple) -> np.ndarray:
    """
    Generate arbitrary precision float data (FP8, E4M3, E5M2, etc.).

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
