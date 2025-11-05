"""ONNX utilities for test framework.

This module provides utilities for working with standard ONNX types (TensorProto)
and converting between ONNX and QONNX representations. Critical for maintaining
clean separation between ONNX stage (pure standard) and QONNX stage (FINN/Brainsmith).

Key Concepts:
- TensorProto types: Standard ONNX tensor types (INT8, FLOAT, etc.)
- QONNX DataType: FINN/Brainsmith semantic annotations (overlaid on TensorProto)
- Container types: ONNX uses FLOAT containers for quantized integers (FINN convention)

Architecture:
    Stage 1 (ONNX):  make_onnx_model() → TensorProto types
    Stage 2 (QONNX): annotate_with_qonnx() → DataType annotations

Golden Reference: Always validates against Stage 1 (pure ONNX semantics)
FINN/Brainsmith Execution: Uses Stage 2 (QONNX annotations)
"""

import numpy as np
from typing import Tuple
from onnx import TensorProto
from qonnx.core.datatype import DataType


def tensorproto_for_datatype(dtype: DataType) -> int:
    """Map QONNX DataType to ONNX TensorProto type.

    Follows FINN/QONNX convention:
    - Integer types (INT*, UINT*): Use FLOAT container
    - Float types (FLOAT32, FLOAT16): Use matching TensorProto type

    This allows ONNX Runtime to execute quantized models using floating-point
    operations while FINN/Brainsmith can later interpret them as fixed-point.

    Args:
        dtype: QONNX DataType instance

    Returns:
        TensorProto type constant (e.g., TensorProto.FLOAT)

    Example:
        >>> tensorproto_for_datatype(DataType["INT8"])
        1  # TensorProto.FLOAT

        >>> tensorproto_for_datatype(DataType["FLOAT32"])
        1  # TensorProto.FLOAT

        >>> tensorproto_for_datatype(DataType["FLOAT16"])
        10  # TensorProto.FLOAT16
    """
    # FLOAT16 is special - use native TensorProto.FLOAT16
    if dtype == DataType["FLOAT16"]:
        return TensorProto.FLOAT16

    # Everything else uses FLOAT container (FINN convention)
    # This includes:
    # - Integer types (INT8, INT16, UINT8, etc.) → FLOAT container
    # - FLOAT32 → FLOAT
    # - Binary/Bipolar/Ternary → FLOAT container
    return TensorProto.FLOAT


def generate_onnx_test_data(
    tensor_type: int,
    shape: Tuple[int, ...],
    seed: int = None,
    value_range: Tuple[float, float] = None
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


def get_onnx_tensor_type(model, tensor_name: str) -> int:
    """Get ONNX TensorProto type for a tensor.

    Args:
        model: ModelWrapper instance
        tensor_name: Name of tensor

    Returns:
        TensorProto type constant

    Raises:
        ValueError: If tensor not found or has no type info
    """
    # Check graph inputs
    for inp in model.graph.input:
        if inp.name == tensor_name:
            return inp.type.tensor_type.elem_type

    # Check graph outputs
    for out in model.graph.output:
        if out.name == tensor_name:
            return out.type.tensor_type.elem_type

    # Check value_info (intermediate tensors)
    for vi in model.graph.value_info:
        if vi.name == tensor_name:
            return vi.type.tensor_type.elem_type

    raise ValueError(
        f"Tensor '{tensor_name}' not found in model or has no type information"
    )


def datatype_to_numpy_dtype(dtype: DataType) -> np.dtype:
    """Convert QONNX DataType to NumPy dtype.

    Args:
        dtype: QONNX DataType instance

    Returns:
        NumPy dtype

    Example:
        >>> datatype_to_numpy_dtype(DataType["INT8"])
        dtype('int8')

        >>> datatype_to_numpy_dtype(DataType["FLOAT32"])
        dtype('float32')
    """
    if dtype == DataType["FLOAT32"]:
        return np.float32
    elif dtype == DataType["FLOAT16"]:
        return np.float16
    elif dtype in [DataType["INT8"], DataType["BINARY"], DataType["BIPOLAR"]]:
        return np.int8
    elif dtype == DataType["INT16"]:
        return np.int16
    elif dtype == DataType["INT32"]:
        return np.int32
    elif dtype == DataType["UINT8"]:
        return np.uint8
    elif dtype == DataType["UINT16"]:
        return np.uint16
    elif dtype == DataType["UINT32"]:
        return np.uint32
    else:
        # Default to float32 for unknown types
        return np.float32


def get_tensorproto_name(tensor_type: int) -> str:
    """Get human-readable name for TensorProto type.

    Args:
        tensor_type: TensorProto type constant

    Returns:
        Type name string

    Example:
        >>> get_tensorproto_name(TensorProto.FLOAT)
        'FLOAT'

        >>> get_tensorproto_name(TensorProto.INT8)
        'INT8'
    """
    type_map = {
        TensorProto.FLOAT: "FLOAT",
        TensorProto.UINT8: "UINT8",
        TensorProto.INT8: "INT8",
        TensorProto.UINT16: "UINT16",
        TensorProto.INT16: "INT16",
        TensorProto.INT32: "INT32",
        TensorProto.INT64: "INT64",
        TensorProto.STRING: "STRING",
        TensorProto.BOOL: "BOOL",
        TensorProto.FLOAT16: "FLOAT16",
        TensorProto.DOUBLE: "DOUBLE",
        TensorProto.UINT32: "UINT32",
        TensorProto.UINT64: "UINT64",
        TensorProto.COMPLEX64: "COMPLEX64",
        TensorProto.COMPLEX128: "COMPLEX128",
    }
    return type_map.get(tensor_type, f"UNKNOWN({tensor_type})")
