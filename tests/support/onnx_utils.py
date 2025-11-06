"""ONNX model introspection utilities.

This module provides runtime utilities for querying ONNX model properties.

**Note:** Type conversion utilities have been moved to tests/fixtures/model_annotation.py:
- tensorproto_for_datatype()
- datatype_to_actual_tensorproto()
- datatype_to_numpy_dtype()
- get_tensorproto_name()

**Note:** Data generation utilities have been moved to tests/fixtures/test_data.py:
- generate_test_data()
- generate_onnx_test_data()

This module now only contains runtime introspection functions.
"""

from qonnx.core.modelwrapper import ModelWrapper


def get_onnx_tensor_type(model: ModelWrapper, tensor_name: str) -> int:
    """Get ONNX TensorProto type for a tensor.

    Args:
        model: ModelWrapper instance
        tensor_name: Name of tensor

    Returns:
        TensorProto type constant

    Raises:
        ValueError: If tensor not found or has no type info

    Example:
        >>> from onnx import TensorProto
        >>> tensor_type = get_onnx_tensor_type(model, "input")
        >>> tensor_type == TensorProto.FLOAT
        True
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


__all__ = ["get_onnx_tensor_type"]
