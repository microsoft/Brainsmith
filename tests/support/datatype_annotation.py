"""DataType annotation utilities for QONNX models.

This module provides two approaches for adding QONNX DataType information to models:

1. **Direct Annotation** (Recommended for testing):
   - Sets DataType metadata directly on tensors
   - No additional nodes added to graph
   - Simpler, faster, works for all backends (cppsim/rtlsim)
   - Use for: Kernel testing, pipeline testing

2. **Quant Node Insertion** (Use for specific scenarios):
   - Inserts IntQuant/FloatQuant/BipolarQuant nodes
   - Performs actual quantization during execution
   - Matches Brevitas export workflow
   - Use for: Testing Quant nodes themselves, validating quantization behavior

For most kernel tests, use direct annotation (annotate_model_datatypes).
For Quant node validation or Brevitas compatibility, use insert_input_quant_nodes
from tests.support.quant_insertion.

Supported DataTypes:
- Arbitrary integers 1-32 bits (INT/UINT)
- BIPOLAR, TERNARY
- FLOAT32

Unsupported DataTypes:
- Non-FP32 float types (FLOAT<exp,mant,bias>) - Brainsmith compiler limitation
"""

import warnings
from typing import Dict, Optional

from qonnx.core.datatype import ArbPrecFloatType, DataType
from qonnx.core.modelwrapper import ModelWrapper


def annotate_model_datatypes(
    model: ModelWrapper,
    tensor_datatypes: Dict[str, DataType],
    warn_unsupported: bool = True,
) -> ModelWrapper:
    """Set QONNX DataType annotations directly on tensors (no Quant nodes).

    This is the recommended approach for kernel testing. It produces identical
    InferDataTypes results as Quant node insertion, with simpler graph structure.

    Args:
        model: QONNX model to annotate
        tensor_datatypes: Dict mapping tensor names to QONNX DataTypes
        warn_unsupported: If True, warn when non-FP32 float types are used

    Returns:
        Model with DataType annotations added

    Example:
        >>> model = annotate_model_datatypes(model, {
        ...     "input": DataType["INT8"],
        ...     "param": DataType["INT16"],
        ...     "output": DataType["INT32"]
        ... })

    Notes:
        - No graph nodes are added (only metadata)
        - Produces same InferDataTypes results as Quant insertion
        - Works for all backends (cppsim/rtlsim)
        - Recommended for kernel testing
    """
    for tensor_name, dtype in tensor_datatypes.items():
        # Warn about unsupported non-FP32 float types
        if warn_unsupported:
            _check_datatype_support(dtype)

        model.set_tensor_datatype(tensor_name, dtype)

    return model


def annotate_inputs_and_outputs(
    model: ModelWrapper,
    input_datatypes: Dict[str, DataType],
    output_datatype: Optional[DataType] = None,
    warn_unsupported: bool = True,
) -> ModelWrapper:
    """Annotate model inputs and optionally outputs with DataTypes.

    Convenience function for the common case of annotating all inputs
    and optionally setting a default output datatype.

    Args:
        model: QONNX model to annotate
        input_datatypes: Dict mapping input names to QONNX DataTypes
        output_datatype: Optional DataType for all outputs (default: infer from inputs)
        warn_unsupported: If True, warn when non-FP32 float types are used

    Returns:
        Model with DataType annotations added

    Example:
        >>> model = annotate_inputs_and_outputs(model, {
        ...     "input": DataType["INT8"],
        ...     "param": DataType["INT8"]
        ... }, output_datatype=DataType["INT32"])

    Notes:
        - Annotates all specified inputs
        - If output_datatype is None, outputs remain unannotated (InferDataTypes will set them)
        - If output_datatype is provided, all outputs get same datatype
    """
    # Annotate inputs
    for input_name, dtype in input_datatypes.items():
        if warn_unsupported:
            _check_datatype_support(dtype)
        model.set_tensor_datatype(input_name, dtype)

    # Annotate outputs if requested
    if output_datatype is not None:
        if warn_unsupported:
            _check_datatype_support(output_datatype)

        output_names = [out.name for out in model.graph.output]
        for out_name in output_names:
            model.set_tensor_datatype(out_name, output_datatype)

    return model


def _check_datatype_support(dtype: DataType) -> None:
    """Check if datatype is supported by Brainsmith compiler and warn if not.

    Supported datatypes:
    - Arbitrary integers 1-32 bits (INT/UINT)
    - BIPOLAR, TERNARY
    - FLOAT32

    Args:
        dtype: QONNX DataType to check

    Warnings:
        UserWarning: If non-FP32 float type is used (not yet supported by compiler)
    """
    if isinstance(dtype, ArbPrecFloatType) and dtype != DataType["FLOAT32"]:
        warnings.warn(
            f"Non-FP32 float type {dtype} is not yet supported by the Brainsmith compiler. "
            f"Supported types: arbitrary integers 1-32 bits (INT/UINT), BIPOLAR, TERNARY, FLOAT32. "
            f"Tests may fail during backend code generation (cppsim/rtlsim).",
            UserWarning,
            stacklevel=3,
        )


# Export both approaches for convenience
__all__ = [
    "annotate_model_datatypes",
    "annotate_inputs_and_outputs",
]
