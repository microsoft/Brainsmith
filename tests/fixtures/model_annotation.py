"""Model annotation utilities for QONNX testing.

This module consolidates model annotation approaches from multiple files:

**Consolidated from:**
- tests/support/datatype_annotation.py (Direct annotation - v2.3 standard)
- tests/support/quant_insertion.py (Quant node insertion - legacy)
- tests/support/onnx_utils.py (Type conversion utilities)

**Provides two complementary approaches:**

1. **Direct Annotation (v2.3+, recommended):**
   - annotate_model_datatypes() - Set DataType metadata directly
   - annotate_inputs_and_outputs() - Convenience wrapper
   - No Quant nodes added to graph
   - Simpler, faster, works for all backends (cppsim/rtlsim)
   - Use for: Most kernel tests, pipeline tests

2. **Quant Node Insertion (legacy, for Quant testing):**
   - insert_input_quant_nodes() - Add IntQuant/FloatQuant/BipolarQuant nodes
   - Explicit quantization in graph
   - Matches Brevitas export workflow
   - Use for: Testing Quant nodes themselves, validating quantization behavior

**Type Conversion Utilities:**
- tensorproto_for_datatype() - DataType → TensorProto (FINN convention)
- datatype_to_actual_tensorproto() - DataType → TensorProto (Stage 1)
- datatype_to_numpy_dtype() - DataType → NumPy dtype
- get_tensorproto_name() - TensorProto → human-readable string

Usage:
    # v2.3 standard (recommended)
    from tests.fixtures.model_annotation import annotate_model_datatypes
    model = annotate_model_datatypes(model, {"input": DataType["INT8"]})

    # Legacy Quant insertion (for testing Quant nodes)
    from tests.fixtures.model_annotation import insert_input_quant_nodes
    model = insert_input_quant_nodes(model, {"input": DataType["INT8"]})

    # Type conversions
    from tests.fixtures.model_annotation import datatype_to_actual_tensorproto
    tensorproto_type = datatype_to_actual_tensorproto(DataType["INT8"])

Supported DataTypes:
- Arbitrary integers 1-32 bits (INT/UINT)
- BIPOLAR, TERNARY
- FLOAT32, FLOAT16
- Arbitrary precision floats (FLOAT<exp,mant,bias>)
"""

import re
import warnings
from typing import Dict, Optional

import numpy as np
import onnx.helper as oh
from onnx import TensorProto
from qonnx.core.datatype import ArbPrecFloatType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.floatquant import compute_max_val


# ============================================================================
# Direct Annotation (v2.3 Standard)
# ============================================================================


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


# ============================================================================
# Quant Node Insertion (Legacy, for Quant testing)
# ============================================================================


def insert_input_quant_nodes(
    model: ModelWrapper, input_datatypes: Dict[str, DataType]
) -> ModelWrapper:
    """Insert IntQuant/FloatQuant/BipolarQuant nodes before model inputs.

    Automatically selects the correct Quant node type:
    - IntQuant for INT*/UINT*/BINARY types
    - FloatQuant for FLOAT<exp,mant,bias> types
    - BipolarQuant for BIPOLAR type
    - No node for FLOAT32/FLOAT64 (standard ONNX)

    Use when:
    - Testing Quant node behavior explicitly
    - Validating Brevitas export compatibility
    - Debugging quantization issues

    For most kernel tests, use annotate_model_datatypes() instead.

    Args:
        model: QONNX model with float32 inputs
        input_datatypes: {input_name: desired_datatype}

    Returns:
        Model with Quant nodes inserted before specified inputs

    Example:
        >>> model = insert_input_quant_nodes(model, {
        ...     "input": DataType["INT9"],
        ...     "param": DataType["FLOAT<5,10,15>"]
        ... })
    """
    for input_name, datatype in input_datatypes.items():
        if datatype == DataType["BIPOLAR"]:
            _insert_bipolar_quant(model, input_name)
        elif isinstance(datatype, ArbPrecFloatType):
            _insert_float_quant(model, input_name, datatype)
        elif (
            datatype.name.startswith("INT")
            or datatype.name.startswith("UINT")
            or datatype == DataType["BINARY"]
        ):
            _insert_int_quant(model, input_name, datatype)
        elif datatype == DataType["FLOAT32"]:
            # No Quant node needed - already standard ONNX float32
            pass
        else:
            raise ValueError(f"Unsupported DataType for Quant insertion: {datatype}")

    return model


def _insert_int_quant(
    model: ModelWrapper, input_name: str, datatype: DataType
) -> None:
    """Insert IntQuant node for INT*/UINT*/BINARY types.

    Args:
        model: QONNX model to modify
        input_name: Name of input to quantize
        datatype: Target DataType (INT8, INT9, UINT4, BINARY, etc.)
    """
    # Extract parameters from DataType
    bitwidth = datatype.bitwidth()
    signed = 1 if datatype.signed() else 0

    # Create initializers (scale=1.0, zeropt=0.0 for pure integer)
    scale_name = model.make_new_valueinfo_name()
    model.set_initializer(scale_name, np.array(1.0, dtype=np.float32))

    zeropt_name = model.make_new_valueinfo_name()
    model.set_initializer(zeropt_name, np.array(0.0, dtype=np.float32))

    bitwidth_name = model.make_new_valueinfo_name()
    model.set_initializer(bitwidth_name, np.array(float(bitwidth), dtype=np.float32))

    # Rename original input to "raw_<input_name>"
    raw_input_name = f"raw_{input_name}"
    _rename_graph_input(model, input_name, raw_input_name)

    # Create IntQuant node (op_type="Quant" for IntQuant)
    quant_node = oh.make_node(
        "Quant",
        inputs=[raw_input_name, scale_name, zeropt_name, bitwidth_name],
        outputs=[input_name],
        domain="qonnx.custom_op.general",
        signed=signed,
        narrow=0,  # Full range [-2^(n-1), 2^(n-1)-1]
        rounding_mode="ROUND",
        name=f"IntQuant_{input_name}",
    )

    # Insert at beginning of graph
    model.graph.node.insert(0, quant_node)

    # Set output DataType annotation
    model.set_tensor_datatype(input_name, datatype)


def _insert_float_quant(
    model: ModelWrapper, input_name: str, datatype: ArbPrecFloatType
) -> None:
    """Insert FloatQuant node for FLOAT<exp,mant,bias> types.

    Args:
        model: QONNX model to modify
        input_name: Name of input to quantize
        datatype: Target ArbPrecFloatType (e.g., FLOAT<5,10,15>)
    """
    # Parse datatype: "FLOAT<8,23,127>"
    match = re.match(r"FLOAT<(\d+),(\d+),(\d+)>", datatype.name)
    if not match:
        raise ValueError(f"Invalid FLOAT datatype format: {datatype.name}")

    exp_bitwidth = int(match.group(1))
    mant_bitwidth = int(match.group(2))
    exp_bias = int(match.group(3))

    # Compute max value for this float format
    max_val = compute_max_val(exp_bitwidth, mant_bitwidth, exp_bias)

    # Create initializers
    scale_name = model.make_new_valueinfo_name()
    model.set_initializer(scale_name, np.array(1.0, dtype=np.float32))

    exp_bw_name = model.make_new_valueinfo_name()
    model.set_initializer(exp_bw_name, np.array(float(exp_bitwidth), dtype=np.float32))

    mant_bw_name = model.make_new_valueinfo_name()
    model.set_initializer(
        mant_bw_name, np.array(float(mant_bitwidth), dtype=np.float32)
    )

    bias_name = model.make_new_valueinfo_name()
    model.set_initializer(bias_name, np.array(float(exp_bias), dtype=np.float32))

    max_val_name = model.make_new_valueinfo_name()
    model.set_initializer(max_val_name, np.array(max_val, dtype=np.float32))

    # Rename original input
    raw_input_name = f"raw_{input_name}"
    _rename_graph_input(model, input_name, raw_input_name)

    # Create FloatQuant node
    quant_node = oh.make_node(
        "FloatQuant",
        inputs=[
            raw_input_name,
            scale_name,
            exp_bw_name,
            mant_bw_name,
            bias_name,
            max_val_name,
        ],
        outputs=[input_name],
        domain="qonnx.custom_op.general",
        rounding_mode="ROUND",
        has_inf=0,
        has_nan=0,
        has_subnormal=0,
        saturation=1,
        name=f"FloatQuant_{input_name}",
    )

    model.graph.node.insert(0, quant_node)
    model.set_tensor_datatype(input_name, datatype)


def _insert_bipolar_quant(model: ModelWrapper, input_name: str) -> None:
    """Insert BipolarQuant node for BIPOLAR type.

    Args:
        model: QONNX model to modify
        input_name: Name of input to quantize
    """
    # Create scale initializer (scale=1.0)
    scale_name = model.make_new_valueinfo_name()
    model.set_initializer(scale_name, np.array(1.0, dtype=np.float32))

    # Rename original input
    raw_input_name = f"raw_{input_name}"
    _rename_graph_input(model, input_name, raw_input_name)

    # Create BipolarQuant node
    quant_node = oh.make_node(
        "BipolarQuant",
        inputs=[raw_input_name, scale_name],
        outputs=[input_name],
        domain="qonnx.custom_op.general",
        name=f"BipolarQuant_{input_name}",
    )

    model.graph.node.insert(0, quant_node)
    model.set_tensor_datatype(input_name, DataType["BIPOLAR"])


def _rename_graph_input(model: ModelWrapper, old_name: str, new_name: str) -> None:
    """Helper to rename a graph input.

    Args:
        model: QONNX model to modify
        old_name: Current input name
        new_name: New input name

    Raises:
        ValueError: If input not found in model
    """
    for i, inp in enumerate(model.graph.input):
        if inp.name == old_name:
            model.graph.input[i].name = new_name
            return
    raise ValueError(f"Input '{old_name}' not found in model")


# ============================================================================
# Type Conversion Utilities
# ============================================================================


def tensorproto_for_datatype(dtype: DataType) -> int:
    """Map QONNX DataType to ONNX TensorProto type (FINN convention).

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


def datatype_to_actual_tensorproto(dtype: DataType) -> int:
    """Convert QONNX DataType to actual ONNX TensorProto type (Stage 1).

    This is different from tensorproto_for_datatype() which returns FLOAT containers
    for all integer types (FINN convention, Stage 2). This function returns actual
    TensorProto types that preserve correct semantics in ONNX Runtime.

    Use Cases:
    - Stage 1 (pure ONNX golden reference): Use this function for correct semantics
    - Stage 2 (QONNX FINN/Brainsmith): Use tensorproto_for_datatype() for FLOAT containers

    Examples:
        >>> # Stage 1: Actual integer types for golden reference
        >>> datatype_to_actual_tensorproto(DataType["INT8"])
        TensorProto.INT8  # ONNX Runtime uses integer division (7÷2=3)

        >>> # Stage 2: FLOAT containers for FINN/Brainsmith
        >>> tensorproto_for_datatype(DataType["INT8"])
        TensorProto.FLOAT  # Hardware interprets as fixed-point

        >>> # Float types are same in both
        >>> datatype_to_actual_tensorproto(DataType["FLOAT32"])
        TensorProto.FLOAT

    Args:
        dtype: QONNX DataType instance

    Returns:
        ONNX TensorProto type constant with correct semantics

    Note:
        Integer division example: INT8(7) ÷ INT8(2) = INT8(3) with actual types,
        but FLOAT(7) ÷ FLOAT(2) = FLOAT(3.5) with FINN convention.
    """
    # FLOAT16 is special - use native TensorProto.FLOAT16
    if dtype == DataType["FLOAT16"]:
        return TensorProto.FLOAT16

    if dtype.is_integer():
        # Integer types → proper integer TensorProto
        if dtype.signed():
            if dtype.bitwidth() <= 8:
                return TensorProto.INT8
            elif dtype.bitwidth() <= 16:
                return TensorProto.INT16
            elif dtype.bitwidth() <= 32:
                return TensorProto.INT32
            else:
                return TensorProto.INT64
        else:
            if dtype.bitwidth() <= 8:
                return TensorProto.UINT8
            elif dtype.bitwidth() <= 16:
                return TensorProto.UINT16
            elif dtype.bitwidth() <= 32:
                return TensorProto.UINT32
            else:
                return TensorProto.UINT64
    else:
        # Float types (FLOAT32, DOUBLE, etc.)
        return TensorProto.FLOAT


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


# Export all public functions
__all__ = [
    # Direct annotation (v2.3)
    "annotate_model_datatypes",
    "annotate_inputs_and_outputs",
    # Quant insertion (legacy)
    "insert_input_quant_nodes",
    # Type conversions
    "tensorproto_for_datatype",
    "datatype_to_actual_tensorproto",
    "datatype_to_numpy_dtype",
    "get_tensorproto_name",
]
