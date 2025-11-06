"""Automatic Quant node insertion for QONNX models.

This module provides utilities to insert Quant nodes (IntQuant, FloatQuant, BipolarQuant)
before model inputs to convert from float32 containers to QONNX-annotated types.
"""

import re
from typing import Dict

import numpy as np
import onnx.helper as oh
from qonnx.core.datatype import ArbPrecFloatType, DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.general.floatquant import compute_max_val


def insert_input_quant_nodes(
    model: ModelWrapper, input_datatypes: Dict[str, DataType]
) -> ModelWrapper:
    """
    Insert appropriate Quant nodes before model inputs.

    Automatically selects the correct Quant node type:
    - IntQuant for INT*/UINT*/BINARY types
    - FloatQuant for FLOAT<exp,mant,bias> types
    - BipolarQuant for BIPOLAR type
    - No node for FLOAT32/FLOAT64 (standard ONNX)

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
