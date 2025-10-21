# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Expand Norms Transform

Port of the ExpandNorms transform to the plugin system.
"""

import numpy as np
from onnx import helper as oh
from onnx import TensorProto
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


class ExpandNorms(Transformation):
    """Expand any standard LayerNorms/RMSNorms into the functional 
    norm and Mul/Add nodes for affine scale and bias."""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False

        existing_domains = {op.domain for op in model.model.opset_import}
        if "brainsmith.operators.general" not in existing_domains:
            model.model.opset_import.append(
                oh.make_opsetid("brainsmith.operators.general", 1)
            )
        
        for node in graph.node:
            node_ind += 1
            if node.op_type == "LayerNormalization":
                graph_modified = True
                ln_act_in = node.input[0]
                act_out = node.output[0]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) > 2 else None
                axis_attr = get_by_name(node.attribute, "axis")
                axis = axis_attr.i if axis_attr is not None else -1
                epsilon_attr = get_by_name(node.attribute, "epsilon")
                epsilon = epsilon_attr.f if epsilon_attr is not None else 1e-5
                idt = model.get_tensor_datatype(ln_act_in)
                wdt = model.get_tensor_datatype(scale)
                if bias:
                    bdt = model.get_tensor_datatype(bias)
                odt = model.get_tensor_datatype(act_out)

                act_shape = model.get_tensor_shape(ln_act_in)

                func_ln_node = oh.make_node(
                    "FuncLayerNorm",
                    [ln_act_in],
                    [act_out],
                    domain="brainsmith.operators.general",
                    backend="general",
                    axis=axis,
                    epsilon=epsilon,
                    InputDataType=idt.name,
                    OutputDataType=odt.name,
                    name=f"FuncLayerNorm_{node.name}",
                )

                # Get scale, eliminate if all ones
                has_scale = not np.all(scale == 1)
                if has_scale:
                    scale_act_in = oh.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, act_shape)
                    graph.value_info.append(scale_act_in)

                    func_ln_node.output[0] = scale_act_in.name
                    mul_node = oh.make_node("Mul", [scale_act_in.name, scale], [act_out])
                    
                    model.set_tensor_datatype(scale_act_in.name, idt)

                # Check if optional bias exists
                has_bias = bias is not None
                if has_bias:
                    bias_act_in = oh.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, act_shape)
                    graph.value_info.append(bias_act_in)
                    if has_scale:
                        mul_node.output[0] = bias_act_in.name
                    else:
                        func_ln_node.output[0] = bias_act_in.name
                    add_node = oh.make_node("Add", [bias_act_in.name, bias], [act_out])

                    model.set_tensor_datatype(bias_act_in.name, wdt)

                insert_point = node_ind
                graph.node.insert(insert_point, func_ln_node)
                if has_scale:
                    insert_point += 1
                    graph.node.insert(insert_point, mul_node)
                if has_bias:
                    insert_point += 1
                    graph.node.insert(insert_point, add_node)
                graph.node.remove(node)
                graph_modified = True

        model = model.transform(InferShapes())
        return (model, graph_modified)
