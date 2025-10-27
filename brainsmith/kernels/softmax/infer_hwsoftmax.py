############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


class InferSoftmax(Transformation):
    """
    Converts standard ONNX Softmax nodes to brainsmith custom Softmax nodes.

    Only transforms nodes with op_type='Softmax' in standard ONNX domains.
    Skips nodes already in the brainsmith.kernels domain to prevent infinite loops.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            # Only convert standard ONNX Softmax nodes, not custom brainsmith ones
            is_standard_softmax = (
                n.op_type == "Softmax" and
                n.domain != "brainsmith.kernels"
            )
            if is_standard_softmax:
                input_shape = model.get_tensor_shape(n.input[0])
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "Softmax",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="brainsmith.kernels",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name=n.name,
                    SIMD=1,
                    NumChannels=input_shape[-1],
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)