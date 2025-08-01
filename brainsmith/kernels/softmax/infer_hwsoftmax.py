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
from brainsmith.core.plugins import transform


@transform(name="InferHWSoftmax", kernel="HWSoftmax",
    description="Convert Softmax nodes to HWSoftmax hardware operations",
    author="shane.fleming",
    version="1.0.0",
)
class InferHWSoftmax(Transformation):
    """
    Infers a regular softmax node without merging the multithreshold
    and setting the softmax to perform the quantisation.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Softmax":
                input_shape = model.get_tensor_shape(n.input[0])
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWSoftmax",
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