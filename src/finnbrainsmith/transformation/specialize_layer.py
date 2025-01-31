############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

from onnx.helper import make_node
from finnbrainsmith.custom_op.fpgadataflow.hls import custom_op as hls_variants
from typing import Optional
from onnx.helper import NodeProto
from qonnx.transformation.base import Transformation

def specilise_factory(hwop:NodeProto, fpgapart:str)->Optional[NodeProto]:
    """ Given a ONNX node produce a new version that is specialised """
    if hwop.op_type == "Shuffle": 
        if hwop.get_nodeattr("inner_dim") == 1 and hwop.get_nodeattr("SIMD") > 1:
            optype = "InnerDimShuffle_hls"
        else:
            optype = "Shuffle_hls"

        return make_node(
                optype, 
                hwop.input,
                hwop.output,
                domain="finnbrainsmith.custom_op.fpgadataflow.hls"
        )
    return None

class SpecializeLayerVisitor(Transformation):
    """ Visits every node in the graph and calls it's specialise function
    adding relevant global state information, such as the fpgapart.
    """

    def __init__(self, fpgapart):
        super().__init__()
        self.fpgapart = fpgapart

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            if not node.domain == "finnbrainsmith.custom_op.fpgadataflow":
                continue
            node_ind += 1

            new_node = specialise_factory()
            graph.node.insert(node_ind, new_node)

            # remove old nodes
            graph.node.remove(node)
            graph_modified = True
        return (model, graph_modified)


