############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Set pumped compute attribute for hardware operations."""

import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation


class SetPumpedCompute(Transformation):
    """For all MVAUs and DynMatMuls set the pumped compute attribute"""
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in graph.node:
            if node.op_type == "MVAU_rtl":
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
                graph_modified = True
        return (model, graph_modified)
