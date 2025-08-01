############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Set pumped compute attribute for hardware operations."""

from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry
from brainsmith.core.plugins import transform

@transform(
    name="SetPumpedCompute",
    stage="kernel_opt",
    description="Set pumped compute attribute for MVAUs and DynMatMuls",
    author="Shane Fleming"
)
class SetPumpedCompute(Transformation):
    """For all MVAUs and DynMatMuls set the pumped compute attribute"""
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if (node.op_type == "MVAU_rtl"):
                inst = registry.getCustomOp(node)
                inst.set_nodeattr("pumpedCompute", 1)
        return (model, False)