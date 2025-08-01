############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""Temporary shuffle sizing fix for BERT builds."""

from qonnx.transformation.base import Transformation
import qonnx.custom_op.registry as registry
from brainsmith.core.plugins import transform

@transform(
    name="TempShuffleFixer",
    stage="kernel_opt",
    description="Temporary fix for shuffle sizing in BERT builds",
    author="Shane Fleming"
)
class TempShuffleFixer(Transformation):
    """A temporary transformation that ensures that shuffles are sized correctly for the
    initial BERT builds"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph

        for node in graph.node:
            if node.op_type == "Shuffle_hls":
                inst = registry.getCustomOp(node)
                inner_moves = inst.get_nodeattr("inner_moves")
                simd = inst.get_nodeattr("SIMD")
                if (inner_moves == 1) and (simd > 1):
                    print(f"WARNING: as a safety precaution changing the shuffle where the inner dimension moves to SIMD=1 \n{node=}")
                    inst.set_nodeattr("SIMD", 1)
        return (model, False)