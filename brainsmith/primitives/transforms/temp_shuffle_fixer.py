############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""
Temporary shuffle sizing fix for BERT builds.

#TAFK TODO: Remove this temporary fix once proper shuffle sizing is implemented.
"""

import logging

import qonnx.custom_op.registry as registry
from qonnx.transformation.base import Transformation

logger = logging.getLogger(__name__)


class TempShuffleFixer(Transformation):
    """A temporary transformation that ensures that shuffles are sized correctly for the
    initial BERT builds"""

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False

        for node in graph.node:
            if node.op_type == "Shuffle_hls":
                inst = registry.getCustomOp(node)
                inner_moves = inst.get_nodeattr("inner_moves")
                simd = inst.get_nodeattr("SIMD")
                if (inner_moves == 1) and (simd > 1):
                    logger.warning(
                        "Safety precaution: changing shuffle SIMD to 1 where inner_moves=1 (node: %s)",
                        node.name,
                    )
                    inst.set_nodeattr("SIMD", 1)
                    graph_modified = True
        return (model, graph_modified)
