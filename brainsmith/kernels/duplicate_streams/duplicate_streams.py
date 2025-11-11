############################################################################
# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""DuplicateStreams hardware kernel - stream fanout routing."""


from onnx import NodeProto
from qonnx.core.modelwrapper import ModelWrapper

import brainsmith.dataflow as df
from brainsmith.dataflow import FULL_DIM, KernelOp
from brainsmith.dataflow.spec_helpers import derive_dim
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.registry import kernel


@kernel(
    description="Stream duplication for tensor fanout (1 input → N outputs)",
    author="Migrated from AMD FINN",
    metadata={"category": "routing"},
    is_infrastructure=True,  # Inserted by topology transforms (not pattern matching)
)
class DuplicateStreams(KernelOp):
    """Hardware kernel for stream duplication (fanout routing).

    Duplicates a single input stream to N output streams. Unlike typical
    kernels with fixed I/O counts, DuplicateStreams has variable output
    count determined at graph construction time.

    Properties:
    - Preserves shape across all outputs (identical to input)
    - Preserves datatype (routing, no computation)
    - All outputs stream in parallel (fanout, not sequential)

    Schema Notes:
    - Uses dynamic schema - inspects node.output to determine output count
    - Each output is identical to input (shape, datatype, tiling)

    Example:
        Input:  [1, 64, 64, 128] INT8, PE=16
        Output0: [1, 64, 64, 128] INT8, PE=16
        Output1: [1, 64, 64, 128] INT8, PE=16
        Output2: [1, 64, 64, 128] INT8, PE=16
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    # ================================================================
    # Schema (Required by KernelOp)
    # ================================================================

    @classmethod
    def build_schema(
        cls,
        node: NodeProto,
        model: ModelWrapper | None
    ) -> df.KernelSchema:
        """Build schema with dynamic output count from node structure.

        This is a DYNAMIC SCHEMA - inspects node.output to determine count.
        Unlike static schemas (ChannelwiseOp), the output list is built
        from the actual ONNX node structure.

        Args:
            node: ONNX node (provides output count via len(node.output))
            model: Optional ModelWrapper (unused - no validation context needed)

        Returns:
            KernelSchema with N outputs matching node.output
        """
        inputs=[
            df.InputSchema(
                name="input",
                block_tiling=[FULL_DIM],      # Process full dimensions
                stream_tiling=["PE"],          # Channel parallelism
            ),
        ]

        num_outputs = len(node.output)

        # Build output schemas dynamically (all identical to input)
        outputs = [
            df.OutputSchema(
                name=f"output{i}",
                block_tiling=[FULL_DIM],          # Same as input
                stream_tiling=[derive_dim("input", ShapeHierarchy.STREAM, -1)],    # Match input PE
                datatype="input",                  # Passthrough datatype
            )
            for i in range(num_outputs)
        ]

        return df.KernelSchema(
            name="DuplicateStreams",
            inputs=inputs,
            outputs=outputs,
            constraints=[
                # No special constraints - pure routing
            ],
        )

    # ================================================================
    # ONNX Shape Compatibility
    # ================================================================

    def make_shape_compatible_op(self, model):
        """Create ONNX-compatible op for shape inference.

        DuplicateStreams is a 1→N fanout. We use Split for compatibility,
        though it's not semantically identical (Split divides, we duplicate).
        """
        from onnx import helper

        # Use Split with all outputs
        return helper.make_node(
            "Split",
            inputs=[self.onnx_node.input[0]],
            outputs=list(self.onnx_node.output),
            axis=-1  # Split along channel (not actually splitting, just for shape)
        )

    # ================================================================
    # Execution (Reference Implementation)
    # ================================================================

    def execute_node(self, context, graph):
        """Execute stream duplication using NumPy.

        Args:
            context: Execution context (tensor name → numpy array)
            graph: ONNX graph
        """
        node = self.onnx_node
        inp = context[node.input[0]]

        # Duplicate input to all outputs
        for outp in node.output:
            context[outp] = inp.copy()  # Duplicate (not share reference)
