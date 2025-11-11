############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""Insert DuplicateStreams layers for tensor fanout."""

from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes


class InsertDuplicateStreams(Transformation):
    """Insert DuplicateStreams HW layer for any tensor with fanout >= 2.

    Unlike node-pattern inference (Add→ChannelwiseOp), this is a graph-level
    transform that detects multi-consumer tensors and inserts routing layers.

    Usage:
        model = model.transform(InsertDuplicateStreams())

    Example:
        Before:
            Conv → tensor_X → [Add_1, Add_2, Mul_3]

        After:
            Conv → tensor_X → DuplicateStreams → [clone_0, clone_1, clone_2]
                                                    ↓        ↓        ↓
                                                  Add_1   Add_2    Mul_3
    """

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper):
        """Scan graph and insert DuplicateStreams for multi-consumer tensors."""
        graph = model.graph
        graph_modified = False
        node_ind = 0

        # Check if global input needs duplication
        if graph.input:
            global_input = graph.input[0].name
            if self._needs_duplication(model, global_input):
                self._insert_duplicator(model, global_input, 0)
                graph_modified = True
                node_ind += 1

        # Check all node outputs
        for node in list(graph.node):  # List copy - we'll modify graph
            for output_tensor in node.output:
                if self._needs_duplication(model, output_tensor):
                    self._insert_duplicator(model, output_tensor, node_ind)
                    graph_modified = True
                    node_ind += 1
            node_ind += 1

        # Cleanup after modifications
        if graph_modified:
            model = model.transform(SortGraph())
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)

    def _needs_duplication(self, model: ModelWrapper, tensor_name: str) -> bool:
        """Check if tensor has fanout >= 2.

        Args:
            model: ONNX model wrapper
            tensor_name: Tensor to check

        Returns:
            True if tensor feeds 2+ consumers
        """
        successors = model.find_consumers(tensor_name)
        return successors is not None and len(successors) >= 2

    def _insert_duplicator(
        self,
        model: ModelWrapper,
        output_tensor: str,
        insert_index: int
    ) -> None:
        """Insert DuplicateStreams node and rewire consumers.

        Args:
            model: ONNX model wrapper
            output_tensor: Tensor with multiple consumers
            insert_index: Where to insert new node
        """
        graph = model.graph
        successors = model.find_consumers(output_tensor)
        n_outputs = len(successors)

        # Get tensor metadata from graph
        out_shape = model.get_tensor_shape(output_tensor)
        dt = model.get_tensor_datatype(output_tensor)

        # Create clone tensors (one per consumer)
        out_tensor_clones = []
        for i in range(n_outputs):
            clone = helper.make_tensor_value_info(
                model.make_new_valueinfo_name(),
                TensorProto.FLOAT,
                out_shape
            )
            graph.value_info.append(clone)
            model.set_tensor_datatype(clone.name, dt)  # Preserve datatype
            out_tensor_clones.append(clone.name)

        # Create DuplicateStreams node
        # Modern: No shape nodeattrs, minimal attributes
        dup_node = helper.make_node(
            "DuplicateStreams",
            inputs=[output_tensor],
            outputs=out_tensor_clones,
            name=f"DuplicateStreams_{output_tensor.replace('/', '_')}",
            domain="brainsmith.kernels",
        )

        # Set backend attribute to enable specialization
        # Required by FINN's SpecializeKernel transform (line 68-76)
        dup_node.attribute.append(
            helper.make_attribute("backend", "fpgadataflow")
        )

        # Insert node into graph
        graph.node.insert(insert_index, dup_node)

        # Rewire consumers to use clone tensors
        clone_idx = 0
        for successor in successors:
            for i, succ_input in enumerate(successor.input):
                if succ_input == output_tensor:
                    successor.input[i] = out_tensor_clones[clone_idx]
                    clone_idx += 1
                    # Break inner loop - one clone per consumer connection
                    break
