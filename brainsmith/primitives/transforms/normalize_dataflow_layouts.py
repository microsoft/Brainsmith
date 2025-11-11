"""
NormalizeDataflowLayouts transformation

Converts all NCHW (channel-first) tensors to NHWC (channel-last) layout globally,
eliminating the need for per-kernel layout checking. This ensures all dataflow
operations work with channels in the last dimension for natural streaming behavior.

Graph outputs that were originally NCHW are converted back via reverse Transposes
to maintain the original layout contract.
"""

import qonnx.core.data_layout as DataLayout
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc


class NormalizeDataflowLayouts(Transformation):
    """
    Global preprocessing transformation that normalizes all tensor layouts to NHWC.

    This transformation converts all NCHW (channel-first) tensors in the graph to
    NHWC (channel-last) layout by inserting Transpose nodes. This ensures that all
    dataflow kernel operations can assume channel-last layout without individual
    layout checks.

    The transformation preserves the original layout contract for graph outputs by
    inserting reverse Transposes where needed.

    Algorithm:
    1. Identify all tensors with NCHW layout
    2. For each NCHW tensor, insert NCHW → NHWC Transpose after its producer
    3. Update all consumer nodes to use the new NHWC tensor
    4. For graph outputs that were originally NCHW, insert NHWC → NCHW reverse Transpose

    After this transformation, all subsequent kernel inference passes can assume
    NHWC layout throughout the dataflow region.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model: ModelWrapper):
        graph = model.graph
        graph_modified = False

        # Track which tensors need conversion and which are graph outputs
        tensors_to_convert = []
        graph_output_names = {output.name for output in graph.output}

        # Preserve original datatypes for graph inputs/outputs
        # (InferDataTypes may reset these)
        original_datatypes = {}
        for inp in graph.input:
            dt = model.get_tensor_datatype(inp.name)
            if dt is not None:
                original_datatypes[inp.name] = dt
        for out in graph.output:
            dt = model.get_tensor_datatype(out.name)
            if dt is not None:
                original_datatypes[out.name] = dt

        # Phase 1: Identify all NCHW tensors
        # We need to check all value_info tensors plus graph inputs/outputs
        all_tensor_names = set()

        # Collect from value_info
        for vi in graph.value_info:
            all_tensor_names.add(vi.name)

        # Collect from graph inputs
        for inp in graph.input:
            all_tensor_names.add(inp.name)

        # Collect from graph outputs
        for out in graph.output:
            all_tensor_names.add(out.name)

        # Check layout for each tensor
        for tensor_name in all_tensor_names:
            layout = model.get_tensor_layout(tensor_name)
            # Check if layout is NCHW (comparing against the list)
            if layout == DataLayout.NCHW:
                tensors_to_convert.append(tensor_name)

        if not tensors_to_convert:
            # No NCHW tensors found, nothing to do
            return (model, graph_modified)

        # Phase 2: Convert NCHW tensors to NHWC
        # We need to process these carefully to maintain topological order

        for tensor_name in tensors_to_convert:
            # Find the producer node for this tensor
            producer = model.find_producer(tensor_name)

            if producer is None:
                # This is a graph input, handle specially
                self._convert_graph_input(model, tensor_name)
                graph_modified = True
            else:
                # Find the position to insert the Transpose
                # We want to insert it right after the producer
                producer_idx = None
                for idx, node in enumerate(graph.node):
                    if node == producer:
                        producer_idx = idx
                        break

                if producer_idx is not None:
                    # Insert Transpose after producer (NCHW → NHWC)
                    new_tensor = nchw_to_nhwc(
                        tensor_name,
                        model,
                        producer_idx + 1,  # Insert after producer
                        reverse=False,
                    )

                    # Update all consumers to use the new NHWC tensor
                    # EXCEPT if this is a graph output (we'll handle that separately)
                    if tensor_name not in graph_output_names:
                        self._redirect_consumers(model, tensor_name, new_tensor)

                    graph_modified = True

        # Phase 3: Handle graph outputs
        # If a graph output was originally NCHW, we need to insert a reverse Transpose
        # to convert it back from NHWC to NCHW to preserve the output contract

        for output in graph.output:
            if output.name in tensors_to_convert:
                # This output needs to be converted back to NCHW
                # The tensor is currently in NHWC format due to Phase 2
                # We need to insert a reverse Transpose before the output

                # Find the current producer of this output
                producer = model.find_producer(output.name)
                if producer is not None:
                    producer_idx = None
                    for idx, node in enumerate(graph.node):
                        if node == producer:
                            producer_idx = idx
                            break

                    if producer_idx is not None:
                        # The output tensor is currently NHWC
                        # We need to rename it and insert a reverse Transpose
                        original_output_name = output.name

                        # Create intermediate NHWC tensor name
                        nhwc_tensor_name = model.make_new_valueinfo_name()

                        # Update producer to output to the intermediate tensor
                        for i, out_name in enumerate(producer.output):
                            if out_name == original_output_name:
                                producer.output[i] = nhwc_tensor_name

                        # Get the shape and datatype of the original output
                        output_shape = model.get_tensor_shape(original_output_name)
                        output_dtype = model.get_tensor_datatype(original_output_name)

                        # Create the intermediate NHWC tensor
                        # Shape should be NHWC version of the original
                        bs, ch, h, w = output_shape
                        nhwc_shape = (bs, h, w, ch)

                        nhwc_tensor = helper.make_tensor_value_info(
                            nhwc_tensor_name, TensorProto.FLOAT, nhwc_shape
                        )
                        graph.value_info.append(nhwc_tensor)
                        model.set_tensor_datatype(nhwc_tensor_name, output_dtype)
                        model.set_tensor_layout(nhwc_tensor_name, DataLayout.NHWC)

                        # Insert reverse Transpose (NHWC → NCHW)
                        transpose_node = helper.make_node(
                            "Transpose",
                            [nhwc_tensor_name],
                            [original_output_name],
                            perm=[0, 3, 1, 2],  # NHWC → NCHW
                        )
                        graph.node.insert(producer_idx + 1, transpose_node)

                        # Update the output layout back to NCHW
                        model.set_tensor_layout(original_output_name, DataLayout.NCHW)

                        graph_modified = True

        # Run shape and datatype inference to update metadata
        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

            # Restore original datatypes for graph inputs/outputs
            for tensor_name, dtype in original_datatypes.items():
                model.set_tensor_datatype(tensor_name, dtype)

        return (model, graph_modified)

    def _convert_graph_input(self, model: ModelWrapper, tensor_name: str):
        """
        Convert a graph input from NCHW to NHWC by inserting a Transpose at the beginning.

        Args:
            model: The ONNX model wrapper
            tensor_name: Name of the graph input tensor to convert
        """
        graph = model.graph

        # Create new NHWC tensor
        original_shape = model.get_tensor_shape(tensor_name)
        bs, ch, h, w = original_shape
        nhwc_shape = (bs, h, w, ch)

        # Rename the original input and create a new NHWC version
        nhwc_tensor_name = model.make_new_valueinfo_name()

        nhwc_tensor = helper.make_tensor_value_info(nhwc_tensor_name, TensorProto.FLOAT, nhwc_shape)
        graph.value_info.append(nhwc_tensor)

        dtype = model.get_tensor_datatype(tensor_name)
        model.set_tensor_datatype(nhwc_tensor_name, dtype)
        model.set_tensor_layout(nhwc_tensor_name, DataLayout.NHWC)

        # Insert Transpose at the beginning (NCHW → NHWC)
        transpose_node = helper.make_node(
            "Transpose",
            [tensor_name],
            [nhwc_tensor_name],
            perm=[0, 2, 3, 1],  # NCHW → NHWC
        )
        graph.node.insert(0, transpose_node)

        # Redirect all consumers of the original input to use the NHWC version
        self._redirect_consumers(model, tensor_name, nhwc_tensor_name)

    def _redirect_consumers(self, model: ModelWrapper, old_tensor: str, new_tensor: str):
        """
        Redirect all consumers of old_tensor to use new_tensor instead.

        Args:
            model: The ONNX model wrapper
            old_tensor: Original tensor name
            new_tensor: New tensor name to redirect to
        """
        graph = model.graph

        for node in graph.node:
            # Update all input references
            for i, input_name in enumerate(node.input):
                if input_name == old_tensor:
                    node.input[i] = new_tensor
