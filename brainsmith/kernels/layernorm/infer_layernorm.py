"""LayerNorm hardware inference transform."""

import qonnx.core.data_layout as DataLayout
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.onnx import nchw_to_nhwc
from brainsmith.plugin.core import transform


@transform(name="InferLayerNorm", kernel="LayerNorm", stage=None,
    description="Convert FuncLayerNorm to LayerNorm hardware operations",
    author="shane.fleming",
    version="1.0.0",
    requires=["qonnx", "onnx"]
)
class InferLayerNorm(Transformation):
    """Convert LayerNorm into HW, only norming over channel dim"""

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "FuncLayerNorm":
                act_in = node.input[0]
                act_out = node.output[0]
                # Get any shape info that needs reuse
                shape_in = model.get_tensor_shape(act_in)
                # Get datatypes
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)

                norm_axis = helper.get_node_attr_value(node, "axis")
                if model.get_tensor_layout(act_in) == DataLayout.NCHW:
                    act_in = nchw_to_nhwc(act_in, model, node_ind)
                    node_ind += 1
                    shape_in = model.get_tensor_shape(act_in)
                    # shift axis for norm appropriately
                    norm_axis = (norm_axis+2)%4
                ch = shape_in[-1]

                # keep track of where we need to insert the HLS Op
                # it has to be ahead of the output transform
                insert_point = node_ind
                if model.get_tensor_layout(act_out) == DataLayout.NCHW:
                    act_out = nchw_to_nhwc(act_out, model, node_ind, reverse=True)
                    node_ind += 1

                # Check if 1D, norming on channel axis
                if not (norm_axis == -1 or norm_axis == len(shape_in)-1):
                    continue

                # create node with no parallelization first
                simd = 1
                assert ch % simd == 0, "Requirement IFC divisable by PE is violated."
                # create and insert nodes
                new_node = helper.make_node(
                    "LayerNorm",
                    [act_in],
                    [act_out],
                    domain="brainsmith.kernels.layernorm",
                    backend="fpgadataflow",
                    SIMD=simd,
                    ifm_dim=shape_in,
                    NumChannels=shape_in[-1],
                    epsilon=helper.get_node_attr_value(node, "epsilon"),
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="LayerNorm_" + node.name,
                )
                graph.node.insert(insert_point, new_node)
                # remove old node
                graph.node.remove(node)

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)