from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.remove import RemoveUnusedNodes
from qonnx.util.basic import get_by_name


class InferRoPE(Transformation):
    """Convert any Rotary Position Embedding (RoPE) node
    to the HW implementation layer.
    Supported RoPE node: https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.RotaryEmbedding
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for node in graph.node:
            node_ind += 1
            if node.op_type == "RotaryEmbedding" and node.domain == "com.microsoft":
                # Get inputs/output
                act_in = node.input[0]
                pos_ids = node.input[1]
                cos = node.input[2]
                sin = node.input[3]
                act_out = node.output[0]
                # Get attributes

                # Interleaved Selects between adjacent and non-adjacent pairs of sin/cos
                interleaved = get_by_name(node.attribute, "interleaved").i
                # Are are sequences padded to the same length
                # packed_batch = get_by_name(node.attribute, "is_packed_batching").i
                num_heads = get_by_name(node.attribute, "num_heads").i
                rot_dim = get_by_name(node.attribute, "rotary_embedding_dim").i
                scale = get_by_name(node.attribute, "scale").f

                # Get custom attributes (not in Op by default)
                #rope_theta = get_by_name(node).get_nodeattr("RopeTheta")
                # Get any needed tensor info
                shape_in = model.get_tensor_shape(act_in)
                idt = model.get_tensor_datatype(act_in)
                odt = model.get_tensor_datatype(act_out)

                # We assume input order [batch, seq_len, hidden_dim]
                # TODO: Add logic to handle NCW cases
                seq_len=shape_in[-2]
                hidden_dim=shape_in[-1]

                # Create node with no parallelization first
                simd = 1
                assert hidden_dim % simd == 0, "Requirement channel divisable by SIMD is violated."
                # Create and insert node
                new_node = helper.make_node(
                    "RotaryEmbedding",
                    [act_in, cos, sin],
                    [act_out],
                    domain="finn.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    SequenceLength=seq_len,
                    HiddenDimension=hidden_dim,
                    HeadDimension=hidden_dim//num_heads,
                    NumHeads=num_heads,
                    simd=simd,
                    inputDataType=idt.name,
                    outputDataType=odt.name,
                    name="RotaryEmbedding_" + node.name,
                )
                graph.node.insert(node_ind, new_node)
                # remove old node
                graph.node.remove(node)
                graph_modified = True
                model = model.transform(RemoveUnusedNodes())

        return (model, graph_modified)
