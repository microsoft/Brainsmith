import numpy as np
import qonnx.core.data_layout as DataLayout
import warnings
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.base import Transformation
from qonnx.transformation.general import SortGraph
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from qonnx.util.onnx import nchw_to_nhwc
from finnbrainsmith.transformation.shuffle_helpers import shuffle_perfect_loopnest_coeffs
from finnbrainsmith.transformation.shuffle_helpers import innerloop_moves 


class InferShuffle(Transformation):
    """
    Find transpose layers with (optionally) reshape layers around them
    and convert them into a shuffle operator
    """
    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        graph_modified = False
        node_ind = 0
        for n in graph.node:
            node_ind += 1 # Do I really need to track this? Isn't there a better way?
            if(n.op_type == "Transpose"):
                to_remove = [n]

                new_in_tensor = None
                new_out_tensor = None

                perm = n.attribute[0] 

                new_in_tensor = n.input[0]
                in_shape = model.get_tensor_shape(n.input[0]) 
                in_reshaped = in_shape 

                # Detect a reshape at the input and capture it
                producer = model.find_producer(n.input[0])
                if producer is not None:
                    if ( producer.op_type == "Reshape" ):
                        new_in_tensor = producer.input[0]
                        in_shape = model.get_tensor_shape(new_in_tensor) 
                        in_reshaped = model.get_tensor_shape(n.input[0]) 
                        to_remove.append(producer)
                        node_ind -= 1

                new_out_tensor = n.output[0] 
                out_shape = model.get_tensor_shape(new_out_tensor)
                out_reshaped = out_shape

                # Detect a reshape at the output and capture it
                consumer = model.find_consumer(n.output[0])
                if consumer is not None:
                    if ( consumer.op_type == "Reshape" ):
                        new_out_tensor = consumer.output[0]
                        out_shape = model.get_tensor_shape(n.output[0])
                        out_reshaped = model.get_tensor_shape(new_out_tensor)
                        to_remove.append(consumer)
                        node_ind -= 1

                idt = model.get_tensor_datatype(new_in_tensor)
                odt = model.get_tensor_datatype(new_out_tensor)

                # Some sanity checks for the transformation
                if(idt != odt): 
                    raise RuntimeError(f"""
                    Input datatype and output datatype of the shuffle must be the same, 
                    did something go wrong during transformation?
                """)

                if (len(perm.ints) != len(in_reshaped)):
                    raise RuntimeError(f"""
                    Permutation list {perm.ints=} does not match the reshaped input dimension {in_reshaped=}
                """)

                if (len(perm.ints) != len(out_shape)):
                    raise RuntimeError(f"""
                    Permutation list {perm.ints=} does not match the reshaped out dimension {out_reshaped=}
                """)

                simd = 1 
                new_node = helper.make_node(
                            "Shuffle",
                            [new_in_tensor],
                            [new_out_tensor],
                            domain="finnbrainsmith.custom_op.fpgadataflow",
                            backend="fpgadataflow",
                            in_shape=in_shape,
                            in_reshaped=in_reshaped,
                            out_shape=out_shape,
                            out_reshaped=out_reshaped,
                            data_type=idt.name,
                            name=f"Shuffle_{n.name}",
                            loop_coeffs=shuffle_perfect_loopnest_coeffs(shape=in_reshaped, perm=perm.ints),
                            inner_moves=innerloop_moves(shape=in_reshaped, perm=list(perm.ints)),
                            simd=simd
                        )
                new_node.attribute.extend([perm])
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i) # Is this okay to do while iterating? (QuantSoftMax does...)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())

        return (model, graph_modified)


class InferQuantSoftmax(Transformation):
    """
    Find softmax layers that are followed by a MultiThreshold layer
    and replace them with QuantizedSoftmax
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            # check that an optype of Softmax is present followed by a MultiThreshold
            consumer = model.find_consumer(n.output[0])
            if (
                n.op_type == "Softmax"
                and consumer is not None
                and consumer.op_type == "MultiThreshold"
            ):
                # get the shape of the input/output tensor
                input_shape = model.get_tensor_shape(n.input[0])
                assert input_shape == model.get_tensor_shape(
                    consumer.input[0]
                ), "Softmax and MultiThreshold input shapes do not match"
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(consumer.output[0])
                # create node with no parallelization first
                simd = 1
                # create and insert new node
                new_node = helper.make_node(
                    "QuantSoftmax",
                    [n.input[0]],  # input tensor(s)
                    [consumer.output[0]],  # output tensor(s)
                    domain="finnbrainsmith.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name="Quant" + n.name,
                    simd=simd,
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                # remove multithreshold too
                graph.node.remove(consumer)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

