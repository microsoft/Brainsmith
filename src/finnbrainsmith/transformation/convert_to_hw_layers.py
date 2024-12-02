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
            if(n.op_type == "Transpose"):
                node_ind += 1 # Do I really need to track this? Isn't there a better way?
                print(f"Found a transpose that we need to convert to a shuffle {n.name}")
                to_remove = [n]

                new_in_tensor = None
                new_out_tensor = None

                perm = n.attribute[0] 

                # Detect a reshape at the input
                producer = model.find_producer(n.input[0])
                if ( producer.op_type == "Reshape" ):
                    new_in_tensor = model.find_producer(producer.input[0]).output[0]
                    in_shape = model.get_tensor_shape(model.find_producer(producer.input[0]).output[0]) 
                    in_reshaped = model.get_tensor_shape(producer.output[0])
                    to_remove.append(producer)
                    node_ind -= 1
                    print(f"\tIt has a reshape at it's input {producer.name}")
                else:
                    new_in_tensor = n.input[0]
                    in_shape = model.get_tensor_shape(n.input[0]) #TODO:What if a producer has multiple outputs?
                    in_reshaped = in_shape
                print(f"\t{in_shape=} {in_reshaped=}")

                # Detect a reshape at the output
                consumer = model.find_consumer(n.output[0])
                out_shape = model.get_tensor_shape(n.output[0]) 
                if ( consumer.op_type == "Reshape" ):
                    out_reshape = model.get_tensor_shape(consumer.output[0]) 
                    new_out_tensor = consumer.output[0]
                    to_remove.append(consumer)
                    node_ind += 1
                    print(f"\tIt has a reshape at it's output {consumer.name}")
                else:
                    out_reshaped = out_shape
                    new_out_tensor = n.output[0] 
                print(f"\t{out_shape=} {out_reshaped=}")

                idt = model.get_tensor_datatype(new_in_tensor)
                odt = model.get_tensor_datatype(new_out_tensor)
                assert idt == odt, "Input datatype and output datatype of the shuffle must be the same, did something go wrong during transformation?"

                simd = 1 # TODO: allow for this to be increased
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
                            simd=simd
                        )
                new_node.attribute.extend([perm])
                graph.node.insert(node_ind, new_node)

                for i in to_remove:
                    graph.node.remove(i) # Is this okay to do while iterating? (QuantSoftMax does...)
                graph_modified = True

        #if graph_modified:
        #    model = model.transform(InferShapes())
        #    model = model.transform(InferDataTypes())

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

