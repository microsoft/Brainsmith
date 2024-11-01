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
                    domain="finn.custom_op.fpgadataflow",
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

