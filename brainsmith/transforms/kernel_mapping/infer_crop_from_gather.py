"""Crop hardware inference transform from Gather operations."""

import numpy as np
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name
from brainsmith.plugin.decorators import transform


def elements_are_consecutive(indices):
    if indices.size == 1:
        return True
    else:
        indices.sort()
        return np.all(np.diff(indices) == 1)


@transform(
    name="InferCropFromGather",
    stage="kernel_mapping",
    description="Convert Gather operations to Crop hardware operations",
    author="shane.fleming",
    version="1.0.0",
    requires=["qonnx", "onnx", "numpy"]
)
class InferCropFromGather(Transformation):
    """
    Find gather layers that can be converted into a Crop layer
    and replace them with a Crop layer
    """

    def __init__(self, simd=1):
        super().__init__()
        self.simd = simd

    def is_initializer(self, tensor_name, model):
        return model.get_initializer(tensor_name) is not None

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            node_ind += 1
            consumer = model.find_consumer(n.output[0])
            if n.op_type == "Gather":

                # check if the data input is a streaming tensor (i.e. not an initializer)
                if self.is_initializer(n.input[0], model):
                    continue
                # ensure that the indices input is an initializer
                if not self.is_initializer(n.input[1], model):
                    continue

                # ensure that the axis is among the two innermost dimensions
                input_shape = model.get_tensor_shape(n.input[0])
                max_index = len(input_shape) - 1
                axis = get_by_name(n.attribute, "axis").i
                assert axis in [max_index, max_index - 1], "Crop Operates on two innermost dimensions"
                is_vertical = axis == max_index # otherwise horizontal
                assert is_vertical == False, "This operator does not current support vertical crops"

                # ensure that the output shape matches the expected output shape
                output_shape = model.get_tensor_shape(n.output[0])

                # assume that the indices input is an int64 scalar
                indices = model.get_initializer(n.input[1])
                assert indices.dtype == np.int64, "Indices must be int64 scalar"
                assert elements_are_consecutive(indices[0]), "Indices must be consecutive"

                # set the number of pixels to crop off each edge
                width =  input_shape[-1]
                assert width % self.simd == 0, "Width must be divisible by SIMD"
                crop_north = int(np.min(indices))
                crop_south = input_shape[axis] - int(np.max(indices)) - 1
                crop_east = 0
                crop_west = 0

                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])

                # create and insert new node
                new_node = helper.make_node(
                    "Crop",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="brainsmith.libraries.kernels.crop",
                    backend="fpgadataflow",
                    data_type=idt0.name,
                    name="Crop" + n.name,
                    simd=self.simd,
                    height=input_shape[-2],
                    width=width,
                    channel_fold=1,
                    crop_north=crop_north,
                    crop_east=crop_east,
                    crop_west=crop_west,
                    crop_south=crop_south,
                    input_shape=input_shape,
                    output_shape=output_shape,
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                # remove multithreshold too
                #graph.node.remove(consumer)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)