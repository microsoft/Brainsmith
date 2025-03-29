############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

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
                            SIMD=simd,
                            
                            NumChannels=in_reshaped[-1]
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

class InferHWSoftmax(Transformation):
    """
    Infers a regular softmax node without merging the multithreshold
    and setting the softmax to perform the quantisation.
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "Softmax":
                input_shape = model.get_tensor_shape(n.input[0])
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWSoftmax",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finnbrainsmith.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    input_data_type=idt0.name,
                    output_data_type=odt0.name,
                    name=n.name,
                    SIMD=1,
                    NumChannels=input_shape[-1],
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

class InferHWReduceSum(Transformation):
    """
    Infers a HWReduceSum operator from a ReduceSum operator node
    """

    def __init__(self):
        super().__init__()

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        for n in graph.node:
            if n.op_type == "ReduceSum":
                input_shape = model.get_tensor_shape(n.input[0])
                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])
                new_node = helper.make_node(
                    "HWReduceSum",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="finnbrainsmith.custom_op.fpgadataflow",
                    backend="fpgadataflow",
                    ifm_dim=input_shape,
                    data_type=idt0.name,
                    name=n.name,
                    SIMD=1
                )
                graph.node.insert(node_ind, new_node)
                graph.node.remove(n)
                graph_modified = True

        if graph_modified:
            model = model.transform(InferShapes())
            model = model.transform(InferDataTypes())
        return (model, graph_modified)

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
                    domain="finnbrainsmith.custom_op.fpgadataflow",
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


def elements_are_consecutive(indices):
    if indices.size == 1:
        return True
    else:
        indices.sort()
        return np.all(np.diff(indices) == 1)


class InferCropFromGather(Transformation):
    """
    Find gather layers that can be converted into a Crop layer
    and replace them with a Crop layer
    """

    def __init__(self, simd= 1):
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
                    domain="finnbrainsmith.custom_op.fpgadataflow",
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
