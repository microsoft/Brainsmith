############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
from onnx import helper
from qonnx.transformation.base import Transformation
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import get_by_name


def elements_are_consecutive(indices):
    if indices.size == 1:
        return True
    else:
        indices.sort()
        return np.all(np.diff(indices) == 1)


class InferCropFromGather(Transformation):
    """Transform Gather nodes with consecutive indices into Crop nodes.

    Supports spatial cropping on height and width dimensions in NHWC layout:
    - Gather(axis=1, indices=[start:end]) → Crop with crop_north/crop_south
    - Gather(axis=2, indices=[start:end]) → Crop with crop_east/crop_west

    Requirements:
    - Data input must be dynamic (not an initializer)
    - Indices input must be static (initializer)
    - Indices must be consecutive integers
    - Axis must be 1 (height) or 2 (width) for NHWC 4D tensors
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

                # Get input shape and axis
                input_shape = model.get_tensor_shape(n.input[0])
                axis_attr = get_by_name(n.attribute, "axis")
                axis = axis_attr.i if axis_attr else 0

                # Normalize negative axis
                if axis < 0:
                    axis = len(input_shape) + axis

                # Validate axis is spatial dimension (height or width) in NHWC layout
                # NHWC: [N=0, H=1, W=2, C=3]
                # Only support axis=1 (height) or axis=2 (width)
                if len(input_shape) == 4:
                    assert axis in [1, 2], f"Crop only supports axis=1 (height) or axis=2 (width) in NHWC layout, got axis={axis}"
                else:
                    # For non-4D tensors, support two innermost spatial dimensions
                    max_index = len(input_shape) - 1
                    assert axis in [max_index - 2, max_index - 1], f"Crop only supports two innermost spatial dimensions"

                # Get output shape
                output_shape = model.get_tensor_shape(n.output[0])

                # Validate indices
                indices = model.get_initializer(n.input[1])
                assert indices is not None, "Indices must be an initializer"
                assert indices.dtype == np.int64, "Indices must be int64"
                indices_flat = indices.flatten()
                assert elements_are_consecutive(indices_flat), "Indices must be consecutive"

                # Validate SIMD divisibility
                channels = input_shape[-1]
                assert channels % self.simd == 0, f"Channels ({channels}) must be divisible by SIMD ({self.simd})"

                # Compute crop parameters based on axis
                min_idx = int(np.min(indices_flat))
                max_idx = int(np.max(indices_flat))

                if axis == 1:  # Height dimension in NHWC
                    # Cropping height: remove rows from top and bottom
                    crop_north = min_idx
                    crop_south = input_shape[axis] - max_idx - 1
                    crop_east = 0
                    crop_west = 0
                elif axis == 2:  # Width dimension in NHWC
                    # Cropping width: remove columns from left and right
                    crop_north = 0
                    crop_south = 0
                    crop_west = min_idx
                    crop_east = input_shape[axis] - max_idx - 1
                else:
                    # For non-4D tensors, map to innermost spatial dimensions
                    # This maintains backward compatibility
                    max_index = len(input_shape) - 1
                    if axis == max_index - 1:  # Width-like dimension
                        crop_north = 0
                        crop_south = 0
                        crop_west = min_idx
                        crop_east = input_shape[axis] - max_idx - 1
                    else:  # axis == max_index - 2, height-like dimension
                        crop_north = min_idx
                        crop_south = input_shape[axis] - max_idx - 1
                        crop_east = 0
                        crop_west = 0

                idt0 = model.get_tensor_datatype(n.input[0])
                odt0 = model.get_tensor_datatype(n.output[0])

                # Extract height and width for legacy node attributes
                # NHWC layout: [N, H, W, C]
                height = input_shape[-2]
                width = input_shape[-1]

                # create and insert new node
                new_node = helper.make_node(
                    "Crop",
                    [n.input[0]],  # input tensor(s)
                    [n.output[0]],  # output tensor(s)
                    domain="brainsmith.kernels",
                    backend="fpgadataflow",
                    data_type=idt0.name,
                    name="Crop" + n.name,
                    simd=self.simd,
                    height=height,
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
