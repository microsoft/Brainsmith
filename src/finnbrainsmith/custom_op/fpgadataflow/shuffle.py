############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import warnings
from onnx.helper import make_node
from qonnx.core.datatype import DataType
from scipy.special import softmax

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Shuffle(HWCustomOp):
    """Abstraction layer for HW Shuffle (rearrange and transpose) layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
                "data_type" : ("s", True, ""),
                "in_reshaped" : ("ints", True, []),
                "in_shape" : ("ints", True, []),
                "out_reshaped" : ("ints", True, []),
                "out_shape" : ("ints", True, []),
                "loop_coeffs" : ("ints", True, []), 
                "inner_moves" : ("i", True, 0),
                "perm" : ("ints", True, []),
                "SIMD": ("i", False, 1),
                "NumChannels": ("i", False, 128) 
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_reshaped")

    def get_normal_output_shape(self, ind=0):
        return self.get_nodeattr("out_reshaped")

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1]) # [STF] Not sure this is correct...

    def execute_node(self, context, graph):
        node = self.onnx_node
        input_data = context[node.input[0]]
        input_reshaped = input_data.reshape(self.get_nodeattr("in_reshaped")) 
        transposed = np.transpose(input_reshaped, axes=self.get_nodeattr("perm"))
        output_reshaped = transposed.reshape(self.get_nodeattr("out_reshaped"))
        context[node.output[0]] = output_reshaped

    def get_input_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def make_shape_compatible_op(self, model):
        in_shape = self.get_normal_input_shape()
        out_shape = self.get_normal_output_shape()
        return make_node(
            "Shuffle",
            inputs=[self.onnx_node.input[0]],
            outputs=[self.onnx_node.output[0]],
            in_shape=list(in_shape),
            out_shape=list(out_shape)
        )

    def infer_node_datatype(self, model):
        node = self.onnx_node
        dt = model.get_tensor_datatype(node.input[0])
        if dt != self.get_input_datatype():
            warn_str = f"data_type changing for {node.name}: {str(self.get_input_datatype())} -> {str(dt)}"
            warnings.warn(warn_str)
        self.set_nodeattr("data_type", dt.name)
        model.set_tensor_datatype(node.output[0], dt)

    def verify_node(self):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_output_datatype(self, ind=0):
        data_type = DataType[self.get_nodeattr("data_type")]
        return data_type

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_oshape[-1] % simd == 0, "SIMD must divid into output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divid into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)
