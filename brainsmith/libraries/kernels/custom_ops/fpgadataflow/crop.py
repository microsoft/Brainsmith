############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Josh Monson <joshmonson@microsoft.com>
############################################################################

import numpy as np
import warnings
from onnx.helper import make_node
from qonnx.core.datatype import DataType
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp

class Crop(HWCustomOp):
    """Abstraction layer for HW Shuffle (rearrange and transpose) layers."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {
                "data_type" : ("s", True, ""),
                "height"     : ("i", True, []),
                "width"      : ("i", True, []),
                "channel_fold" : ("i", True, []),
                "crop_north" : ("i", True, []),
                "crop_east"  : ("i", True, []),
                "crop_west"  : ("i", True, []),
                "crop_south" : ("i", True, []),
                "simd"       : ("i", False, 1),
                "input_shape": ("ints", True, []),
                "output_shape": ("ints", True, [])
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("input_shape")

    def get_normal_output_shape(self, ind=0):
        return self.get_nodeattr("output_shape")

    def get_number_output_values(self):
        return np.prod(self.get_folded_output_shape()[:-1])

    def quantise_to_int(self, arr, dtype):
        raise NotImplementedError("This function is not yet immplemented.")

    def execute_node(self, context, graph):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_input_datatype(self, ind=0):
        return DataType[self.get_nodeattr("data_type")]

    def make_shape_compatible_op(self, model):
        in_shape = self.get_normal_input_shape()
        out_shape = self.get_normal_output_shape()
        return make_node(
            "Crop",
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

    def verify_node(self):
        raise NotImplementedError("This function is not yet immplemented.")

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("simd")
        return obits * simd

    def get_output_datatype(self, ind=0):
        return DataType[self.get_nodeattr("data_type")]

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        simd = self.get_nodeattr("simd")
        assert normal_oshape[-1] % simd == 0, "SIMD must divid into output dimension"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("simd")
        assert normal_ishape[-1] % simd == 0, "SIMD must divid into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)
