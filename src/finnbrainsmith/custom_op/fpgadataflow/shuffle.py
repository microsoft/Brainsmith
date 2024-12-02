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
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("in_shape")

    def get_normal_output_shape(self, ind=0):
        pass

    def get_number_output_values(self):
        pass

    def quantise_to_int(self, arr, dtype):
        pass

    def execute_node(self, context, graph):
        pass

    def get_input_datatype(self, ind=0):
        pass

    def make_shape_compatible_op(self, model):
        pass

    def infer_node_datatype(self, model):
        pass

    def verify_node(self):
        raise NotImplementedError

    def get_instream_width(self, ind=0):
        pass

    def get_outstream_width(self, ind=0):
        pass

    def get_output_datatype(self, ind=0):
        pass

    def get_folded_output_shape(self, ind=0):
        pass

    def get_folded_input_shape(self, ind=0):
        pass
