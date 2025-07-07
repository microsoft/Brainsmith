############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import torch
import numpy as np
import torch.nn.functional as F
from qonnx.core.datatype import DataType
import warnings
import textwrap

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.custom_op.registry import register_op
from brainsmith.core.plugins import kernel

# TODO: Explain any shape assumptions -- TAFK

@kernel(
    name="LayerNorm",
    description="Hardware implementation of LayerNorm",
    author="thomas-keller",
    version="1.0.0"
)
@register_op(domain="brainsmith.kernels.layernorm", op_type="LayerNorm")
class LayerNorm(HWCustomOp):
    """Abstraction layer for HW implementation of the LayerNorm layer."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = super().get_nodeattr_types()
        my_attrs.update({
            "SIMD": ("i", True, 0),
            "NumChannels": ("i", True, 128),
            "ifm_dim": ("ints", True, []),
            "epsilon": ("f", True, 1e-5),
            # FINN DataTypes for inputs, weight, bias, outputs
            "inputDataType": ("s", True, ""),
            "outputDataType": ("s", True, ""),
            # Possible execution modes for simulating this node
            #   Note: Override to support python mode
            "exec_mode": (
                "s", False, "python", {"", "rtlsim", "cppsim", "python"}
            ),
        })
        return my_attrs

    def execute_node(self, context, graph):
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")

        if mode == "python":
            self._execute_node_python(context, graph)

    # Executes elementwise operation in python
    def _execute_node_python(self, context, graph):
        node = self.onnx_node
        # Get tensor values
        in_values = context[node.input[0]]
        out_values = context[node.output[0]]
        # Get any shape info that needs reuse
        ishape = in_values.shape
        oshape = out_values.shape
        # Functionally verify with PyTorch implementation, since weight & bias are removed
        in_act = torch.from_numpy(in_values)
        out_act = F.layer_norm(in_act, [ishape[-1]], eps=self.get_nodeattr("epsilon"))
        context[node.output[0]] = np.asarray(out_act, dtype=np.float32).reshape(oshape)

    # Verifies the node attributes, inputs and outputs
    def verify_node(self):
        # TODO: Implement
        pass

    def get_normal_input_shape(self, ind=0):
        return self.get_nodeattr("ifm_dim")

    def get_normal_output_shape(self, ind=0):
        return self.get_normal_input_shape()

    def get_folded_input_shape(self, ind=0):
        # even though there is no folding in the current hlslib op,
        # insert a time multiplexing axis to remain compatible with the
        # shapes produced by the rest of the dataflow pipeline
        normal_ishape = list(self.get_normal_input_shape())
        simd = self.get_nodeattr("SIMD")
        assert normal_ishape[-1] % simd == 0, "SIMD must divide into input dimension"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        return self.get_folded_input_shape()

    def get_number_output_values(self):
        nf = np.prod(self.get_folded_output_shape()[:-1])
        return nf

    def make_shape_compatible_op(self, model):
        return super().make_const_shape_op(self.get_normal_input_shape())

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        if ind == 0:
            return DataType[self.get_nodeattr("inputDataType")]
        else:
            raise Exception("Undefined input ind for this layer type")

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output."""
        return DataType[self.get_nodeattr("outputDataType")]

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for %s: %s -> %s " % (
                node.name,
                str(self.get_input_datatype()),
                str(idt),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def get_instream_width(self, ind=0):
        i_bits = self.get_input_datatype().bitwidth()
        in_width = i_bits * self.get_nodeattr("SIMD")
        return in_width

    def get_outstream_width(self, ind=0):
        o_bits = self.get_output_datatype().bitwidth()
        out_width = o_bits * self.get_nodeattr("SIMD")
        return out_width

    #def calc_wmem(self):
    #    """Calculates and returns WMEM."""
    #    pass

    #def calc_tmem(self):
    #    """Calculates and returns TMEM."""
    #    pass

    #def uram_estimation(self):
    #    pass

    #def bram_estimation(self):
    #    pass

    #def bram_efficiency_estimation(self):
    #    pass

    #def uram_efficiency_estimation(self):
    #    """Function for URAM efficiency estimation: actual parameter storage
    #    needed divided by the allocated URAM storage (from estimation)"""
    #    pass

    #def minimize_accumulator_width(self, model):
    #    """Minimize the accumulator bit width according to the weight values,
    #    input data types, and size of dot product"""
    #    pass

    #def generate_params(self, model, path):
    #    pass

    #def get_op_and_param_counts(self):
    #    pass

    #def derive_characteristic_fxns(self, period):
    #    pass

