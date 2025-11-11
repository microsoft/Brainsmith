# Copyright (C) 2023, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import warnings

import numpy as np
import onnx
import onnxruntime as ort
from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp
from qonnx.core.datatype import DataType


def get_rope_onnx_filename(theta_base, batch_size, num_attention_heads, seq_len, head_size):
    return f"rope_rt{theta_base}_b{batch_size}_nh{num_attention_heads}_s{seq_len}_hs{head_size}.onnx"

class RotaryEmbedding(HWCustomOp):
    """Abstraction layer for HW impplementation of RotaryEmbedding.
    Pads input image by given amount."""

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

        # Onnx Filename is always generated with batch size 1
        # since RTL generation is not affected by batch size
        batch_size = 1
        self.onnx_filename = get_rope_onnx_filename(self.get_nodeattr("RopeTheta"), batch_size,
                                                    self.get_nodeattr("NumHeads"),
                                                    self.get_nodeattr("SequenceLength"),
                                                    self.get_nodeattr("HeadDimension"))
        self.onnx_dir = os.path.dirname(__file__) + "/rotaryembedding/onnxgraphs/"
        self.onnx_path = self.onnx_dir + self.onnx_filename
        self.onnx_model = onnx.load(self.onnx_path)
        onnx.checker.check_model(self.onnx_model)

        # debug_names = ["/Concat_1", "/Mul_2", "/Mul_3"]
        # print(f"Adding intermediate tensors as outputs for debugging: {debug_names}")
        # for node in self.onnx_model.graph.node:
        #     if node.name in debug_names:
        #         print(f"Node {node.name} has output {node.output}")
        #         for output in node.output:
        #             intermediate_tensor = onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None)
        #             self.onnx_model.graph.output.append(intermediate_tensor)

        # onnx.shape_inference.infer_shapes(self.onnx_model)

        # onnx.save(self.onnx_model, self.onnx_dir + "debug.onnx")

    def get_nodeattr_types(self):
        my_attrs = {
            # Sequence Length of the Input
            "SequenceLength": ("i", True, 0),
            # hidden dimension of the input
            "HiddenDimension": ("i", True, 0),
            # head dimension of the input
            "HeadDimension": ("i", True, 0),
            # Number of attention heads
            "NumHeads": ("i", True, 0),
            # Rope Theta
            "RopeTheta": ("f", True, 10000.0),

            # SIMD Input parallelism
            "SIMD": ("i", False, 1),
            # FINN input datatype
            "inputDataType": ("s", True, ""),
            "weightDataType": ("s", True, ""),
            # shape describing input vecs per execution
            "numInputVectors": ("i", False, 1),
        }
        my_attrs.update(super().get_nodeattr_types())
        return my_attrs


    def get_exp_cycles(self):
        return 0

    def get_normal_input_shape(self, ind=0):
        seq_len  = self.get_nodeattr("SequenceLength")
        head_dim = self.get_nodeattr("HeadDimension")
        num_heads = self.get_nodeattr("NumHeads")
        ishape = (1, num_heads, seq_len, head_dim)
        return ishape

    def get_normal_output_shape(self, ind=0):
        seq_len  = self.get_nodeattr("SequenceLength")
        head_dim = self.get_nodeattr("HeadDimension")
        num_heads = self.get_nodeattr("NumHeads")
        oshape = (1, num_heads, seq_len, head_dim)
        return oshape

    def get_folded_input_shape(self, ind=0):
        normal_ishape = list(self.get_normal_input_shape())
        head_dim = self.get_nodeattr("HeadDimension")
        simd = self.get_nodeattr("SIMD")
        assert head_dim % simd == 0, "SIMD must divide input channels"
        fold = int(normal_ishape[-1] / simd)
        folded_ishape = normal_ishape[:-1] + [fold, simd]
        return tuple(folded_ishape)

    def get_folded_output_shape(self, ind=0):
        normal_oshape = list(self.get_normal_output_shape())
        head_dim = self.get_nodeattr("HeadDimension")
        simd = self.get_nodeattr("SIMD")
        assert head_dim % simd == 0, "SIMD must divide input channels"
        fold = int(normal_oshape[-1] / simd)
        folded_oshape = normal_oshape[:-1] + [fold, simd]
        return tuple(folded_oshape)

    def make_shape_compatible_op(self, model):
        exp_ishape = self.get_normal_input_shape()
        oshape = self.get_normal_output_shape()
        ishape = tuple(model.get_tensor_shape(self.onnx_node.input[0]))
        assert ishape == exp_ishape, "Unexpect input shape for RotaryEmbedding."
        return super().make_const_shape_op(oshape)

    def infer_node_datatype(self, model):
        node = self.onnx_node
        idt = model.get_tensor_datatype(node.input[0])
        if idt != self.get_input_datatype():
            warn_str = "inputDataType changing for {}: {} -> {} ".format(
                node.name,
                str(self.get_input_datatype().name),
                str(idt.name),
            )
            warnings.warn(warn_str)
        self.set_nodeattr("inputDataType", idt.name)
        # set output datatype from property
        odt = self.get_output_datatype()
        model.set_tensor_datatype(node.output[0], odt)

    def verify_node(self):
        pass

    def get_input_datatype(self, ind=0):
        """Returns FINN DataType of input."""
        ret = DataType[self.get_nodeattr("inputDataType")]
        # the hlslib op always pads with zeros, so ensure that the DataType
        # is able to represent zeros
        return ret

    def get_weight_datatype(self, ind=0):
        """Returns FINN DataType of weights."""
        ret = DataType[self.get_nodeattr("weightDataType")]
        return ret

    def get_output_datatype(self, ind=0):
        """Returns FINN DataType of output. (Same as input datatype)"""
        return self.get_input_datatype()

    def get_instream_width(self, ind=0):
        ibits = self.get_input_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return ibits * simd

    def get_outstream_width(self, ind=0):
        obits = self.get_output_datatype().bitwidth()
        simd = self.get_nodeattr("SIMD")
        return obits * simd

    def get_number_output_values(self):
        folded_oshape = self.get_folded_output_shape()
        return np.prod(folded_oshape[:-1])

    def execute_node(self, context, graph):
        ort_session = ort.InferenceSession(self.onnx_path)
        inputs = {
            "act_in": context['act_in'],
        }

        output_names = ["act_out"]

        outputs = ort_session.run(output_names, inputs)

        context[self.onnx_node.output[0]] = outputs[0]
