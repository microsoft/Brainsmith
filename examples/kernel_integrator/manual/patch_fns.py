# Copyright (C) 2024, Advanced Micro Devices, Inc.
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

import numpy as np
import warnings
from qonnx.core.datatype import DataType
from qonnx.custom_op.general.multithreshold import multithreshold
from qonnx.util.basic import interleave_matrix_outer_dim_from_partitions

from finn.custom_op.fpgadataflow.hwcustomop import HWCustomOp


class Thresholding(HWCustomOp):
    """Abstraction layer for HW implementation of Thresholding."""

    def verify_node(self):
        info_messages = []
        # verify that "backend" is set to "fpgadataflow"
        backend_value = self.get_nodeattr("backend")
        if backend_value == "fpgadataflow":
            info_messages.append("Attribute backend is set correctly")
        else:
            info_messages.append('Attribute backend should be set to "fpgadataflow"')

        # verify that all necessary attributes exist
        # TODO collect automatically from get_nodeattr_types
        try:
            self.get_nodeattr("code_gen_dir_cppsim")
            self.get_nodeattr("executable_path")
            self.get_nodeattr("NumChannels")
            self.get_nodeattr("PE")
            self.get_nodeattr("inputDataType")
            self.get_nodeattr("outputDataType")
            info_messages.append("All necessary attributes exist")
        except Exception:
            info_messages.append("""The required Threshold_Batch attributes do not exist.""")

        return info_messages

    def minimize_accumulator_width(self, model):
        "Minimize threshold width ('accumulator width' here due to convention)"
        idt = self.get_input_datatype(0)
        if str(idt).startswith("FLOAT") or self.get_nodeattr("weightDataType").startswith("FLOAT"):
            return DataType[self.get_nodeattr("weightDataType")]
        thresholds = model.get_initializer(self.onnx_node.input[1])
        threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)
        min_threshold = thresholds.min()
        max_threshold = thresholds.max()
        min_input = idt.min()
        max_input = idt.max()
        # get range required by threshold values
        tdt_min = min(min_input, min_threshold)
        tdt_max = max(max_input, max_threshold)
        if tdt_min < 0:
            if abs(tdt_min) > tdt_max:
                tdt = DataType.get_smallest_possible(tdt_min)
            else:
                tdt = DataType.get_smallest_possible(-tdt_max - 1)
        else:
            tdt = DataType.get_smallest_possible(tdt_max)
        assert np.vectorize(tdt.allowed)(
            threshold_tensor
        ).all(), "Thresholds can't be expressed with type %s" % str(tdt)
        self.set_nodeattr("weightDataType", tdt.name)
        # Update QONNX DataType of tensor for consistency
        model.set_tensor_datatype(self.onnx_node.input[1], tdt)
        return DataType[self.get_nodeattr("weightDataType")]

    def get_exp_cycles(self):
        # Channels/PE * batch size * fmdim * fmdim
        return np.prod(self.get_folded_output_shape()[:-1])

    def execute_node(self, context, graph):
        node = self.onnx_node
        inp_values = context[node.input[0]]
        th_val = context[node.input[1]]
        out_bias = self.get_nodeattr("ActVal")
        # MT expects inputs to be in the shape (N,C,H,W) or (N, C)
        # if 4D then input values in context are (N,H,W,C) and need to
        # be transposed.
        # if 2D then inputs can be passed directly to MT function
        is_4d = len(inp_values.shape) == 4
        if is_4d:
            inp_values = np.transpose(inp_values, (0, 3, 1, 2))
        y = multithreshold(inp_values, th_val, out_bias=out_bias)
        if is_4d:
            y = y.transpose(0, 2, 3, 1)
        act = DataType[self.get_nodeattr("outputDataType")]
        if act == DataType["BIPOLAR"]:
            # binary to bipolar
            y = 2 * y - 1
        context[node.output[0]] = y

    def calc_tmem(self):
        """Calculates and returns TMEM."""
        num_channels = self.get_nodeattr("NumChannels")
        pe = self.get_nodeattr("PE")
        return num_channels // pe
