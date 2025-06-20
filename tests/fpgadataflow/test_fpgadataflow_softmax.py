############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import pytest
import torch
import os
from onnx import helper
import finn.core.onnx_exec as oxe
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_datatypes import InferDataTypes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import brainsmith.transformation.convert_to_hw_layers as to_bs_hw
from finn.analysis.fpgadataflow.exp_cycles_per_layer import exp_cycles_per_layer
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.fpgadataflow.create_dataflow_partition import (
    CreateDataflowPartition,
)
from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
)
import finn.transformation.streamline.absorb as absorb
from onnx import helper
import torch
import torch.nn as nn
import brevitas.nn as qnn
import numpy as np
test_fpga_part:str = "xcv80-lsva4737-2MHP-e-S"
target_clk_ns = 5
export_onnx_path = "pytest_softmax_dut.onnx"

class SoftMaxSimple(nn.Module):
    def __init__(self):
        super(SoftMaxSimple, self).__init__()
        self.softmax = nn.Softmax(dim=-1) # softmax along the last dimension

    def forward(self, x):
        x = self.softmax(x)
        return x

def create_nonquant_model(io_shape=(1, 12, 128, 128), idt=DataType["INT8"]):
    '''
    Create a quantized softmax model.
    Input and output are quantized to Int8ActPerTensorFloat, this is to make sure
    that the softmax layer is followed by a Quant node.
    '''
    dut = SoftMaxSimple()
    input = torch.rand(io_shape)
    export_qonnx(dut, input, export_onnx_path, opset_version=11)
    qonnx_cleanup(export_onnx_path, out_file=export_onnx_path)
    # set the model input to UINT8
    model = ModelWrapper(export_onnx_path)
    model.set_tensor_datatype(model.graph.input[0].name, idt)
    return model

def make_single_hwsoftmax_modelwrapper(impl_style="hls", simd=1, idt=DataType["UINT8"], ifm_dim=(128, 128)):
    '''
    Create a single quantized softmax node with variable parameters.
    this is before SpecializeLayers() transformation.
    '''
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(ifm_dim))
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, list(ifm_dim))
    new_node = helper.make_node(
        "HWSoftmax",
        ["global_in"],
        ["global_out"],
        domain="brainsmith.custom_op.fpgadataflow",
        backend="fpgadataflow",
        ifm_dim=list(ifm_dim),
        input_data_type=idt.name,
        simd=simd,
        preferred_impl_style=impl_style,
        rtlsim_trace="hwsoftmax_debug_trace.wdb",
    )
    graph = helper.make_graph(
        [new_node],
        "softmax_graph",
        inputs=[inp],
        outputs=[outp]
    )
    model = qonnx_make_model(graph)
    model = ModelWrapper(model)

    model.set_tensor_datatype("global_in", idt)
    model.set_tensor_datatype("global_out", DataType["FLOAT32"])

    return model

@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.parametrize("simd", ["simd1", "simd2", "simd4"])
@pytest.mark.parametrize("idt", ["INT8", "INT9"])
@pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim"])
@pytest.mark.parametrize("ifm_dim", [(1, 128, 384), (1,12,128,128), (1,12,64,128)])
@pytest.mark.fpgadataflow
def test_fpga_dataflow_hwsoftmax(impl_style, simd, idt, exec_mode, ifm_dim):
    os.environ['LIVENESS_THRESHOLD'] = '500000' # Need to bump this up for these RTL sims
    idt = DataType[idt]
    odt = DataType["FLOAT32"]
    simd = int(simd[-1])
    io_shape = ifm_dim
    tollerance = 1e-5 
    model = make_single_hwsoftmax_modelwrapper(impl_style=impl_style, simd=simd, idt=idt, ifm_dim=ifm_dim)

    if(ifm_dim[-1] % simd != 0):
        pytest.skip(f"Skipping this test because the inner dimension is not a multiple of {simd}")

    input = gen_finn_dt_tensor(idt, io_shape)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model
    ref_model = create_nonquant_model(io_shape)
    y_ref = oxe.execute_onnx(ref_model, input_t)[out_name]

    y_out = oxe.execute_onnx(model, input_t)[out_name]
    assert np.allclose(y_ref, y_out, atol=tollerance), "Model output does not match expected output"

    if exec_mode == "cppsim":
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("cppsim"))
        model = model.transform(PrepareCppSim())
        model = model.transform(CompileCppSim())
    elif exec_mode == "rtlsim":
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model = model.transform(SetExecMode("rtlsim"))
        model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
        model = model.transform(HLSSynthIP())
        model = model.transform(PrepareRTLSim())
    else:
        raise RuntimeError(f"Unknown {exec_mode=}")

    # run the model
    y_hw = oxe.execute_onnx(model, input_t)[out_name]

    # Ensure the number of cycles the layer takes to run in rtlsim
    # aligns with the expected number of cycles.
    if exec_mode == "rtlsim":
        op_type = "HWSoftmax_" + impl_style
        node = model.get_nodes_by_op_type(op_type)[0]
        inst = getCustomOp(node)
        cycles_rtlsim = inst.get_nodeattr("cycles_rtlsim")
        exp_cycles_dict = model.analysis(exp_cycles_per_layer)
        exp_cycles = exp_cycles_dict[node.name]
        assert np.isclose(exp_cycles, cycles_rtlsim, atol=10)
        assert exp_cycles != 0

    y_hw_flat = y_hw.flatten()
    y_ref_flat = y_ref.flatten()
    for i in range(len(y_hw_flat)):
        if np.allclose(y_hw_flat[i], y_ref_flat[i], atol=tollerance) == False:
            print(f"Index: {i}, Expected: {y_ref_flat[i]}, Got: {y_hw_flat[i]}")

    assert np.allclose(y_ref, y_hw, atol=tollerance), "Model output does not match expected output"
