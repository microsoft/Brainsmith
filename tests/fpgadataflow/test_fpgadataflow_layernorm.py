from typing import Tuple
import pytest
import torch
import onnx
import torch.nn as nn
import brevitas.nn as qnn
import finn.core.onnx_exec as oxe
from brevitas.export import export_qonnx
from qonnx.util.cleanup import cleanup as qonnx_cleanup
from onnx import TensorProto, helper
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.infer_shapes import InferShapes 
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from qonnx.util.basic import gen_finn_dt_tensor, qonnx_make_model
from qonnx.transformation.infer_datatypes import InferDataTypes
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw
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
from finnbrainsmith.transformation.expand_norms import ExpandNorms

# Debugging dependencies, to remove
import os

from qonnx.transformation.fold_constants import FoldConstants

from qonnx.transformation.general import (
    ApplyConfig,
    GiveUniqueNodeNames,
)

from finn.transformation.streamline import Streamline
import finn.transformation.streamline.absorb as absorb
import numpy as np

# from finn.builder.build_dataflow_config import DataflowBuildConfig
from finn.transformation.qonnx.quant_act_to_multithreshold import (
    default_filter_function_generator as dff_gen,
)
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5

def onnx_path(suffx):
    if not os.path.exists('tafk-debug'):
        os.makedirs('tafk-debug')
    return f'tafk-debug/pytest_layernorm_{suffx}.onnx'

def _create_quant_node(node_name, inp_name, output_or_dtype, shape):
    if isinstance(output_or_dtype, str):
        Quant_out = None
        output_name = output_or_dtype
    else:
        Quant_out = helper.make_tensor_value_info(f"{node_name}_out", output_or_dtype, shape)
        # Quant_out = helper.make_tensor_value_info(f"{node_name}_out", TensorProto.FLOAT, shape)
        output_name = Quant_out.name 
    Quant = helper.make_node(
            'Quant',
            domain='qonnx.custom_op.general',
            inputs=[inp_name, f'{node_name}_scale', f'{node_name}_zeropt', f'{node_name}_bitwidth'],
            outputs=[output_name],
            narrow=0,
            signed=1,
            rounding_mode="ROUND",
            name=node_name
    )
    return Quant, Quant_out

def build_layernorm_graph(
        input_datatype:str,
        weight_datatype:str,
        bias_datatype:str,
        output_datatype:str,
        epsilon:float,
        idm:tuple, # Input dimension
) -> ModelWrapper:

    # Datatypes restricted to "FLOAT16" or "FLOAT32" in current implementation
    bw = []
    for dt in [input_datatype, weight_datatype, bias_datatype, output_datatype]:
        match dt:
            case "INT8":
                bw += [8]
            case "FLOAT16":
                bw += [16]
            case "FLOAT32":
                bw += [32]
            case _:
                raise ValueError(f"LayerNorm only supports FP16/FP32 w/b. Invalid input: {dt}")
    
    #(scale, zero_point, bitwidth)
    input_quant_params  = [1.0, 0.0, bw[0]]
    scale_quant_params  = [1.0/(1<<bw[1]), 0.0, bw[1]]
    bias_quant_params   = [1.0/(1<<bw[2]), 0.0, bw[2]]
    output_quant_params = [1.0/(1<<bw[3]), 0.0, bw[3]]

    max_scale = 2**(bw[1]/2)
    max_bias = 2**(bw[2]/2)

    last_dim = idm[-1]
    scale_bias_shape = [last_dim]

    # Create I/Os 
    inp = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(idm))
    outp = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, list(idm))

    # Create model
    graph = helper.make_graph(
        nodes=[], name="LayerNorm_graph", inputs=[inp], outputs=[outp]
    )
    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    # Quant input node
    Quant_0, Quant_0_out = _create_quant_node('Quant_0', inp.name, TensorProto.FLOAT, list(idm))
    model.graph.node.append(Quant_0)
    model.graph.value_info.append(Quant_0_out)

    # Quant scale node
    Quant_LayerNorm_scale, Quant_LayerNorm_scale_out = _create_quant_node('LayerNorm_Scale_Quant', 'layernorm0_scale_param', TensorProto.FLOAT, scale_bias_shape)
    model.graph.node.append(Quant_LayerNorm_scale)
    model.graph.value_info.append(Quant_LayerNorm_scale_out)

    # Quant bias node
    Quant_LayerNorm_bias, Quant_LayerNorm_bias_out = _create_quant_node('LayerNorm_Bias_Quant', 'layernorm0_b_param', TensorProto.FLOAT, scale_bias_shape)
    model.graph.node.append(Quant_LayerNorm_bias)
    model.graph.value_info.append(Quant_LayerNorm_bias_out)

    # LayerNormalization node
    LayerNorm_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    LayerNorm_0 = helper.make_node(
        'LayerNormalization',
        inputs=[Quant_0_out.name, Quant_LayerNorm_scale_out.name, Quant_LayerNorm_bias_out.name],
        outputs=[LayerNorm_0_out.name],
        name='Layernorm_1',
        epsilon=epsilon,
        axis=-1,
    )
    model.graph.node.append(LayerNorm_0)
    model.graph.value_info.append(LayerNorm_0_out)

    # Quant output node
    Quant_1, _ = _create_quant_node('Quant_1', LayerNorm_0_out.name, outp.name, list(idm))
    model.graph.node.append(Quant_1)

    # Quant node initializers
    model.set_initializer("Quant_0_scale", np.asarray(input_quant_params[0], dtype=np.float32))
    model.set_initializer("Quant_0_zeropt", np.asarray(input_quant_params[1], dtype=np.float32))
    model.set_initializer("Quant_0_bitwidth", np.asarray(input_quant_params[2], dtype=np.float32))

    model.set_initializer("Quant_1_scale", np.asarray(output_quant_params[0], dtype=np.float32))
    model.set_initializer("Quant_1_zeropt", np.asarray(output_quant_params[1], dtype=np.float32))
    model.set_initializer("Quant_1_bitwidth", np.asarray(output_quant_params[2], dtype=np.float32))

    model.set_initializer("LayerNorm_Bias_Quant_scale", np.asarray(bias_quant_params[0], dtype=np.float32))
    model.set_initializer("LayerNorm_Bias_Quant_zeropt", np.asarray(bias_quant_params[1], dtype=np.float32))
    model.set_initializer("LayerNorm_Bias_Quant_bitwidth", np.asarray(bias_quant_params[2], dtype=np.float32))

    model.set_initializer("LayerNorm_Scale_Quant_scale", np.asarray(scale_quant_params[0], dtype=np.float32))
    model.set_initializer("LayerNorm_Scale_Quant_zeropt", np.asarray(scale_quant_params[1], dtype=np.float32))
    model.set_initializer("LayerNorm_Scale_Quant_bitwidth", np.asarray(scale_quant_params[2], dtype=np.float32))

    model.set_initializer("layernorm0_scale_param", (max_scale*np.random.rand(last_dim)).astype(np.float32))
    model.set_initializer("layernorm0_b_param", (max_bias*np.random.rand(last_dim)).astype(np.float32))
    model.set_initializer("layernorm0_epsilon_param", np.asarray(epsilon, dtype=np.float32))

    model.save(onnx_path(-1))

    # Force the opset to 17 (TODO: Must be a better way to do this)
    _model = onnx.load(onnx_path(-1))
    op = onnx.OperatorSetIdProto()
    op.version = 17
    _model_opset17 = helper.make_model(_model.graph, opset_imports=[op])    
    onnx.save(_model_opset17, onnx_path(-1))

    return ModelWrapper(onnx_path(-1)) 


# @pytest.mark.parametrize("exec_mode", ["cppsim", "rtlsim", "stitched_ip"])
# @pytest.mark.parametrize("simd", ["simd1", "simd2", "simd3", "simd4"])
@pytest.mark.parametrize("impl_style", ["hls"])
@pytest.mark.parametrize("exec_mode", ["cppsim"])
@pytest.mark.parametrize("simd", ["simd1"])
@pytest.mark.parametrize("idt", ["INT8"])
@pytest.mark.parametrize("wdt", ["FLOAT16"])
@pytest.mark.parametrize("bdt", ["FLOAT16"])
@pytest.mark.parametrize("odt", ["FLOAT16"])
@pytest.mark.parametrize("ifm_dim", [(1, 128, 384), (1, 12, 12, 128)])
@pytest.mark.fpgadataflow
def test_fpga_dataflow_layernorm(impl_style, exec_mode, simd, idt, wdt, bdt, odt, ifm_dim):
    '''
    This test checks that the ONNX LayerNormalization can lowered to FINN LayerNorm
    '''
    if (exec_mode == "stitched_ip" or exec_mode == "rtlsim") and simd != "simd1":
        pytest.skip("Skipping this test to avoid long test times")
    
    idt = DataType[idt]
    odt = DataType[odt]
    wdt = DataType[wdt]
    bdt = DataType[bdt]
    
    simd = int(simd[-1])
    folding_config = {
        "Defaults": {},
        "LayerNorm_0": {
            "simd": simd,
            "preferred_impl_style": impl_style
        },
    }
    io_shape = ifm_dim
    epsilon = 1e-05
    tolerance = 2
    
    model = build_layernorm_graph(idt, wdt, bdt, odt, epsilon, ifm_dim)
    # model = build_func_layernorm_graph(idt, odt, epsilon, ifm_dim)

    model = model.transform(InferShapes())
    model.save(onnx_path(0))

    if(ifm_dim[-1] % simd != 0):
        pytest.skip(f"Skipping this test because the channel dimension is not a multiple of {simd}")

    input = np.random.randn(*io_shape)    
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model    
    y_ref = oxe.execute_onnx(model, input_t)[out_name]

    try:
        # Lower graph to HWCustomOps
        model = model.transform(ExpandNorms())
        model.save(onnx_path(1)) # Debug
        model = model.transform(ExtractQuantScaleZeroPt())
        model.save(onnx_path(11)) # Debug
        model = model.transform(FoldConstants())
        model.save(onnx_path(12)) # Debug
        model = model.transform(ConvertQONNXtoFINN(filter_function=dff_gen(max_multithreshold_bit_width=32)))
        model.save(onnx_path(2)) # Debug
        # Fold Constants
        model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
        model = model.transform(absorb.AbsorbAddIntoMultiThreshold())
        model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
        model.save(onnx_path(3)) # Debug
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())
        model.save(onnx_path(4)) # Debug
        model = model.transform(to_bs_hw.InferLayerNorm())
        model = model.transform(GiveUniqueNodeNames())
        model.save(onnx_path(5)) # Debug
        # model = model.transform(RoundAndClipThresholds())
        model = model.transform(to_hw.InferThresholdingLayer())
        # model = model.transform(to_hw.InferElementwiseBinaryOperation())
        model = model.transform(GiveUniqueNodeNames())
        model.save(onnx_path(6)) # Debug

        ## Isolate fpga dataflow layers
        #parent_model = model.transform(CreateDataflowPartition())
        #parent_model.save(onnx_path(5)) # Debug
        #sdp_node = parent_model.get_nodes_by_op_type("StreamingDataflowPartition")[0]
        #sdp_node_path = getCustomOp(sdp_node).get_nodeattr("model")
        #model = ModelWrapper(sdp_node_path)
        #model.save(onnx_path(6)) # Debug

        model = model.transform(ApplyConfig(folding_config))
        model = model.transform(SpecializeLayers(test_fpga_part))
        model = model.transform(GiveUniqueNodeNames())
        model.save(onnx_path(7)) # Debug
        
        # model = model.transform(SetExecMode("python"))
        # y_python = oxe.execute_onnx(model, input_t)[out_name]
        # assert np.allclose(y_ref, y_python, atol=tolerance), "HWCustomOp output does not match expected output"

        # Execute selected sim
        if exec_mode == "cppsim":
            model = model.transform(SetExecMode("cppsim"))
            model = model.transform(PrepareCppSim())
            model = model.transform(CompileCppSim())
        elif exec_mode == "rtlsim":
            model = model.transform(SetExecMode("rtlsim"))
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(PrepareRTLSim())
        elif exec_mode == "stitched_ip":
            model = model.transform(PrepareIP(test_fpga_part, target_clk_ns))
            model = model.transform(HLSSynthIP())
            model = model.transform(CreateStitchedIP(test_fpga_part, target_clk_ns))
    except Exception as e:
        pytest.fail(f"Failed to transform the model: {str(e)}")
    
    input = np.random.randn(*io_shape)
    in_name = model.graph.input[0].name
    input_t = {in_name: input}
    y_hw = oxe.execute_onnx(model, input_t)[model.graph.output[0].name]

    j = 0
    y_ref = y_ref.flatten()
    y_hw = y_hw.flatten()
    for i in range(len(y_ref)):
        if not np.allclose(y_ref[i], y_hw[i], atol=tolerance):
            print(f'at {i}: {y_ref[i]} != {y_hw[i]}')
            j+=1
        if j > 20:
            assert False, "Too much!"

    assert np.allclose(y_ref, y_hw, atol=tolerance), "HW sim output does not match expected output"

    print(f"Test matches")
