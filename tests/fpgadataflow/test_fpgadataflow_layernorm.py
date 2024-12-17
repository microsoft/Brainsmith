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
from qonnx.transformation.base import Transformation

test_fpga_part = "xczu3eg-sbva484-1-e"
target_clk_ns = 5

def onnx_path(suffx):
    if not os.path.exists('graphs-tafk-debug'):
        os.makedirs('graphs-tafk-debug')
    return f'graphs-tafk-debug/pytest_layernorm_{suffx}.onnx'

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
    act_in = helper.make_tensor_value_info("global_in", TensorProto.FLOAT, list(idm))
    act_out = helper.make_tensor_value_info("global_out", TensorProto.FLOAT, list(idm))

    # Create model
    graph = helper.make_graph(
        nodes=[], name="LayerNorm_graph", inputs=[act_in], outputs=[act_out]
    )
    model = qonnx_make_model(graph, producer_name="LayerNorm_graph")
    model = ModelWrapper(model)

    # Quant scale & bias
    LayerNorm_scale_out = helper.make_tensor_value_info("LayerNorm_Scale_Quant", TensorProto.FLOAT, scale_bias_shape)
    LayerNorm_bias_out = helper.make_tensor_value_info("LayerNorm_Bias_Quant", TensorProto.FLOAT, scale_bias_shape)
    model.graph.value_info.append(LayerNorm_scale_out)
    model.graph.value_info.append(LayerNorm_bias_out)

    # Quant input node
    Quant_0, Quant_0_out = _create_quant_node('Quant_0', act_in.name, TensorProto.FLOAT, list(idm))
    model.graph.node.append(Quant_0)
    model.graph.value_info.append(Quant_0_out)

    # LayerNormalization node
    LayerNorm_0_out = helper.make_tensor_value_info(model.make_new_valueinfo_name(), TensorProto.FLOAT, list(idm))
    LayerNorm_0 = helper.make_node(
        'LayerNormalization',
        inputs=[Quant_0_out.name, LayerNorm_scale_out.name, LayerNorm_bias_out.name],
        outputs=[act_out.name],
        name='Layernorm_1',
        epsilon=epsilon,
        axis=-1,
    )
    model.graph.node.append(LayerNorm_0)
    model.graph.value_info.append(LayerNorm_0_out)

    # Tensor initializers
    model.set_initializer("LayerNorm_Scale_Quant", (max_scale*np.random.rand(last_dim)).astype(np.float32))
    model.set_initializer("LayerNorm_Bias_Quant", (max_bias*np.random.rand(last_dim)).astype(np.float32))
    model.set_initializer("layernorm0_epsilon_param", np.asarray(epsilon, dtype=np.float32))
    # Quant node initializers
    model.set_initializer("Quant_0_scale", np.asarray(input_quant_params[0], dtype=np.float32))
    model.set_initializer("Quant_0_zeropt", np.asarray(input_quant_params[1], dtype=np.float32))
    model.set_initializer("Quant_0_bitwidth", np.asarray(input_quant_params[2], dtype=np.float32))

    model.save(onnx_path(-1))

    # Force the opset to 17 (TODO: Must be a better way to do this)
    _model = onnx.load(onnx_path(-1))
    op = onnx.OperatorSetIdProto()
    op.version = 17
    _model_opset17 = helper.make_model(_model.graph, opset_imports=[op])    
    onnx.save(_model_opset17, onnx_path(-1))

    model_w = ModelWrapper(onnx_path(-1)) 

    # Datatype annotations
    model_w.set_tensor_datatype(Quant_0_out.name, input_datatype)
    model_w.set_tensor_datatype(LayerNorm_scale_out.name, weight_datatype)
    model_w.set_tensor_datatype(LayerNorm_bias_out.name, bias_datatype)
    model_w.set_tensor_datatype(act_out.name, output_datatype)

    return model_w


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
        "ElementwiseMul_0": {
            "preferred_impl_style": impl_style
        },
        "ElementwiseAdd_0": {
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

    input = np.random.randn(*io_shape).astype(np.float32)
    in_name = model.graph.input[0].name
    out_name = model.graph.output[0].name
    input_t = {in_name: input}

    # Create reference values using the qonnx model    
    y_ref = oxe.execute_onnx(model, input_t)[out_name]
    print()

    try:
        # model = model.transform(QuantizeLayerNormalization(
        #     input_datatype ='INT8',
        #     weight_datatype='FLOAT16',
        #     bias_datatype  ='FLOAT16',
        #     output_datatype='FLOAT16')
        # )
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
        model = model.transform(to_hw.InferElementwiseBinaryOperation())
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
    
    input = np.random.randn(*io_shape).astype(np.float32)
    in_name = model.graph.input[0].name
    input_t = {in_name: input}
    # import pdb; pdb.set_trace()

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


class QuantizeLayerNormalization(Transformation):
    """Add quantization to LayerNormalization nodes in the graph. 
    Temporary implementation pending full quantization support in FINN."""

    def __init__(self, input_datatype=None, weight_datatype=None, bias_datatype=None, output_datatype=None):
        super().__init__()
        self.idt = input_datatype
        self.wdt = weight_datatype
        self.bdt = bias_datatype
        self.odt = output_datatype

    def apply(self, model):
        graph = model.graph
        node_ind = 0
        graph_modified = False
        print('Beginning...')
        for node in graph.node:
            print('Outer')
            node_ind += 1
            print(node.name)
            # Detect LayerNorm
            if node.op_type == "LayerNormalization":
                print('Inner')
                # Get tensors
                act_in = node.input[0]
                act_out = node.output[0]
                scale = node.input[1]
                bias = node.input[2] if len(node.input) > 2 else None
                # Datatype annotations
                model.set_tensor_datatype(act_in, DataType[self.idt])
                model.set_tensor_datatype(scale, DataType[self.wdt])
                model.set_tensor_datatype(act_out, DataType[self.odt])
                if bias:
                    model.set_tensor_datatype(bias, DataType[self.bdt])
                graph_modified = True
                print('     Done')
        return (model, graph_modified)
    