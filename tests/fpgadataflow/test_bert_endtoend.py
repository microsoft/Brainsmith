import onnx  
import os
import pytest
import shutil
import argparse
import math
import torch
from torch import nn
from transformers import BertConfig, BertModel
from transformers import AutoModel
from transformers.utils.fx import symbolic_trace

import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode

from onnxsim import simplify  
from qonnx.util.cleanup import cleanup
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames

from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
import finn.transformation.streamline as absorb
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.hlssynth_ip import HLSSynthIP
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP

from finnbrainsmith.util.bert import custom_step_remove_head, custom_step_remove_tail, custom_step_cleanup
import finnbrainsmith.transformation.convert_to_hw_layers as to_bs_hw

# Global consts used by Brevitas build step
bit_width=8
dtype=torch.float32
smax_val=2**(bit_width-1)-1
umax_val=2**(bit_width)-1

class IntActPerTensorFloatConstScale(Int8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    max_val=smax_val
    min_val=-smax_val

class IntWeightPerTensorFloatConstScale(Int8WeightPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    scaling_const=smax_val

class UintActPerTensorFloatConstScale(Uint8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    max_val=umax_val

class UintActPerTensorFloatConstScale1(Uint8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    max_val=1.0

class IntActTanh(Int8ActPerTensorFloat):
    scaling_impl_type="const"
    restrict_scaling_type="fp"
    narrow_range=True
    max_val=1.0
    min_val=-1.0

def gen_initial_bert_model(
        outfile:str="bert.onnx",
        hidden_size:int=384,
        num_attention_heads:int=12,
        intermediate_size:int=1536
        )->None:
    """ Generates the initial BERT model from Brevitas. (Write more here) """
    config = BertConfig(
      hidden_size=384,
      num_hidden_layers=1,
      num_attention_heads=12,
      intermediate_size=1536,
      attn_implementation="sdpa",
      hidden_act="relu",
    )
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    seq_len = 128
    batch_size = 1
    
    with torch.no_grad():
        for name, module in model.named_modules():
            if type(module) == nn.Linear:
                module.weight *= (smax_val / (module.weight.abs().max() * math.sqrt(float(module.out_features))))
    
    input_ids = torch.randint(vocab_size, (batch_size,seq_len), dtype=torch.int64)
    attention_mask = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.float32)
    token_type_ids = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.int64)
    inp = {
        'input_ids': input_ids,
    #    'attention_mask': attention_mask,
    #    'token_type_ids': token_type_ids,
    }
    
    input_names = inp.keys()
    model = symbolic_trace(model, input_names)
    
    pre_output = model(**inp)
    
    print("Replace SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacing done.")
    
    post_output = model(**inp)
    
    # Old version (some old transformers version)
    #print(pre_output.pooler_output.shape)
    #print(pre_output.pooler_output)
    #print(f"{pre_output.pooler_output.shape} - {post_output.pooler_output.shape}")
    #print(pre_output.pooler_output - post_output.pooler_output)
    
    # Sanity check that the layer replacement worked
    print(pre_output["pooler_output"].shape)
    print(pre_output["pooler_output"])
    print(f"{pre_output['pooler_output'].shape} - {post_output['pooler_output'].shape}")
    print(pre_output['pooler_output'] - post_output['pooler_output'])
    
    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            #'input_quant': IntActPerTensorFloatConstScale,
            'input_quant': lambda module: UintActPerTensorFloatConstScale if module.in_features == config.intermediate_size and unsigned_hidden_act else IntActPerTensorFloatConstScale,
            'weight_quant': IntWeightPerTensorFloatConstScale,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': IntActPerTensorFloatConstScale,
            'attn_output_weights_quant': UintActPerTensorFloatConstScale1,
            'q_scaled_quant': IntActPerTensorFloatConstScale,
            'k_transposed_quant': IntActPerTensorFloatConstScale,
            'v_quant': IntActPerTensorFloatConstScale,
            'attn_output_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': IntActTanh,
            'return_quant_tensor': False})
    
    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    quant_model.to(dtype=dtype)
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)
    
    with torch.no_grad():
        bo.export_qonnx(
            quant_model,
            (input_ids),
            outfile,
            do_constant_folding=True,
            input_names=['input_ids'],
            #dynamic_axes={
            #    'input_ids': {
            #        0: 'batch_size',
            #        1: 'sequence_length',
            #    },  
            #},  
            opset_version=17,
        )

def custom_streamlining_step(model,cfg):
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(absorb.AbsorbMulIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    return model

def attempt_convert_step(model, cfg):
    model = model.transform(ConvertQONNXtoFINN())
    return model

def attempt_specialise_layers(model, cfg):
    model = model.transform(SpecializeLayers(fpgapart=cfg.fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    return model

def custom_step_infer_hardware(model, cfg):
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferStreamingEltwise())
    model = model.transform(to_hw.InferLookupLayer())
    model = model.transform(to_hw.InferThresholdingLayer())
    model = model.transform(to_hw.InferQuantizedMatrixVectorActivation())
    model = model.transform(to_bs_hw.InferShuffle())
    model = model.transform(to_bs_hw.InferQuantSoftmax())
    return model

def custom_step_create_ip(model, cfg):
    model = model.transform(PrepareIP(cfg.fpga_part, cfg.synth_clk_period_ns))
    model = model.transform(HLSSynthIP())
    model = model.transform(CreateStitchedIP(cfg.fpga_part, cfg.synth_clk_period_ns))
    return model
    
@pytest.mark.parametrize("hidden_size", [384])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("intermediate_size", [1536])
@pytest.mark.parametrize("gen_ip", [True, False])
def test_endtoend(
        hidden_size:int,
        num_attention_heads:int,
        intermediate_size:int,
        gen_ip:bool
    ):
    tmp = "./intermediate_models"
    os.makedirs(tmp, exist_ok=True)

    # Initial model generation
    gen_initial_bert_model(
        outfile=f"{tmp}/initial.onnx",
        hidden_size=hidden_size,
        num_attention_heads=num_attention_heads,
        intermediate_size=intermediate_size
    )

    # Initial model cleanup
    model = onnx.load(f"{tmp}/initial.onnx")  
    model_simp, check = simplify(model)  
    if check:  
        onnx.save(model_simp, f"{tmp}/simp.onnx")  
    else:  
        raise RuntimeError(f"Unable to simplify the Brevitas bert model")
    cleanup(in_file=f"{tmp}/simp.onnx", out_file=f"{tmp}/qonnx_cleanup.onnx")
    
    steps = [ 
              custom_step_cleanup, 
              custom_step_remove_head, 
              custom_step_remove_tail,
              attempt_convert_step,
              custom_streamlining_step,
              custom_step_infer_hardware,
              attempt_specialise_layers
            ]

    if gen_ip:
        steps.append(custom_step_create_ip)

    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        output_dir=tmp,
        synth_clk_period_ns=5,
        fpga_part="xcv80-lsva4737-2MHP-e-S",
        generate_outputs=[],
    )
    
    _ = build.build_dataflow_cfg(f"{tmp}/qonnx_cleanup.onnx", cfg)
    shutil.copy2(f"{tmp}/intermediate_models/{steps[-1].__name__}.onnx", "_end2end_test_output.onnx")

