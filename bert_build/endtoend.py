############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT 
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import warnings  
warnings.simplefilter("ignore")  

import onnx  
import os
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
from qonnx.transformation.general import GiveReadableTensorNames, GiveUniqueNodeNames, ConvertDivToMul
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg

from finnbrainsmith.util.bert import (
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_generate_reference_io,
        custom_step_cleanup,
        custom_step_infer_hardware,
        custom_streamlining_step,
        custom_step_qonnx2finn,
)

from finn.builder.build_dataflow_steps import (
    step_qonnx_to_finn,
    step_tidy_up,
    step_streamline,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance,
    step_out_of_context_synthesis,
    step_synthesize_bitfile,
    step_make_pynq_driver,
    step_deployment_package,
)

def gen_initial_bert_model(
        outfile:str="bert.onnx",
        hidden_size:int=384,
        num_hidden_layers:int=3,
        num_attention_heads:int=12,
        intermediate_size:int=1536,
        bitwidth:int=8,
        seqlen:int=128
        )->None:
    """ Generates the initial BERT model from Brevitas. (Write more here) """

    # Global consts used by Brevitas build step
    dtype=torch.float32

    config = BertConfig(
      hidden_size=hidden_size,
      num_hidden_layers=num_hidden_layers,
      num_attention_heads=num_attention_heads,
      intermediate_size=intermediate_size,
      attn_implementation="sdpa",
      hidden_act="relu",
    )
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    vocab_size = model.config.vocab_size
    seq_len = seqlen 
    batch_size = 1
    
    input_ids = torch.randint(vocab_size, (batch_size,seq_len), dtype=torch.int64)
    attention_mask = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.float32)
    token_type_ids = torch.randint(high=2, size=(batch_size,seq_len), dtype=torch.int64)
    inp = {
        'input_ids': input_ids,
    }
    
    input_names = inp.keys()
    model = symbolic_trace(model, input_names)
    
    pre_output = model(**inp)
    
    print("Replace SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacing done.")
    
    post_output = model(**inp)
    
    # Sanity check that the layer replacement worked
    #print(pre_output["pooler_output"].shape)
    #print(pre_output["pooler_output"])
    #print(f"{pre_output['pooler_output'].shape} - {post_output['pooler_output'].shape}")
    #print(pre_output['pooler_output'] - post_output['pooler_output'])
    
    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            #'input_quant': Int8ActPerTensorFloat,
            'input_quant': lambda module: Uint8ActPerTensorFloat if module.in_features == config.intermediate_size and unsigned_hidden_act else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'weight_bit_width': bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': bitwidth,
            'attn_output_quant': None,
            'return_quant_tensor': False})
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': bitwidth,
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
            opset_version=17,
        )



def main(args):
    tmp = "./intermediate_models"
    os.makedirs(tmp, exist_ok=True)

    # Initial model generation
    gen_initial_bert_model(
        outfile=f"{tmp}/initial.onnx",
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
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
        # Cleanup and custom graph surgery
        custom_step_cleanup,
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_qonnx2finn,

        custom_step_generate_reference_io,
        custom_streamlining_step,
        custom_step_infer_hardware,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_measure_rtlsim_performance,
        step_set_fifo_depths,
        step_create_stitched_ip,
    ]

    cfg = build_cfg.DataflowBuildConfig(
        standalone_thresholds=True,
        steps=steps,
        target_fps=args.fps,
        output_dir=tmp,
        synth_clk_period_ns=args.clk,
        folding_config_file=args.param,
        stop_step=args.stop_step,
        auto_fifo_depths=args.fifodepth,
        fifosim_n_inferences=2,
        verification_atol=1e-1,
        split_large_fifos=True,
        stitched_ip_gen_dcp=args.dcp,
        board="V80",
        generate_outputs=[
            build_cfg.DataflowOutputType.STITCHED_IP,
            ],
        verify_input_npy="input.npy",
        verify_expected_output_npy="expected_output.npy",
        verify_save_full_context=True,
        verify_steps=[
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
    )
    
    _ = build.build_dataflow_cfg(f"{tmp}/qonnx_cleanup.onnx", cfg)
    if args.stop_step is None:
        shutil.copy2(f"{tmp}/intermediate_models/{steps[-1].__name__}.onnx", args.output)
    else:
        shutil.copy2(f"{tmp}/intermediate_models/{args.stop_step}.onnx", args.output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='TinyBERT FINN demo script')
    parser.add_argument('-o', '--output', help='Output ONNX file path', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, help='Sets BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, help='Sets BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, help='Sets BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, help='The quantisation bitwidth (either 4 or 8)')
    parser.add_argument('-f', '--fps', type=int, default=3000, help='The target fps for auto folding')
    parser.add_argument('-c', '--clk', type=float, default=3.33, help='The target clock rate for the hardware')
    parser.add_argument('-s', '--stop_step', type=str, default=None, help='Step to stop at in the build flow')
    parser.add_argument('-p', '--param', type=str, default=None, help='Use a preconfigured file for the folding parameters')
    parser.add_argument('-x', '--fifodepth', type=bool, default=True, help='Skip the FIFO depth stage')
    parser.add_argument('-q', '--seqlen', type=int, default=128, help='Sets the sequence length parameter')
    parser.add_argument('-d', '--dcp', type=bool, default=True, help='Generate a DCP')

    args = parser.parse_args()
    main(args)
