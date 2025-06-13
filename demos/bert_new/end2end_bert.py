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
import argparse
import torch
import json
from torch import nn
from transformers import BertConfig, BertModel
from transformers.utils.fx import symbolic_trace
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat
from brevitas.quant import Int8WeightPerTensorFloat
from brevitas.quant import Uint8ActPerTensorFloat
import brevitas.onnx as bo
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from brevitas.graph.quantize import layerwise_quantize
from brevitas.graph.calibrate import calibration_mode
import brainsmith


def generate_bert_model(
        output_dir: str,
        hidden_size: int = 384,
        num_hidden_layers: int = 3,
        num_attention_heads: int = 12,
        intermediate_size: int = 1536,
        bitwidth: int = 8,
        seqlen: int = 128
        ) -> str:
    """
    Generate BERT model directly to output directory.
    
    Returns:
        Path to generated ONNX model
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "bert_model.onnx")

    # Validate BERT configuration - hidden_size must be divisible by num_attention_heads
    if hidden_size % num_attention_heads != 0:
        # Auto-adjust num_attention_heads to largest valid value
        valid_heads = [h for h in [8, 12, 16, 20, 24] if hidden_size % h == 0]
        if valid_heads:
            original_heads = num_attention_heads
            num_attention_heads = max(valid_heads)
            print(f"🔧 Auto-adjusted attention heads: {original_heads} → {num_attention_heads} (for hidden_size {hidden_size})")
        else:
            # Fallback: find any valid divisor
            for h in range(8, min(hidden_size//8, 32) + 1):
                if hidden_size % h == 0:
                    original_heads = num_attention_heads
                    num_attention_heads = h
                    print(f"🔧 Auto-adjusted attention heads: {original_heads} → {num_attention_heads} (for hidden_size {hidden_size})")
                    break
            else:
                raise ValueError(f"Cannot find valid num_attention_heads for hidden_size={hidden_size}")

    # Global consts used by Brevitas build step
    dtype = torch.float32

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
            # 'input_quant': Int8ActPerTensorFloat,
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
            model_path,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=17,
        )
    
    print(f"BERT model generated: {model_path}")
    return model_path


def main(args):
    """Main function showcasing simple brainsmith.forge() power"""
    print("🚀 BERT Accelerator Demo - Powered by brainsmith.forge()")
    print(f"📦 Generating BERT model: {args.num_hidden_layers} layers, {args.hidden_size}D")
    print("✨ Watch one function call create an FPGA accelerator!")
    
    # Generate BERT model directly to output directory
    model_path = generate_bert_model(
        output_dir=args.output_dir,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
    )
    
    # Get minimal blueprint path (optimized for demo)
    blueprint_path = brainsmith.libraries.blueprints.get_blueprint('bert_minimal')
    print(f"📋 Using demo blueprint: {blueprint_path}")
    
    print(f"🎯 Target board: {args.board}")
    
    # Execute forge with simplified API - let blueprint handle optimization
    print("🚀 Generating BERT accelerator with brainsmith.forge()...")
    result = brainsmith.forge(
        model_path=model_path,
        blueprint_path=blueprint_path,
        target_device=args.board,
        output_dir=args.output_dir
    )
    
    # Handle results with structured output
    handle_forge_results(result, args)


def handle_forge_results(result: dict, args) -> None:
    """Simple success-focused result handling with wow factor"""
    print("📦 Processing results...")
    
    if result.get('dataflow_core'):
        print("🎉 SUCCESS! BERT accelerator generated!")
        print(f"📁 Your accelerator is ready in: {args.output_dir}")
        
        # Always show basic metrics for wow factor
        if result.get('metrics'):
            metrics = result['metrics']
            if 'performance' in metrics:
                perf = metrics['performance']
                throughput = perf.get('throughput_ops_sec', 0)
                if throughput > 0:
                    print(f"⚡ Throughput: {throughput:.0f} operations/second")
            
            if 'resources' in metrics:
                res = metrics['resources']
                lut_util = res.get('lut_utilization', 0)
                if lut_util > 0:
                    print(f"🏗️  Resource usage: {lut_util:.0%} LUTs")
        
        # Simple metadata
        metadata = {
            'model': f"BERT-{args.num_hidden_layers}L-{args.hidden_size}D",
            'board': args.board,
            'status': 'success',
            'generated_by': 'brainsmith.forge()'
        }
        
        metadata_path = os.path.join(args.output_dir, 'bert_accelerator_info.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"\n🚀 That's it! One function call created your FPGA accelerator.")
        print(f"🎯 Model: BERT {args.num_hidden_layers} layers, {args.hidden_size} hidden size")
        print(f"💡 Ready to deploy on {args.board}")
        
    else:
        print("❌ Accelerator generation failed")
        print("💡 Check the logs for details")

def create_argument_parser():
    """Create simplified CLI argument parser for forge() showcase"""
    parser = argparse.ArgumentParser(description='BERT Accelerator Demo - Powered by brainsmith.forge()')
    
    # Essential parameters
    parser.add_argument('--output-dir', required=True,
                       help='Output directory for results')
    parser.add_argument('--blueprint', default='bert_accelerator',
                       help='Blueprint name for BERT accelerator')
    
    # BERT model configuration
    bert_group = parser.add_argument_group('BERT Model Configuration')
    bert_group.add_argument('--hidden-size', type=int, default=384,
                           help='BERT hidden size')
    bert_group.add_argument('--num-layers', type=int, default=3,
                           help='Number of BERT layers')
    bert_group.add_argument('--num-heads', type=int, default=12,
                           help='Number of attention heads')
    bert_group.add_argument('--intermediate-size', type=int, default=1536,
                           help='Feed-forward intermediate size')
    bert_group.add_argument('--sequence-length', type=int, default=128,
                           help='Maximum sequence length')
    bert_group.add_argument('--bitwidth', type=int, default=8,
                           help='Quantization bit width (4 or 8)')
    
    # Optimization parameters
    opt_group = parser.add_argument_group('Optimization Configuration')
    opt_group.add_argument('--target-fps', type=int, default=3000,
                          help='Target throughput in FPS')
    opt_group.add_argument('--clock-period', type=float, default=5.0,
                          help='Target clock period in ns')
    opt_group.add_argument('--board', default='V80',
                          help='Target FPGA board')
    
    # Note: DSE options removed for simplified forge() showcase
    # Advanced options preserved in Makefile for expert users
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Convert CLI args to internal format for compatibility
    args.hidden_size = args.hidden_size
    args.num_hidden_layers = args.num_layers
    args.num_attention_heads = args.num_heads
    args.intermediate_size = args.intermediate_size
    args.seqlen = args.sequence_length
    
    main(args)
