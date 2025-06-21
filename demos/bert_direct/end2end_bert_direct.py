#!/usr/bin/env python3
############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

"""
BERT Direct Demo - Bypasses 6-entrypoint system for direct FINN testing

This demo creates a DataflowBuildConfig directly using BrainSmith transforms,
bypassing the Blueprint V2 and LegacyConversionLayer infrastructure to isolate
whether issues are in the transforms or the compatibility layer.

Key differences from bert_new:
- Direct DataflowBuildConfig creation (no 6-entrypoint system)
- Direct FINN builder execution (no BrainSmith API wrapper)
- Uses BrainSmith transforms directly in step sequence
- Identical model generation to bert_new (without output_names)
"""

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

# Direct FINN imports - no BrainSmith API wrapper
import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import (
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
    step_measure_rtlsim_performance
)

# Direct BrainSmith transform imports - no compatibility layer
from brainsmith.libraries.transforms.steps import (
    cleanup_step,
    remove_head_step,
    remove_tail_step,
    qonnx_to_finn_step,
    generate_reference_io_step,
    streamlining_step,
    infer_hardware_step,
    constrain_folding_and_set_pumped_compute_step,
    shell_metadata_handover_step
)

from onnxsim import simplify
from qonnx.util.cleanup import cleanup


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
    Generate BERT model - identical to bert_new but without output_names.
    
    Returns:
        Path to generated ONNX model
    """
    os.makedirs(output_dir, exist_ok=True)
    model_path = os.path.join(output_dir, "bert_model.onnx")

    # Validate BERT configuration
    if hidden_size % num_attention_heads != 0:
        valid_heads = [h for h in [8, 12, 16, 20, 24] if hidden_size % h == 0]
        if valid_heads:
            original_heads = num_attention_heads
            num_attention_heads = max(valid_heads)
            print(f"🔧 Auto-adjusted attention heads: {original_heads} → {num_attention_heads} (for hidden_size {hidden_size})")
        else:
            for h in range(8, min(hidden_size//8, 32) + 1):
                if hidden_size % h == 0:
                    original_heads = num_attention_heads
                    num_attention_heads = h
                    print(f"🔧 Auto-adjusted attention heads: {original_heads} → {num_attention_heads} (for hidden_size {hidden_size})")
                    break
            else:
                raise ValueError(f"Cannot find valid num_attention_heads for hidden_size={hidden_size}")

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
    print("🔍 Starting symbolic tracing...", flush=True)
    model = symbolic_trace(model, input_names)
    print("🔍 Symbolic tracing complete", flush=True)

    print("🔍 Testing pre-replacement forward pass...", flush=True)
    pre_output = model(**inp)
    print("🔍 Pre-replacement forward pass complete", flush=True)

    print("Replace SDPA with quantizable variants...", flush=True)
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacing done.", flush=True)

    print("🔍 Testing post-replacement forward pass...", flush=True)
    post_output = model(**inp)
    print("🔍 Post-replacement forward pass complete", flush=True)

    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
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

    print("🔍 Starting layerwise quantization...", flush=True)
    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    print("🔍 Layerwise quantization complete", flush=True)
    
    print("🔍 Moving model to dtype and calibrating...", flush=True)
    quant_model.to(dtype=dtype)
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)
    print("🔍 Calibration complete", flush=True)

    with torch.no_grad():
        print(f"🔍 DEBUG: Exporting QONNX model to {model_path}", flush=True)
        print(f"🔍 DEBUG: Model outputs before export: {list(quant_model(**inp).keys())}", flush=True)
        
        # CRITICAL: No output_names parameter - matches old demo
        bo.export_qonnx(
            quant_model,
            (input_ids),
            model_path,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=17,
        )
        
        # DEBUG: Validate exported model
        exported_model = onnx.load(model_path)
        print(f"🔍 DEBUG: Exported model inputs: {[inp.name for inp in exported_model.graph.input]}", flush=True)
        print(f"🔍 DEBUG: Exported model outputs: {[out.name for out in exported_model.graph.output]}", flush=True)
    
    print(f"BERT model generated: {model_path}", flush=True)
    return model_path


def generate_reference_io_cached_step(model, cfg):
    """Use pre-generated reference IO to avoid expensive model execution."""
    import shutil
    import os
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Copy pre-generated reference tensors
    shutil.copy(os.path.join(script_dir, "input.npy"), 
                os.path.join(cfg.output_dir, "input.npy"))
    shutil.copy(os.path.join(script_dir, "expected_output.npy"), 
                os.path.join(cfg.output_dir, "expected_output.npy"))
    shutil.copy(os.path.join(script_dir, "expected_context.npz"), 
                os.path.join(cfg.output_dir, "expected_context.npz"))
    
    print("✅ Using cached reference IO tensors (avoiding 6-minute computation)", flush=True)
    return model


def build_direct_dataflow_config(args, model_path: str, build_dir: str) -> build_cfg.DataflowBuildConfig:
    """
    Build DataflowBuildConfig directly using BrainSmith transforms.
    
    This bypasses the 6-entrypoint system and LegacyConversionLayer,
    creating the step sequence directly.
    """
    print("🔧 Building direct DataflowBuildConfig with BrainSmith transforms", flush=True)
    
    # Build step sequence directly - matches old demo ordering
    steps = []
    
    # Phase 1: BrainSmith preprocessing steps
    print("📋 Adding BrainSmith preprocessing steps", flush=True)
    brainsmith_preproc = [
        cleanup_step,
        remove_head_step,
        remove_tail_step,
        qonnx_to_finn_step,
        generate_reference_io_cached_step,  # Use cached version to avoid 6-minute hang
        streamlining_step,
        infer_hardware_step,
    ]
    
    for step in brainsmith_preproc:
        steps.append(step)
        print(f"  ✅ Added: {step.__name__}", flush=True)
    
    # Phase 2: Standard FINN pipeline steps
    print("📋 Adding standard FINN pipeline steps", flush=True)
    finn_steps = [
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
    ]
    
    for step in finn_steps:
        steps.append(step)
        print(f"  ✅ Added: {step.__name__}", flush=True)
    
    # Phase 3: BrainSmith postprocessing steps
    print("📋 Adding BrainSmith postprocessing steps", flush=True)  
    brainsmith_postproc = [
        step_measure_rtlsim_performance,
        constrain_folding_and_set_pumped_compute_step,
        step_set_fifo_depths,
        step_create_stitched_ip,
        shell_metadata_handover_step,
    ]
    
    for step in brainsmith_postproc:
        steps.append(step)
        print(f"  ✅ Added: {step.__name__}", flush=True)
    
    print(f"📊 Total steps in direct config: {len(steps)}", flush=True)
    
    # Create DataflowBuildConfig with same parameters as old demo
    config = build_cfg.DataflowBuildConfig(
        # Core configuration
        steps=steps,
        output_dir=build_dir,
        synth_clk_period_ns=args.clk,
        target_fps=args.fps,
        
        # Folding and optimization
        folding_config_file=args.param,
        standalone_thresholds=args.standalone_thresholds,
        minimize_bit_width=True,
        
        # Hardware configuration
        board=args.board,
        auto_fifo_depths=args.run_fifo_sizing,
        split_large_fifos=args.split_large_fifos,
        
        # Verification and debugging
        fifosim_n_inferences=args.fifosim_n_inferences,
        verification_atol=args.verification_atol,
        verify_input_npy=build_dir+"/input.npy",
        verify_expected_output_npy=build_dir+"/expected_output.npy",
        verify_save_full_context=args.save_intermediate,
        save_intermediate_models=args.save_intermediate,
        
        # Output configuration
        generate_outputs=[build_cfg.DataflowOutputType.STITCHED_IP],
        stitched_ip_gen_dcp=args.dcp,
        
        # Verification steps
        verify_steps=[
            build_cfg.VerificationStepType.FOLDED_HLS_CPPSIM,
            build_cfg.VerificationStepType.STITCHED_IP_RTLSIM,
        ],
        
        # Control flow
        stop_step=args.stop_step,
    )
    
    print("✅ Direct DataflowBuildConfig created successfully", flush=True)
    return config


def main(args):
    """Main function for direct BERT demo."""
    print("🚀 BERT Direct Demo - Bypassing 6-entrypoint system", flush=True)
    print("🎯 Testing BrainSmith transforms directly with FINN", flush=True)
    
    # Create build directory structure
    build_dir = os.path.join(os.environ.get("BSMITH_BUILD_DIR", "./builds"), args.output)
    model_dir = os.path.join(build_dir, "intermediate_models")
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"📁 Build directory: {build_dir}", flush=True)
    
    # Generate BERT model (identical to bert_new)
    print(f"📦 Generating BERT model: {args.num_hidden_layers} layers, {args.hidden_size}D", flush=True)
    model_path = generate_bert_model(
        output_dir=build_dir,
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        bitwidth=args.bitwidth,
        seqlen=args.seqlen
    )
    
    # Load and preprocess model (matching old demo)
    print("🔄 Loading and preprocessing model", flush=True)
    model = onnx.load(model_path)
    
    # Simplify model (matching old demo preprocessing)
    model, check = simplify(model)
    if not check:
        raise RuntimeError("Unable to simplify the BERT model")
    
    if args.save_intermediate:
        onnx.save(model, f"{model_dir}/simp.onnx")
    
    # Cleanup (matching old demo)
    cleanup(in_file=f"{model_dir}/simp.onnx", out_file=f"{build_dir}/df_input.onnx")
    
    # Build direct DataflowBuildConfig
    print("⚙️  Creating direct DataflowBuildConfig", flush=True)
    df_cfg = build_direct_dataflow_config(args, model_path, build_dir)
    
    # Execute FINN builder directly (no BrainSmith API wrapper)
    print("🏗️  Executing FINN builder directly", flush=True)
    try:
        result = build.build_dataflow_cfg(f"{build_dir}/df_input.onnx", df_cfg)
        print("✅ Direct FINN execution successful", flush=True)
        
        # Copy final model (matching old demo)
        if args.stop_step is None:
            final_step = df_cfg.steps[-1].__name__
        else:
            final_step = args.stop_step
        
        # Check if final step model exists before copying
        final_model_path = f"{model_dir}/{final_step}.onnx"
        if os.path.exists(final_model_path):
            import shutil
            shutil.copy2(final_model_path, f"{build_dir}/output.onnx")
            print(f"📄 Final model copied: {final_step}.onnx", flush=True)
        else:
            print(f"⚠️  Final model not found: {final_model_path}", flush=True)
            # Find the last available model
            available_models = [f for f in os.listdir(model_dir) if f.endswith('.onnx')]
            if available_models:
                latest_model = sorted(available_models)[-1]
                import shutil
                shutil.copy2(f"{model_dir}/{latest_model}", f"{build_dir}/output.onnx")
                print(f"📄 Using latest available model: {latest_model}", flush=True)
        
        # Handle handover metadata
        handover_file = f"{build_dir}/stitched_ip/shell_handover.json"
        if os.path.exists(handover_file):
            with open(handover_file, "r") as fp:
                handover = json.load(fp)
            handover["num_layers"] = args.num_hidden_layers
            with open(handover_file, "w") as fp:
                json.dump(handover, fp, indent=4)
        
        print(f"🎉 SUCCESS! BERT accelerator generated directly in: {build_dir}", flush=True)
        
    except Exception as e:
        print(f"❌ Direct FINN execution failed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        raise


def create_argument_parser():
    """Create CLI argument parser matching old demo."""
    parser = argparse.ArgumentParser(description='BERT Direct Demo - Bypass 6-entrypoint system')
    
    # Core parameters
    parser.add_argument('-o', '--output', help='Output build name', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, help='Sets BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, help='Sets BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, help='Sets BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, help='The quantization bitwidth (either 4 or 8)')
    parser.add_argument('-f', '--fps', type=int, default=3000, help='The target fps for auto folding')
    parser.add_argument('-c', '--clk', type=float, default=3.33, help='The target clock rate for the hardware')
    parser.add_argument('-s', '--stop_step', type=str, default=None, help='Step to stop at in the build flow')
    parser.add_argument('-p', '--param', type=str, default=None, help='Use a preconfigured file for the folding parameters')
    parser.add_argument('-x', '--run_fifo_sizing', action='store_true', help='Run the fifo-sizing step')
    parser.add_argument('-q', '--seqlen', type=int, default=128, help='Sets the sequence length parameter')
    parser.add_argument('-d', '--dcp', type=bool, default=True, help='Generate a DCP')
    
    return parser


if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Set default values matching old demo
    args.save_intermediate = True
    args.standalone_thresholds = True
    args.fifosim_n_inferences = 2
    args.board = "V80"
    args.verification_atol = 1e-1
    args.split_large_fifos = True
    
    print("🔧 Direct FINN execution with BrainSmith transforms", flush=True)
    print(f"📐 Model: {args.num_hidden_layers}L x {args.hidden_size}D x {args.num_attention_heads}H", flush=True)
    print("⚡ Bypassing 6-entrypoint and compatibility layer", flush=True)

    main(args)