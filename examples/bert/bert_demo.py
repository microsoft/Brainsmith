############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

import argparse
import json
import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import onnx
import torch

from brainsmith.config import get_build_dir
from brevitas.graph.calibrate import calibration_mode
from brevitas.graph.quantize import layerwise_quantize
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerTensorFloat, Uint8ActPerTensorFloat
from brevitas_examples.llm.llm_quant.prepare_for_quantize import replace_sdpa_with_quantizable_layers
from onnxsim import simplify
from qonnx.core.datatype import DataType
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.util.cleanup import cleanup
from torch import nn
from transformers import BertConfig, BertModel
from transformers.utils.fx import symbolic_trace
import brevitas.nn as qnn
import brevitas.onnx as bo

import custom_steps  # Import custom steps to trigger registration

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import forge

warnings.simplefilter("ignore")


def generate_bert_model(args):
    """Generate quantized BERT model from HuggingFace with Brevitas quantization.
    
    This matches the functionality from old end2end_bert.py::gen_initial_bert_model()
    """
    print(f"Generating BERT model with {args.num_hidden_layers} layers...")
    
    # Global consts used by Brevitas build step
    dtype = torch.float32
    
    # Create BERT configuration
    config = BertConfig(
        hidden_size=args.hidden_size,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        intermediate_size=args.intermediate_size,
        attn_implementation="sdpa",
        hidden_act="relu",
    )
    
    # Initialize model
    model = BertModel(config=config)
    model.to(dtype=dtype)
    model.eval()
    
    # Prepare inputs
    vocab_size = model.config.vocab_size
    seq_len = args.seqlen
    batch_size = 1
    
    input_ids = torch.randint(vocab_size, (batch_size, seq_len), dtype=torch.int64)
    inp = {'input_ids': input_ids}
    
    # Symbolic tracing
    input_names = inp.keys()
    model = symbolic_trace(model, input_names)
    
    # Replace SDPA with quantizable layers
    print("Replacing SDPA with quantizable variants...")
    model = replace_sdpa_with_quantizable_layers(model)
    print("Replacement done.")
    
    # Configure quantization
    unsigned_hidden_act = config.hidden_act == 'relu'
    layerwise_compute_layer_map = {}
    
    # Linear layer quantization
    layerwise_compute_layer_map[nn.Linear] = (
        qnn.QuantLinear,
        {
            'input_quant': lambda module: Uint8ActPerTensorFloat 
                if module.in_features == config.intermediate_size and unsigned_hidden_act 
                else Int8ActPerTensorFloat,
            'weight_quant': Int8WeightPerTensorFloat,
            'weight_bit_width': args.bitwidth,
            'output_quant': None,
            'bias_quant': None,
            'return_quant_tensor': False
        }
    )
    
    # Attention quantization
    layerwise_compute_layer_map[qnn.ScaledDotProductAttention] = (
        qnn.QuantScaledDotProductAttention,
        {
            'softmax_input_quant': Int8ActPerTensorFloat,
            'softmax_input_bit_width': args.bitwidth,
            'attn_output_weights_quant': Uint8ActPerTensorFloat,
            'attn_output_weights_bit_width': args.bitwidth,
            'q_scaled_quant': Int8ActPerTensorFloat,
            'q_scaled_bit_width': args.bitwidth,
            'k_transposed_quant': Int8ActPerTensorFloat,
            'k_transposed_bit_width': args.bitwidth,
            'v_quant': Int8ActPerTensorFloat,
            'v_bit_width': args.bitwidth,
            'attn_output_quant': None,
            'return_quant_tensor': False
        }
    )
    
    # Tanh quantization
    layerwise_compute_layer_map[nn.Tanh] = (
        qnn.QuantTanh,
        {
            'input_quant': None,
            'act_quant': Int8ActPerTensorFloat,
            'act_bit_width': args.bitwidth,
            'return_quant_tensor': False
        }
    )
    
    # Apply quantization
    quant_model = layerwise_quantize(model, compute_layer_map=layerwise_compute_layer_map)
    quant_model.to(dtype=dtype)
    
    # Calibration
    with torch.no_grad(), calibration_mode(quant_model):
        quant_model(**inp)
    
    # Export to ONNX
    with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as tmp:
        tmp_path = tmp.name
        
    with torch.no_grad():
        bo.export_qonnx(
            quant_model,
            (input_ids),
            tmp_path,
            do_constant_folding=True,
            input_names=['input_ids'],
            opset_version=17,
        )
    
    # Load and return model
    model = onnx.load(tmp_path)
    os.unlink(tmp_path)
    
    # Save initial Brevitas model for debugging
    debug_path = os.path.join(args.output_dir, "debug_models")
    os.makedirs(debug_path, exist_ok=True)
    onnx.save(model, os.path.join(debug_path, "00_initial_brevitas.onnx"))
    print(f"Saved initial Brevitas model to debug_models/00_initial_brevitas.onnx")
    print(f"  - Model inputs: {[i.name for i in model.graph.input]}")
    print(f"  - Model outputs: {[o.name for o in model.graph.output]}")
    print(f"  - Number of nodes: {len(model.graph.node)}")
    
    return model


def generate_reference_io(model, output_dir):
    """Generate reference input/output for verification.
    
    This matches custom_step_generate_reference_io from old bert.py
    """
    import finn.core.onnx_exec as oxe
    from qonnx.core.modelwrapper import ModelWrapper
    from qonnx.transformation.infer_shapes import InferShapes
    
    # Wrap model
    model_wrapper = ModelWrapper(model)
    
    # Infer shapes first
    model_wrapper = model_wrapper.transform(InferShapes())
    
    # Generate input
    input_m = model_wrapper.graph.input[0]
    in_shape = [dim.dim_value for dim in input_m.type.tensor_type.shape.dim]
    in_tensor = gen_finn_dt_tensor(DataType["FLOAT32"], in_shape)
    
    # Save input
    np.save(os.path.join(output_dir, "input.npy"), in_tensor)
    
    # Execute model to get expected output
    input_t = {input_m.name: in_tensor}
    out_name = model_wrapper.graph.output[0].name
    
    y_ref = oxe.execute_onnx(model_wrapper, input_t, True)
    
    # Save outputs
    np.save(os.path.join(output_dir, "expected_output.npy"), y_ref[out_name])
    np.savez(os.path.join(output_dir, "expected_context.npz"), **y_ref)
    
    return in_tensor, y_ref[out_name]


def run_brainsmith_dse(model, args):
    """Run Brainsmith with new execution tree architecture."""
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    model_dir = os.path.join(args.output_dir, "intermediate_models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Simplify model (matches old hw_compiler.py)
    model, check = simplify(model)
    if not check:
        raise RuntimeError("Unable to simplify the Brevitas BERT model")
    
    # Save simplified model
    onnx.save(model, os.path.join(model_dir, "simp.onnx"))
    # Also save to debug directory for comparison
    debug_dir = os.path.join(args.output_dir, "debug_models")
    onnx.save(model, os.path.join(debug_dir, "01_after_simplify.onnx"))
    print(f"Saved simplified model to debug_models/01_after_simplify.onnx")
    
    # Run cleanup
    cleanup(
        in_file=os.path.join(model_dir, "simp.onnx"),
        out_file=os.path.join(args.output_dir, "df_input.onnx")
    )
    
    # Save a copy of the cleaned model for visualization
    import shutil
    debug_dir = os.path.join(args.output_dir, "debug_models")
    os.makedirs(debug_dir, exist_ok=True)
    shutil.copy(
        os.path.join(args.output_dir, "df_input.onnx"),
        os.path.join(debug_dir, "02_after_qonnx_cleanup.onnx")
    )
    
    # Get blueprint path from args
    blueprint_path = Path(__file__).parent / args.blueprint
    
    # Forge the FPGA accelerator
    print("Forging FPGA accelerator...")
    results = forge(
        model_path=os.path.join(args.output_dir, "df_input.onnx"),
        blueprint_path=str(blueprint_path),
        output_dir=args.output_dir
    )
    
    # Results are automatically logged by forge()
    # Just check if we succeeded
    stats = results.stats
    if stats['successful'] == 0:
        raise RuntimeError(f"No successful builds")
    
    # The new execution tree handles output automatically
    final_model_dst = os.path.join(args.output_dir, "output.onnx")
    
    # Find the output from the successful execution
    for segment_id, result in results.segment_results.items():
        if result.success and result.output_model:
            shutil.copy2(result.output_model, final_model_dst)
            break
    
    # Handle shell metadata (matches old hw_compiler.py)
    handover_file = os.path.join(args.output_dir, "stitched_ip", "shell_handover.json")
    if os.path.exists(handover_file):
        with open(handover_file, "r") as fp:
            handover = json.load(fp)
        handover["num_layers"] = args.num_hidden_layers
        with open(handover_file, "w") as fp:
            json.dump(handover, fp, indent=4)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Modern BERT FINN demo - Exact parity with old system using Brainsmith DSE'
    )
    
    # Model configuration
    parser.add_argument('-o', '--output', help='Output build directory name', required=True)
    parser.add_argument('-z', '--hidden_size', type=int, default=384, 
                       help='BERT hidden_size parameter')
    parser.add_argument('-n', '--num_attention_heads', type=int, default=12, 
                       help='BERT num_attention_heads parameter')
    parser.add_argument('-l', '--num_hidden_layers', type=int, default=1, 
                       help='Number of hidden layers')
    parser.add_argument('-i', '--intermediate_size', type=int, default=1536, 
                       help='BERT intermediate_size parameter')
    parser.add_argument('-b', '--bitwidth', type=int, default=8, 
                       help='Quantization bitwidth (4 or 8)')
    parser.add_argument('-q', '--seqlen', type=int, default=128, 
                       help='Sequence length parameter')
    
    # Blueprint configuration
    parser.add_argument('--blueprint', type=str, default='bert_demo.yaml',
                       help='Blueprint YAML file to use (default: bert_demo.yaml)')
    
    args = parser.parse_args()
    
    # Determine output directory
    build_dir = get_build_dir()
    print(build_dir)
    args.output_dir = os.path.join(str(build_dir), args.output)
    
    print("=" * 70)
    print("BERT Demo Using Brainsmith DSE")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Hidden layers: {args.num_hidden_layers}")
    print(f"  Hidden size: {args.hidden_size}")
    print(f"  Attention heads: {args.num_attention_heads}")
    print(f"  Intermediate size: {args.intermediate_size}")
    print(f"  Bitwidth: {args.bitwidth}")
    print(f"  Sequence length: {args.seqlen}")
    print(f"  Blueprint: {args.blueprint}")
    print(f"  Output directory: {args.output_dir}")
    print("=" * 70)
    
    try:
        # Step 1: Generate BERT model
        print("\nStep 1: Generating quantized BERT model...")
        model = generate_bert_model(args)
        
        # Step 2: Run Brainsmith DSE
        print("\nStep 2: Running Brainsmith DSE pipeline...")
        result = run_brainsmith_dse(model, args)
        
        print("\n" + "=" * 70)
        print("BUILD COMPLETED SUCCESSFULLY")
        print("=" * 70)
        print(f"Output directory: {args.output_dir}")
        
    except Exception as e:
        print(f"\nERROR: Build failed with error: {e}")
        raise


if __name__ == "__main__":
    main()