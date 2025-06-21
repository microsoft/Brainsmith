#!/usr/bin/env python3
"""Trace where zero-dimension shapes are introduced in the pipeline."""

import onnx
import qonnx.custom_op.registry as registry
import os
import glob

def check_model_for_zero_dims(model_path):
    """Check if a model has any nodes with zero-dimension outputs."""
    model = onnx.load(model_path)
    zero_dim_nodes = []
    
    for node in model.graph.node:
        try:
            inst = registry.getCustomOp(node)
            if hasattr(inst, 'get_folded_output_shape'):
                shape = inst.get_folded_output_shape()
                if shape and any(d == 0 for d in shape):
                    zero_dim_nodes.append({
                        'name': node.name,
                        'type': node.op_type,
                        'shape': shape
                    })
        except:
            # Skip nodes we can't analyze
            pass
    
    return zero_dim_nodes

def main():
    # Get all intermediate models in chronological order
    model_files = [
        "cleanup_step.onnx",
        "remove_head_step.onnx", 
        "remove_tail_step.onnx",
        "qonnx_to_finn_step.onnx",
        "generate_reference_io_step.onnx",
        "streamlining_step.onnx",
        "infer_hardware_step.onnx",
        "step_create_dataflow_partition.onnx",
        "step_specialize_layers.onnx",
        "step_target_fps_parallelization.onnx",
        "step_apply_folding_config.onnx",
        "step_minimize_bit_width.onnx",
        "step_generate_estimate_reports.onnx",
        "step_hw_codegen.onnx",
        "step_hw_ipgen.onnx",
        "step_measure_rtlsim_performance.onnx",
        "constrain_folding_and_set_pumped_compute_step.onnx"
    ]
    
    model_dir = "finn_output/intermediate_models"
    
    print("Tracing zero-dimension introduction through pipeline:\n")
    print("=" * 80)
    
    first_occurrence = None
    
    for i, model_file in enumerate(model_files):
        model_path = os.path.join(model_dir, model_file)
        if not os.path.exists(model_path):
            print(f"{i+1}. {model_file}: NOT FOUND")
            continue
            
        zero_nodes = check_model_for_zero_dims(model_path)
        
        if zero_nodes:
            if not first_occurrence:
                first_occurrence = model_file
                print(f"\n🚨 FIRST OCCURRENCE OF ZERO DIMENSIONS: {model_file}")
                print("=" * 80)
            
            print(f"\n{i+1}. {model_file}: {len(zero_nodes)} nodes with zero dimensions")
            for node in zero_nodes:
                print(f"   - {node['name']} ({node['type']}): {node['shape']}")
        else:
            print(f"{i+1}. {model_file}: ✓ No zero dimensions")
    
    if first_occurrence:
        print(f"\n\n🎯 Zero dimensions first introduced at: {first_occurrence}")
        
        # Find the step that introduced it
        if first_occurrence in model_files:
            idx = model_files.index(first_occurrence)
            if idx > 0:
                print(f"   Introduced by step: {model_files[idx-1]} → {first_occurrence}")

if __name__ == "__main__":
    main()