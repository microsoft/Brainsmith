#!/usr/bin/env python3
"""
Compare BERT models between old and modern demos to identify structural differences.
"""
import os
import sys
import onnx
from collections import defaultdict

def analyze_model(model_path):
    """Analyze an ONNX model and return its structure."""
    if not os.path.exists(model_path):
        return None
        
    model = onnx.load(model_path)
    
    analysis = {
        'path': model_path,
        'inputs': [],
        'outputs': [],
        'nodes': [],
        'node_types': defaultdict(int),
        'value_info': len(model.graph.value_info),
        'initializers': len(model.graph.initializer),
    }
    
    # Analyze inputs
    for inp in model.graph.input:
        shape = [d.dim_value if d.HasField('dim_value') else d.dim_param 
                 for d in inp.type.tensor_type.shape.dim]
        analysis['inputs'].append({
            'name': inp.name,
            'shape': shape,
            'dtype': inp.type.tensor_type.elem_type
        })
    
    # Analyze outputs
    for out in model.graph.output:
        analysis['outputs'].append(out.name)
    
    # Analyze nodes
    for i, node in enumerate(model.graph.node):
        analysis['nodes'].append({
            'index': i,
            'name': node.name,
            'op_type': node.op_type,
            'inputs': list(node.input),
            'outputs': list(node.output)
        })
        analysis['node_types'][node.op_type] += 1
    
    # Find LayerNormalization nodes
    analysis['layer_norm_nodes'] = [
        (i, n.name) for i, n in enumerate(model.graph.node) 
        if n.op_type == "LayerNormalization"
    ]
    
    return analysis

def compare_models(old_analysis, new_analysis):
    """Compare two model analyses and report differences."""
    print("\n=== Model Comparison Report ===\n")
    
    # Compare inputs
    print("INPUTS:")
    print(f"  Old: {len(old_analysis['inputs'])} inputs")
    for inp in old_analysis['inputs']:
        print(f"    - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
    print(f"  New: {len(new_analysis['inputs'])} inputs")
    for inp in new_analysis['inputs']:
        print(f"    - {inp['name']}: shape={inp['shape']}, dtype={inp['dtype']}")
    
    # Compare node types
    print("\nNODE TYPE DISTRIBUTION:")
    all_types = set(old_analysis['node_types'].keys()) | set(new_analysis['node_types'].keys())
    for node_type in sorted(all_types):
        old_count = old_analysis['node_types'].get(node_type, 0)
        new_count = new_analysis['node_types'].get(node_type, 0)
        if old_count != new_count:
            print(f"  {node_type}: Old={old_count}, New={new_count} (DIFFERENT)")
        else:
            print(f"  {node_type}: {old_count}")
    
    # Compare LayerNormalization nodes
    print("\nLAYERNORMALIZATION NODES:")
    print(f"  Old: {len(old_analysis['layer_norm_nodes'])} nodes")
    for idx, name in old_analysis['layer_norm_nodes'][:3]:
        print(f"    - Index {idx}: {name}")
    print(f"  New: {len(new_analysis['layer_norm_nodes'])} nodes")
    for idx, name in new_analysis['layer_norm_nodes'][:3]:
        print(f"    - Index {idx}: {name}")
    
    # Compare first few nodes
    print("\nFIRST 5 NODES:")
    print("  Old:")
    for node in old_analysis['nodes'][:5]:
        print(f"    {node['index']}: {node['name']} ({node['op_type']})")
        print(f"       Inputs: {node['inputs'][:2]}...")
    print("  New:")
    for node in new_analysis['nodes'][:5]:
        print(f"    {node['index']}: {node['name']} ({node['op_type']})")
        print(f"       Inputs: {node['inputs'][:2]}...")
    
    # Check what connects to the input
    print("\nINPUT CONNECTIVITY:")
    for analysis, label in [(old_analysis, "Old"), (new_analysis, "New")]:
        if analysis['inputs']:
            input_name = analysis['inputs'][0]['name']
            consumers = []
            for node in analysis['nodes']:
                if input_name in node['inputs']:
                    consumers.append(f"{node['name']} ({node['op_type']})")
            print(f"  {label} - Nodes consuming input '{input_name}': {len(consumers)}")
            for consumer in consumers[:3]:
                print(f"    - {consumer}")

def main():
    # Paths to compare
    comparisons = [
        ("Initial Brevitas", 
         "old_bert_output/initial_brevitas.onnx",  # You'll need to save this from old demo
         "bert_modern_output/debug_models/00_initial_brevitas.onnx"),
        ("After Simplify",
         "old_bert_output/simp.onnx",  # If available
         "bert_modern_output/debug_models/01_after_simplify.onnx"),
        ("Before Remove Head",
         None,  # Old demo doesn't save intermediate
         "bert_modern_output/debug_models/05_before_remove_head.onnx"),
    ]
    
    for stage, old_path, new_path in comparisons:
        print(f"\n{'='*60}")
        print(f"STAGE: {stage}")
        print(f"{'='*60}")
        
        if old_path and os.path.exists(old_path):
            old_analysis = analyze_model(old_path)
            new_analysis = analyze_model(new_path)
            
            if old_analysis and new_analysis:
                compare_models(old_analysis, new_analysis)
            else:
                print(f"Could not load models for comparison")
        else:
            # Just analyze the new model
            if os.path.exists(new_path):
                new_analysis = analyze_model(new_path)
                if new_analysis:
                    print(f"\nAnalyzing new model only (old not available):")
                    print(f"  Inputs: {len(new_analysis['inputs'])}")
                    print(f"  Nodes: {len(new_analysis['nodes'])}")
                    print(f"  LayerNorm nodes: {len(new_analysis['layer_norm_nodes'])}")
                    print(f"  Node types: {dict(new_analysis['node_types'])}")

if __name__ == "__main__":
    main()