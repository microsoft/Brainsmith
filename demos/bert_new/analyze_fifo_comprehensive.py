#!/usr/bin/env python3
"""Comprehensive FIFO shape mismatch analysis for BERT model."""

import onnx
import qonnx.custom_op.registry as registry
# from tabulate import tabulate  # Optional, fallback to simple printing
import sys

def get_folded_shape(node):
    """Get the folded output shape of a node."""
    try:
        inst = registry.getCustomOp(node)
        if hasattr(inst, 'get_folded_output_shape'):
            return inst.get_folded_output_shape()
        elif hasattr(inst, 'get_normal_output_shape'):
            return inst.get_normal_output_shape()
        else:
            return None
    except:
        return None

def get_folded_input_shape(node):
    """Get the expected folded input shape of a node."""
    try:
        inst = registry.getCustomOp(node)
        if hasattr(inst, 'get_folded_input_shape'):
            return inst.get_folded_input_shape()
        elif hasattr(inst, 'get_normal_input_shape'):
            return inst.get_normal_input_shape()
        else:
            return None
    except:
        return None

def analyze_model(model_path):
    """Analyze model for FIFO shape mismatches."""
    model = onnx.load(model_path)
    
    # Build producer map
    producer_map = {}
    for node in model.graph.node:
        for output in node.output:
            producer_map[output] = node
    
    # Find all mismatches
    mismatches = []
    for node in model.graph.node:
        # Get producer's output shape
        producer_shape = get_folded_shape(node)
        if not producer_shape:
            continue
            
        # Check each output tensor
        for output_tensor in node.output:
            # Find consumers
            consumers = []
            for consumer_node in model.graph.node:
                if output_tensor in consumer_node.input:
                    consumers.append(consumer_node)
            
            # Check shape compatibility with each consumer
            for consumer in consumers:
                consumer_shape = get_folded_input_shape(consumer)
                if not consumer_shape:
                    continue
                
                # Get the input index for this tensor
                input_idx = list(consumer.input).index(output_tensor)
                
                # For multi-input nodes, we need the shape for the specific input
                if isinstance(consumer_shape, list) and len(consumer_shape) > 1:
                    if input_idx < len(consumer_shape):
                        expected_shape = consumer_shape[input_idx]
                    else:
                        expected_shape = consumer_shape[0]
                else:
                    expected_shape = consumer_shape
                
                # Compare shapes
                if producer_shape != expected_shape:
                    mismatches.append({
                        'producer': node.name,
                        'producer_type': node.op_type,
                        'producer_shape': producer_shape,
                        'tensor': output_tensor,
                        'consumer': consumer.name,
                        'consumer_type': consumer.op_type,
                        'consumer_shape': expected_shape,
                        'shape_match': False
                    })
    
    return mismatches

def print_analysis(mismatches):
    """Print detailed analysis of mismatches."""
    print(f"\nTotal FIFO shape mismatches found: {len(mismatches)}\n")
    
    if not mismatches:
        print("No mismatches found!")
        return
    
    # Group by pattern
    patterns = {}
    for m in mismatches:
        pattern = f"{m['producer_type']} → {m['consumer_type']}"
        if pattern not in patterns:
            patterns[pattern] = []
        patterns[pattern].append(m)
    
    print("Mismatch patterns:")
    for pattern, matches in patterns.items():
        print(f"\n{pattern}: {len(matches)} occurrences")
        print("-" * 80)
        
        # Create table data
        table_data = []
        for m in matches:
            prod_elements = 1
            cons_elements = 1
            for d in m['producer_shape']:
                prod_elements *= d
            for d in m['consumer_shape']:
                cons_elements *= d
                
            table_data.append([
                m['producer'].replace('_rtl', ''),
                str(m['producer_shape']),
                f"{prod_elements:,}",
                "→",
                m['consumer'].replace('_rtl', ''),
                str(m['consumer_shape']),
                f"{cons_elements:,}"
            ])
        
        headers = ["Producer", "Output Shape", "Elements", "", "Consumer", "Input Shape", "Elements"]
        # Simple table printing without tabulate
        print("  " + " | ".join(headers))
        print("  " + "-" * (len(" | ".join(headers)) + 20))
        for row in table_data:
            print("  " + " | ".join(str(x) for x in row))
    
    # Detailed listing
    print("\n\nDetailed mismatch listing:")
    print("=" * 100)
    for i, m in enumerate(mismatches, 1):
        print(f"\nMismatch {i}:")
        print(f"  Producer: {m['producer']} ({m['producer_type']})")
        print(f"  Output shape: {m['producer_shape']}")
        print(f"  Tensor: {m['tensor']}")
        print(f"  Consumer: {m['consumer']} ({m['consumer_type']})")
        print(f"  Expected shape: {m['consumer_shape']}")

if __name__ == "__main__":
    # Find the latest intermediate model
    import glob
    import os
    
    model_dir = "finn_output/intermediate_models"
    model_files = glob.glob(f"{model_dir}/*.onnx")
    
    if not model_files:
        print(f"No model files found in {model_dir}")
        sys.exit(1)
    
    # Find the constrain_folding model
    constrain_model = None
    for f in sorted(model_files):
        if "constrain_folding" in f:
            constrain_model = f
    
    if not constrain_model:
        # Use the latest model
        constrain_model = sorted(model_files)[-1]
    
    print(f"Analyzing model: {constrain_model}")
    
    mismatches = analyze_model(constrain_model)
    print_analysis(mismatches)