#!/usr/bin/env python3
"""Analyze FIFO shape mismatch patterns in detail."""

import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import numpy as np
from collections import defaultdict


def get_folded_output_shape(node, model):
    """Get the folded output shape of a node."""
    try:
        inst = getCustomOp(node)
        folded_oshape = inst.get_folded_output_shape()
        return folded_oshape
    except Exception as e:
        return None


def get_folded_input_shape(node, model, input_idx=0):
    """Get the folded input shape expected by a node."""
    try:
        inst = getCustomOp(node)
        folded_ishape = inst.get_folded_input_shape(input_idx)
        return folded_ishape
    except Exception as e:
        return None


def get_normal_output_shape(node, model):
    """Get the normal (unfolded) output shape."""
    try:
        inst = getCustomOp(node)
        normal_oshape = inst.get_normal_output_shape()
        return normal_oshape
    except Exception as e:
        return None


def get_normal_input_shape(node, model, input_idx=0):
    """Get the normal (unfolded) input shape."""
    try:
        inst = getCustomOp(node)
        normal_ishape = inst.get_normal_input_shape(input_idx)
        return normal_ishape
    except Exception as e:
        return None


def analyze_shape_mismatches_detailed(model_path):
    """Analyze the model for FIFO shape mismatches with more detail."""
    print(f"Loading model from: {model_path}")
    model = ModelWrapper(model_path)
    
    mismatches = []
    
    # Build a map of tensor producers and consumers
    tensor_producers = {}
    tensor_consumers = {}
    
    for node in model.graph.node:
        # Record producers
        for output in node.output:
            tensor_producers[output] = node
        
        # Record consumers
        for idx, input_tensor in enumerate(node.input):
            if input_tensor not in tensor_consumers:
                tensor_consumers[input_tensor] = []
            tensor_consumers[input_tensor].append((node, idx))
    
    # Check each connection for shape mismatches
    for tensor_name, producer_node in tensor_producers.items():
        if tensor_name not in tensor_consumers:
            continue
            
        # Get producer's shapes
        producer_folded = get_folded_output_shape(producer_node, model)
        producer_normal = get_normal_output_shape(producer_node, model)
        
        if producer_folded is None:
            continue
            
        # Check each consumer
        for consumer_node, input_idx in tensor_consumers[tensor_name]:
            consumer_folded = get_folded_input_shape(consumer_node, model, input_idx)
            consumer_normal = get_normal_input_shape(consumer_node, model, input_idx)
            
            if consumer_folded is None:
                continue
                
            # Compare shapes
            if producer_folded != consumer_folded:
                # Calculate the total elements to see if they match
                producer_elems = np.prod(producer_folded) if producer_folded else 0
                consumer_elems = np.prod(consumer_folded) if consumer_folded else 0
                
                mismatch = {
                    'producer_name': producer_node.name,
                    'producer_type': producer_node.op_type,
                    'producer_folded': producer_folded,
                    'producer_normal': producer_normal,
                    'producer_elems': producer_elems,
                    'consumer_name': consumer_node.name,
                    'consumer_type': consumer_node.op_type,
                    'consumer_folded': consumer_folded,
                    'consumer_normal': consumer_normal,
                    'consumer_elems': consumer_elems,
                    'tensor_name': tensor_name,
                    'input_idx': input_idx,
                    'elems_match': producer_elems == consumer_elems
                }
                mismatches.append(mismatch)
    
    return mismatches


def categorize_mismatches(mismatches):
    """Categorize mismatches by pattern type."""
    categories = defaultdict(list)
    
    for mismatch in mismatches:
        # Categorize by producer-consumer type pair
        key = f"{mismatch['producer_type']} -> {mismatch['consumer_type']}"
        categories[key].append(mismatch)
    
    return categories


def print_categorized_mismatches(mismatches):
    """Print mismatches categorized by pattern."""
    if not mismatches:
        print("\nNo FIFO shape mismatches found!")
        return
    
    categories = categorize_mismatches(mismatches)
    
    print(f"\nFound {len(mismatches)} FIFO shape mismatches in {len(categories)} patterns:\n")
    print("=" * 100)
    
    for pattern, pattern_mismatches in categories.items():
        print(f"\nPattern: {pattern} ({len(pattern_mismatches)} occurrences)")
        print("-" * 100)
        
        # Show first example in detail
        example = pattern_mismatches[0]
        print(f"Example:")
        print(f"  Producer: {example['producer_name']}")
        print(f"    Folded shape: {example['producer_folded']}")
        print(f"    Normal shape: {example['producer_normal']}")
        print(f"    Total elements: {example['producer_elems']}")
        print(f"  Consumer: {example['consumer_name']}")  
        print(f"    Expected folded shape: {example['consumer_folded']}")
        print(f"    Normal shape: {example['consumer_normal']}")
        print(f"    Total elements: {example['consumer_elems']}")
        print(f"  Elements match: {example['elems_match']}")
        
        if len(pattern_mismatches) > 1:
            print(f"\n  Other occurrences:")
            for m in pattern_mismatches[1:]:
                print(f"    - {m['producer_name']} -> {m['consumer_name']}")
        
        print("-" * 100)


def suggest_fixes(mismatches):
    """Suggest potential fixes for the mismatches."""
    print("\n" + "=" * 100)
    print("SUGGESTED FIXES:")
    print("=" * 100)
    
    categories = categorize_mismatches(mismatches)
    
    for pattern, pattern_mismatches in categories.items():
        print(f"\nFor pattern: {pattern}")
        
        if "Thresholding_rtl -> MVAU_rtl" in pattern:
            print("  - Thresholding output shape doesn't match MVAU input requirements")
            print("  - Thresholding outputs (0, 128, 384, 1) but MVAU expects (1, 128, 96, 4)")
            print("  - This is a shape transformation issue - need StreamingDataWidthConverter")
            
        elif "MVAU_rtl -> Shuffle_hls" in pattern:
            print("  - MVAU output shape doesn't match Shuffle input requirements")
            print("  - MVAU outputs (1, 128, 128, 3) but Shuffle expects (1, 128, 12, 32, 1)")
            print("  - The shapes have same total elements but different chunking")
            
        elif "Thresholding_rtl -> DynMVU_rtl" in pattern:
            print("  - Thresholding output doesn't match DynMVU input packing")
            print("  - Need to adjust folding parameters or add reshaping")
            
        elif "MVAU_rtl -> ElementwiseMul_hls" in pattern:
            print("  - MVAU folded output doesn't match ElementwiseMul expectations")
            print("  - Shape transformation needed between layers")
            
        elif "MVAU_rtl -> Thresholding_rtl" in pattern:
            print("  - MVAU output packing doesn't match Thresholding input")
            print("  - Need to ensure consistent folding between layers")


def main():
    # Analyze the model after constrain_folding
    model_path = "/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/constrain_folding_and_set_pumped_compute_step.onnx"
    
    mismatches = analyze_shape_mismatches_detailed(model_path)
    print_categorized_mismatches(mismatches)
    suggest_fixes(mismatches)


if __name__ == "__main__":
    main()