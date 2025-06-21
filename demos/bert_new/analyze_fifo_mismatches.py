#!/usr/bin/env python3
"""Analyze FINN intermediate model for FIFO shape mismatches."""

import onnx
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
import numpy as np


def get_folded_output_shape(node, model):
    """Get the folded output shape of a node."""
    try:
        inst = getCustomOp(node)
        # Get folded output shape - this is what the producer will output
        folded_oshape = inst.get_folded_output_shape()
        return folded_oshape
    except Exception as e:
        print(f"Warning: Could not get folded output shape for {node.name}: {e}")
        return None


def get_folded_input_shape(node, model, input_idx=0):
    """Get the folded input shape expected by a node."""
    try:
        inst = getCustomOp(node)
        # Get folded input shape - this is what the consumer expects
        folded_ishape = inst.get_folded_input_shape(input_idx)
        return folded_ishape
    except Exception as e:
        print(f"Warning: Could not get folded input shape for {node.name}: {e}")
        return None


def analyze_shape_mismatches(model_path):
    """Analyze the model for FIFO shape mismatches."""
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
            
        # Get producer's folded output shape
        producer_shape = get_folded_output_shape(producer_node, model)
        if producer_shape is None:
            continue
            
        # Check each consumer
        for consumer_node, input_idx in tensor_consumers[tensor_name]:
            consumer_shape = get_folded_input_shape(consumer_node, model, input_idx)
            if consumer_shape is None:
                continue
                
            # Compare shapes
            if producer_shape != consumer_shape:
                mismatch = {
                    'producer_name': producer_node.name,
                    'producer_type': producer_node.op_type,
                    'producer_shape': producer_shape,
                    'consumer_name': consumer_node.name,
                    'consumer_type': consumer_node.op_type,
                    'consumer_shape': consumer_shape,
                    'tensor_name': tensor_name,
                    'input_idx': input_idx
                }
                mismatches.append(mismatch)
    
    return mismatches


def print_mismatches(mismatches):
    """Print mismatches in a clear format."""
    if not mismatches:
        print("\nNo FIFO shape mismatches found!")
        return
        
    print(f"\nFound {len(mismatches)} FIFO shape mismatches:\n")
    print("=" * 80)
    
    for i, mismatch in enumerate(mismatches, 1):
        print(f"\nMismatch #{i}:")
        print(f"  Tensor: {mismatch['tensor_name']}")
        print(f"  Producer: {mismatch['producer_name']} ({mismatch['producer_type']})")
        print(f"    Output shape: {mismatch['producer_shape']}")
        print(f"  Consumer: {mismatch['consumer_name']} ({mismatch['consumer_type']})")
        print(f"    Expected input shape: {mismatch['consumer_shape']}")
        print(f"    Input index: {mismatch['input_idx']}")
        print("-" * 80)


def main():
    # Analyze the model after constrain_folding
    model_path = "/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/constrain_folding_and_set_pumped_compute_step.onnx"
    
    mismatches = analyze_shape_mismatches(model_path)
    print_mismatches(mismatches)
    
    # Also check the model after applying folding config
    print("\n\n" + "=" * 80)
    print("Checking model after apply_folding_config:")
    print("=" * 80)
    
    model_path2 = "/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/step_apply_folding_config.onnx"
    mismatches2 = analyze_shape_mismatches(model_path2)
    print_mismatches(mismatches2)
    
    # Check after hw_codegen as well
    print("\n\n" + "=" * 80)
    print("Checking model after hw_codegen:")
    print("=" * 80)
    
    model_path3 = "/home/tafk/dev/brainsmith-1/demos/bert_new/finn_output/intermediate_models/step_hw_codegen.onnx"
    mismatches3 = analyze_shape_mismatches(model_path3)
    print_mismatches(mismatches3)


if __name__ == "__main__":
    main()