#!/usr/bin/env python3
"""Debug script to find FIFO shape mismatches in ONNX model."""

import onnx
from finn.core.modelwrapper import ModelWrapper
from finn.transformation.fpgadataflow.insert_fifo import _suitable_node
from finn.custom_op.registry import getCustomOp
import numpy as np

# Load the model just before FIFO sizing
model_path = 'finn_output/intermediate_models/step_measure_rtlsim_performance.onnx'
print(f"Loading model from: {model_path}")
model = ModelWrapper(model_path)

print(f"Total nodes in graph: {len(model.graph.node)}")
print("\nChecking for folded shape mismatches between connected nodes...\n")

# Track all connections
connections = []
mismatches = []

for node in model.graph.node:
    if _suitable_node(node):
        try:
            n0 = getCustomOp(node)
            for idx_out, output_name in enumerate(node.output):
                consumers = model.find_direct_successors(node)
                if consumers is not None and len(consumers) == 1:
                    consumer = consumers[0]
                    if _suitable_node(consumer):
                        n1 = getCustomOp(consumer)
                        # Get folded shapes
                        fld_shape = n0.get_folded_output_shape(ind=idx_out)
                        
                        # Find which input of consumer connects to this output
                        for idx_inp, inp in enumerate(consumer.input):
                            if inp == output_name:
                                fld_shape_2 = n1.get_folded_input_shape(ind=idx_inp)
                                
                                # Record connection
                                conn_info = {
                                    'producer': node.name,
                                    'producer_type': node.op_type,
                                    'consumer': consumer.name,
                                    'consumer_type': consumer.op_type,
                                    'tensor': output_name,
                                    'producer_shape': fld_shape,
                                    'consumer_shape': fld_shape_2,
                                    'shapes_match': tuple(fld_shape) == tuple(fld_shape_2)
                                }
                                connections.append(conn_info)
                                
                                # Check for mismatch
                                if not conn_info['shapes_match']:
                                    mismatches.append(conn_info)
                                    print(f"MISMATCH FOUND:")
                                    print(f"  Producer: {node.name} ({node.op_type})")
                                    print(f"    Output shape: {fld_shape}")
                                    print(f"  Consumer: {consumer.name} ({consumer.op_type})")
                                    print(f"    Input shape: {fld_shape_2}")
                                    print(f"  Tensor: {output_name}")
                                    
                                    # Detailed analysis
                                    if fld_shape[-1] == fld_shape_2[-1]:
                                        print(f"  ✓ Stream widths match: {fld_shape[-1]}")
                                    else:
                                        print(f"  ✗ Stream width mismatch: {fld_shape[-1]} vs {fld_shape_2[-1]}")
                                    
                                    size1 = np.prod(fld_shape)
                                    size2 = np.prod(fld_shape_2)
                                    if size1 == size2:
                                        print(f"  ✓ Total sizes match: {size1}")
                                    else:
                                        print(f"  ✗ Size mismatch: {size1} vs {size2}")
                                    print()
                                
        except Exception as e:
            print(f"Error processing node {node.name}: {e}")

# Summary
print(f"\nSummary:")
print(f"Total connections analyzed: {len(connections)}")
print(f"Mismatches found: {len(mismatches)}")

if len(mismatches) == 0:
    print("\nNo shape mismatches found! The error might be in a different model stage.")
    print("\nAll connections:")
    for conn in connections[-10:]:  # Show last 10 connections
        print(f"  {conn['producer']} -> {conn['consumer']}: {conn['producer_shape']} -> {conn['consumer_shape']}")
else:
    print("\nMismatched connections:")
    for m in mismatches:
        print(f"  {m['producer']} -> {m['consumer']}: {m['producer_shape']} -> {m['consumer_shape']}")