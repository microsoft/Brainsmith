#!/usr/bin/env python3
"""Check why Thresholding nodes have 0 in first dimension of output shape."""

import sys
sys.path.append('/home/tafk/dev/brainsmith-1/deps/finn/src')
sys.path.append('/home/tafk/dev/brainsmith-1/deps/qonnx/src')
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from finn.util.fpgadataflow import is_fpgadataflow_node

# Check model after target_fps parallelization
model_path = 'finn_output/intermediate_models/step_target_fps_parallelization.onnx'
print(f"Checking model: {model_path}")

try:
    model = ModelWrapper(model_path)
    
    # Find all Thresholding nodes
    for node in model.graph.node:
        if node.op_type.startswith("Thresholding"):
            print(f"\nNode: {node.name}")
            try:
                custom_op = getCustomOp(node)
                
                # Check various attributes
                pe = custom_op.get_nodeattr("PE") if hasattr(custom_op, 'get_nodeattr') else "N/A"
                num_channels = custom_op.get_nodeattr("NumChannels") if hasattr(custom_op, 'get_nodeattr') else "N/A"
                
                print(f"  PE: {pe}")
                print(f"  NumChannels: {num_channels}")
                
                # Try to get shapes
                if hasattr(custom_op, 'get_normal_output_shape'):
                    normal_shape = custom_op.get_normal_output_shape()
                    print(f"  Normal output shape: {normal_shape}")
                
                if hasattr(custom_op, 'get_folded_output_shape'):
                    folded_shape = custom_op.get_folded_output_shape()
                    print(f"  Folded output shape: {folded_shape}")
                    
                    # Check if first dimension is 0
                    if folded_shape[0] == 0:
                        print(f"  ⚠️  WARNING: First dimension is 0!")
                        # Try to understand why
                        if num_channels != "N/A" and pe != "N/A":
                            print(f"  Calculation: NumChannels={num_channels}, PE={pe}")
                            print(f"  Channel fold = {num_channels}/{pe} = {num_channels/pe if pe != 0 else 'inf'}")
                
            except Exception as e:
                print(f"  Error: {e}")
                
except Exception as e:
    print(f"Error loading model: {e}")
    
# Also check the model after folding config is applied
print("\n" + "="*60)
model_path2 = 'finn_output/intermediate_models/step_apply_folding_config.onnx'
print(f"Checking model after folding config: {model_path2}")

try:
    model2 = ModelWrapper(model_path2)
    
    # Count Thresholding nodes
    thresh_count = 0
    for node in model2.graph.node:
        if node.op_type.startswith("Thresholding"):
            thresh_count += 1
            
    print(f"Total Thresholding nodes: {thresh_count}")
    
except Exception as e:
    print(f"Error: {e}")