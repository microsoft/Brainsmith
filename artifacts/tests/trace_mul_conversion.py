#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper
import os

# Load the model after bert_streamlining
model_path = "/home/tafk/builds/brainsmith/work/intermediate_models/simp.onnx"
if os.path.exists(model_path):
    model = ModelWrapper(model_path)
    
    print("Checking for Mul nodes and their properties:")
    for i, node in enumerate(model.graph.node):
        if "Mul" in node.op_type:
            print(f"\nNode {i}: {node.op_type}")
            print(f"  Name: {node.name}")
            print(f"  Domain: '{node.domain}'")
            print(f"  Inputs: {list(node.input)}")
            print(f"  Outputs: {list(node.output)}")
            
            # Check if this is a scalar multiplication
            for inp in node.input:
                init = model.get_initializer(inp)
                if init is not None:
                    print(f"  Input '{inp}' is an initializer with shape {init.shape}")
                    
    # Check what domains are imported
    print("\nOpset imports:")
    for imp in model.onnx_model.opset_import:
        print(f"  Domain: '{imp.domain}' Version: {imp.version}")