#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper
import os

# Check models in the brainsmith intermediate_models directory
int_dir = "/home/tafk/builds/brainsmith/work/intermediate_models"
if os.path.exists(int_dir):
    models = [f for f in os.listdir(int_dir) if f.endswith('.onnx')]
    print(f"Models in {int_dir}:")
    for m in sorted(models):
        print(f"  {m}")
        
    # Check the latest one
    if models:
        latest = sorted(models)[-1]
        model_path = os.path.join(int_dir, latest)
        print(f"\nChecking {latest}:")
        
        model = ModelWrapper(model_path)
        
        # Look for problematic nodes
        for i, node in enumerate(model.graph.node):
            if node.op_type in ["ElementwiseMul", "Mul"]:
                print(f"  Node {i}: {node.op_type} ({node.name})")
                print(f"    Domain: '{node.domain}'")