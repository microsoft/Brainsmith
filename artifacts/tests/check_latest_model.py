#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper
import os

# Find the latest intermediate model
base_dir = "/home/tafk/builds/brainsmith"
candidates = []

# Check various possible locations
for subdir in ["bert_streamlining", "infer_hardware", "qonnx_to_finn", "bert_cleanup"]:
    path = os.path.join(base_dir, subdir, "input.onnx")
    if os.path.exists(path):
        candidates.append((os.path.getmtime(path), path))

if candidates:
    candidates.sort(reverse=True)
    latest_model = candidates[0][1]
    print(f"Checking latest model: {latest_model}")
    
    model = ModelWrapper(latest_model)
    
    # Look for ElementwiseMul nodes
    elementwise_nodes = []
    for i, node in enumerate(model.graph.node):
        if "Elementwise" in node.op_type or "Mul" in node.op_type:
            elementwise_nodes.append((i, node))
            
    print(f"\nFound {len(elementwise_nodes)} elementwise/mul nodes:")
    for idx, node in elementwise_nodes[:5]:  # Show first 5
        print(f"  Node {idx}: {node.op_type} ({node.name})")
        print(f"    Domain: '{node.domain}'")
else:
    print("No intermediate models found")