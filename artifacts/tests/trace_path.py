#!/usr/bin/env python3
import onnx
from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper('/tmp/bert_test_run/root/input.onnx')

# Start from the first input
current_tensor = model.graph.input[0].name
print(f"Starting from input: {current_tensor}")

# Follow path to first LayerNorm
path = []
while True:
    consumers = model.find_consumers(current_tensor)
    if not consumers:
        print(f"No consumers found for {current_tensor}")
        break
    
    if len(consumers) > 1:
        print(f"WARNING: Multiple consumers ({len(consumers)}) for {current_tensor}")
        for i, node in enumerate(consumers):
            print(f"  Consumer {i}: {node.op_type} -> {node.output}")
    
    current_node = consumers[0] if consumers else None
    if not current_node:
        break
        
    path.append(f"{current_node.op_type} ({current_node.name}) -> {current_node.output}")
    print(f"Node: {current_node.op_type} - outputs: {len(current_node.output)} - {current_node.output}")
    
    if current_node.op_type == "LayerNormalization":
        print("Found LayerNormalization!")
        break
        
    if len(current_node.output) != 1:
        print(f"ERROR: Node has {len(current_node.output)} outputs, expected 1")
        break
        
    current_tensor = current_node.output[0]

print("\nFull path:")
for step in path:
    print(f"  {step}")