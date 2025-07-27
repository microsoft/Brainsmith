#!/usr/bin/env python3
import onnx
from qonnx.core.modelwrapper import ModelWrapper

# Check the model in work/root
model = ModelWrapper('/home/tafk/builds/brainsmith/work/root/input.onnx')

# Start from the first input  
current_tensor = model.graph.input[0].name
print(f"Model in work/root/input.onnx - Starting from input: {current_tensor}")

# Check if this is the actual model being processed
print("\nModel info:")
print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
print(f"  Outputs: {[out.name for out in model.graph.output]}")
print(f"  Node count: {len(model.graph.node)}")

# Follow path to first LayerNorm
while True:
    consumers = model.find_consumers(current_tensor)
    if not consumers:
        print(f"No consumers found for {current_tensor}")
        break
    
    current_node = consumers[0]
    print(f"Node: {current_node.op_type} ({current_node.name}) - outputs: {len(current_node.output)}")
    
    if current_node.op_type == "LayerNormalization":
        print("Found LayerNormalization!")
        break
        
    if len(current_node.output) != 1:
        print(f"ERROR: Node {current_node.name} has {len(current_node.output)} outputs!")
        print(f"  Output names: {current_node.output}")
        # Let's check what consumers this has
        for out in current_node.output:
            cons = model.find_consumers(out)
            print(f"  Output '{out}' has {len(cons)} consumers")
        break
        
    current_tensor = current_node.output[0]