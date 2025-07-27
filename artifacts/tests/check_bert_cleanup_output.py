#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper

# This is the model that remove_head_step actually receives
model = ModelWrapper('/home/tafk/builds/brainsmith/work/bert_cleanup/input.onnx')

print("Model after bert_cleanup_step:")
print(f"  Inputs: {[inp.name for inp in model.graph.input]}")
print(f"  Outputs: {[out.name for out in model.graph.output]}")

# Trace path from input
current_tensor = model.graph.input[0].name
print(f"\nTracing from {current_tensor}:")

for i in range(10):
    consumers = model.find_consumers(current_tensor)
    if not consumers:
        break
        
    if len(consumers) > 1:
        print(f"  Multiple consumers for {current_tensor}!")
        for cons in consumers:
            print(f"    - {cons.op_type} ({cons.name})")
        break
        
    node = consumers[0]
    print(f"  {node.op_type} ({node.name}) - outputs: {node.output}")
    
    if node.op_type == "LayerNormalization":
        print("  Found LayerNormalization!")
        # Check what comes after
        ln_out = node.output[0]
        ln_consumers = model.find_consumers(ln_out)
        print(f"  LayerNorm output has {len(ln_consumers)} consumers")
        break
        
    if len(node.output) > 1:
        print(f"  ERROR: Node has {len(node.output)} outputs!")
        break
        
    current_tensor = node.output[0]