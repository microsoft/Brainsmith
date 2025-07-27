#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper

# Load the model that remove_head_step will process
model = ModelWrapper('/tmp/bert_test_run/root/input.onnx')

print("Debugging remove_head_step logic...")
print(f"Model has {len(model.graph.input)} inputs")
print(f"First input: {model.graph.input[0].name}")

current_tensor = model.graph.input[0].name
step = 0

while True:
    print(f"\nStep {step}: Current tensor = '{current_tensor}'")
    
    # Use find_consumer (singular) like the code does
    try:
        current_node = model.find_consumer(current_tensor)
        print(f"  find_consumer returned: {current_node.op_type} ({current_node.name})")
        print(f"  Node outputs: {len(current_node.output)} - {current_node.output}")
    except Exception as e:
        print(f"  ERROR in find_consumer: {e}")
        # Try find_consumers (plural) to see what's actually there
        consumers = model.find_consumers(current_tensor) 
        print(f"  find_consumers found {len(consumers)} consumers:")
        for cons in consumers:
            print(f"    - {cons.op_type} ({cons.name})")
        break
        
    if current_node.op_type == "LayerNormalization":
        print("  Found LayerNormalization!")
        break
        
    if len(current_node.output) != 1:
        print(f"  ERROR: Node has {len(current_node.output)} outputs, assertion will fail!")
        break
        
    current_tensor = current_node.output[0]
    step += 1
    
    if step > 10:
        print("  Too many steps, stopping...")
        break