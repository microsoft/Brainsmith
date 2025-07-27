#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper

# Load the cleaned model
model = ModelWrapper('/home/tafk/builds/brainsmith/work/debug_models/02_after_qonnx_cleanup.onnx')

print("Tracing from input to find branching points...")

# Start from input
current = model.graph.input[0].name
visited = set()
queue = [(current, 0)]

while queue:
    tensor_name, depth = queue.pop(0)
    if tensor_name in visited:
        continue
    visited.add(tensor_name)
    
    # Find consumers
    consumers = model.find_consumers(tensor_name)
    indent = "  " * depth
    
    if len(consumers) > 1:
        print(f"{indent}BRANCH at '{tensor_name}' - {len(consumers)} consumers:")
        for i, cons in enumerate(consumers):
            print(f"{indent}  [{i}] {cons.op_type} ({cons.name}) -> {cons.output}")
            # Add outputs to queue
            for out in cons.output:
                queue.append((out, depth + 1))
    elif len(consumers) == 1:
        cons = consumers[0]
        print(f"{indent}{cons.op_type} ({cons.name}) -> {cons.output}")
        # Add outputs to queue
        for out in cons.output:
            queue.append((out, depth + 1))
            
        # Stop at first LayerNorm
        if cons.op_type == "LayerNormalization" and depth < 5:
            print(f"{indent}  ^^ First LayerNormalization found!")
            break