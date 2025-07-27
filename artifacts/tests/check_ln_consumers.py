#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper

model = ModelWrapper('/home/tafk/builds/brainsmith/work/bert_cleanup/input.onnx')

# Find the first LayerNorm
for node in model.graph.node:
    if node.op_type == "LayerNormalization":
        print(f"First LayerNorm: {node.name}")
        print(f"  Output: {node.output[0]}")
        
        # Find its consumers
        consumers = model.find_consumers(node.output[0])
        print(f"  Has {len(consumers)} consumers:")
        for cons in consumers:
            print(f"    - {cons.op_type} ({cons.name}) -> {cons.output}")
            
        # The issue is likely that the embedding output feeds into both
        # the attention blocks AND somewhere else (maybe pooler?)
        break