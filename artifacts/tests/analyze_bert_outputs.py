#!/usr/bin/env python3
import onnx
from qonnx.core.modelwrapper import ModelWrapper

# Check the ONNX model structure
model = ModelWrapper('/home/tafk/builds/brainsmith/work/debug_models/00_initial_brevitas.onnx')

print("Original BERT outputs:")
for out in model.graph.output:
    print(f"  {out.name}")
    
print("\nLooking for nodes that produce these outputs:")
for out in model.graph.output:
    for node in model.graph.node:
        if out.name in node.output:
            print(f"  Output '{out.name}' produced by {node.op_type} ({node.name})")
            
# Check cleaned model too
model2 = ModelWrapper('/home/tafk/builds/brainsmith/work/debug_models/02_after_qonnx_cleanup.onnx')

print("\nCleaned model outputs:")
for out in model2.graph.output:
    print(f"  {out.name}")
    producers = model2.find_producer(out.name)
    if producers:
        print(f"    Produced by: {producers.op_type} ({producers.name})")
        
# Look at the last few nodes
print("\nLast 5 nodes in cleaned model:")
for node in model2.graph.node[-5:]:
    print(f"  {node.op_type} ({node.name}) -> {list(node.output)}")