#!/usr/bin/env python3
from qonnx.core.modelwrapper import ModelWrapper

# Check what the model looks like at the point where infer_hardware fails
model = ModelWrapper('/home/tafk/builds/brainsmith/work/bert_streamlining/input.onnx')

print("Model after bert_streamlining:")
print(f"Total nodes: {len(model.graph.node)}")

# Look for ElementwiseMul nodes
elementwise_mul_nodes = []
for i, node in enumerate(model.graph.node):
    if node.op_type == "ElementwiseMul":
        elementwise_mul_nodes.append((i, node))
        
print(f"\nFound {len(elementwise_mul_nodes)} ElementwiseMul nodes")
if elementwise_mul_nodes:
    for idx, node in elementwise_mul_nodes[:3]:  # Show first 3
        print(f"  Node {idx}: {node.name}")
        print(f"    Domain: {node.domain}")
        
# Check all unique op types
op_types = set()
op_domains = {}
for node in model.graph.node:
    op_types.add(node.op_type)
    if node.domain:
        op_domains[node.op_type] = node.domain
        
print(f"\nUnique op types ({len(op_types)}):")
for op in sorted(op_types):
    domain = op_domains.get(op, "")
    print(f"  {op}{' (' + domain + ')' if domain else ''}")