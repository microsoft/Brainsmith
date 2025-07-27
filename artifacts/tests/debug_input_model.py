#!/usr/bin/env python3
import onnx

model = onnx.load('/tmp/bert_test_run/root/input.onnx')
print('Input graph node count:', len(model.graph.node))
print('First 15 nodes:')
for i, node in enumerate(model.graph.node[:15]):
    print(f'  {i}: {node.op_type} - inputs: {len(node.input)} - outputs: {len(node.output)} - output names: {list(node.output)}')

print('\nGraph inputs:')
for inp in model.graph.input:
    print(f'  {inp.name}')

print('\nLooking for LayerNormalization nodes:')
for i, node in enumerate(model.graph.node):
    if node.op_type == "LayerNormalization":
        print(f'  Node {i}: {node.name} - inputs: {list(node.input)} - outputs: {list(node.output)}')