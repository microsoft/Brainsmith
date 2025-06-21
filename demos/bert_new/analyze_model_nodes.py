#!/usr/bin/env python3
"""Analyze node names and types in the BERT model."""

import sys
sys.path.append('/home/tafk/dev/brainsmith-1/deps/finn/src')
sys.path.append('/home/tafk/dev/brainsmith-1/deps/qonnx/src')
import json
from qonnx.core.modelwrapper import ModelWrapper
from finn.util.fpgadataflow import is_fpgadataflow_node

# Load the model after folding config is applied
model = ModelWrapper('finn_output/intermediate_models/step_apply_folding_config.onnx')

# Count nodes by type
node_counts = {}
node_list = []

for node in model.graph.node:
    if is_fpgadataflow_node(node):
        node_type = node.op_type
        if node_type not in node_counts:
            node_counts[node_type] = 0
        node_counts[node_type] += 1
        node_list.append({
            'name': node.name,
            'op_type': node.op_type,
            'index': node_counts[node_type] - 1
        })

print("FPGA Dataflow Nodes in Model:")
print("=" * 50)
for node_type, count in sorted(node_counts.items()):
    print(f"{node_type}: {count} nodes")

print("\n\nNode List:")
print("=" * 50)
for node in node_list:
    print(f"{node['name']} ({node['op_type']})")

# Load folding config
with open('l1_simd12_pe8.json', 'r') as f:
    folding_config = json.load(f)

print("\n\nFolding Config Nodes:")
print("=" * 50)
for node_name in sorted(folding_config.keys()):
    if node_name != "Defaults":
        print(f"{node_name}: {folding_config[node_name]}")

# Check for mismatches
print("\n\nChecking for node name mismatches:")
print("=" * 50)
model_node_names = {node['name'] for node in node_list}
config_node_names = set(folding_config.keys()) - {"Defaults"}

missing_in_model = config_node_names - model_node_names
missing_in_config = model_node_names - config_node_names

if missing_in_model:
    print(f"\nNodes in config but not in model: {missing_in_model}")
if missing_in_config:
    print(f"\nNodes in model but not in config: {missing_in_config}")

if not missing_in_model and not missing_in_config:
    print("\nAll node names match between model and config!")