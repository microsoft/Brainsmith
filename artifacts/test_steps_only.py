#!/usr/bin/env python3
"""Test that step-only execution works correctly after cleanup."""

import json
from pathlib import Path
from brainsmith.core.execution_tree import ExecutionNode
from brainsmith.core.explorer.utils import serialize_tree

# Create a simple execution tree with only steps
root = ExecutionNode(segment_steps=[], finn_config={})

# Add steps - all using the clean 'name' field format
child1 = root.add_child("cleanup", [{"name": "cleanup"}])
child2 = root.add_child("streamline", [{"name": "streamline"}])

# Add more steps to child branches
child1.add_child("quantize", [{"name": "quantize_int8"}])
child2.add_child("optimize", [{"name": "optimize_aggressive"}])

# Serialize the tree to verify clean output
tree_json = serialize_tree(root)
tree_data = json.loads(tree_json)

print("Serialized tree structure:")
print(json.dumps(tree_data, indent=2))

# Verify no transform handling artifacts remain
def check_node(node_data):
    """Recursively check that nodes only have step names."""
    for step in node_data.get("segment_steps", []):
        assert isinstance(step, dict), f"Step should be dict: {step}"
        assert "name" in step, f"Step missing 'name': {step}"
        assert "transforms" not in step, f"Step should not have 'transforms': {step}"
        assert "finn_step_name" not in step, f"Step should not have 'finn_step_name': {step}"
    
    for child in node_data.get("children", {}).values():
        check_node(child)

check_node(tree_data)
print("\n✅ All steps are clean - no transform wrapping artifacts found!")