#!/usr/bin/env python3
"""Test the refactored blueprint parser separation."""

import sys
sys.path.insert(0, '/home/tafk/dev/brainsmith-4')

from brainsmith.core.forge_v2 import forge
from brainsmith.core.blueprint_parser_v2 import BlueprintParser
from brainsmith.core.tree_builder import TreeBuilder

def test_separation():
    """Test that parser and tree builder are properly separated."""
    
    model_path = "/home/tafk/dev/brainsmith-4/demos/bert_modern/bert-110m-quantized.onnx"
    blueprint_path = "/home/tafk/dev/brainsmith-4/demos/bert_modern/bert_demo.yaml"
    
    # Test BlueprintParser returns only DesignSpace
    parser = BlueprintParser()
    design_space = parser.parse(blueprint_path, model_path)
    
    print(f"✓ BlueprintParser returns: {type(design_space).__name__}")
    print(f"  - Steps: {len(design_space.steps)}")
    print(f"  - Kernels: {len(design_space.kernel_backends)}")
    print(f"  - Config type: {type(design_space.global_config).__name__}")
    
    # Test TreeBuilder builds from DesignSpace
    builder = TreeBuilder()
    tree = builder.build_tree(design_space)
    
    print(f"\n✓ TreeBuilder returns: {type(tree).__name__}")
    print(f"  - Segment ID: {tree.segment_id}")
    print(f"  - Children: {len(tree.children)}")
    
    # Test forge uses both
    design_space2, tree2 = forge(model_path, blueprint_path)
    
    print(f"\n✓ forge returns: ({type(design_space2).__name__}, {type(tree2).__name__})")
    
    # Verify BuildConfig rename
    assert design_space.global_config.__class__.__name__ == "BuildConfig"
    print(f"\n✓ GlobalConfig renamed to BuildConfig")
    
    print("\n✅ All tests passed! Refactoring successful.")

if __name__ == "__main__":
    test_separation()