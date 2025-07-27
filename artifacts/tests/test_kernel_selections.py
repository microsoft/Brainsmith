#!/usr/bin/env python3
"""Test that kernel selections are properly passed through the system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.forge import forge
from brainsmith.core.explorer import explore_execution_tree

def test_kernel_selections():
    print("=== Testing Kernel Selections Flow ===\n")
    
    # Parse blueprint
    blueprint_path = Path(__file__).parent.parent.parent / "brainsmith/blueprints/bert.yaml"
    print(f"Loading blueprint: {blueprint_path}")
    
    # Create a dummy model file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        dummy_model = f.name
        f.write(b"dummy")
    
    # Forge the design space
    design_space, tree = forge(
        model_path=dummy_model,
        blueprint_path=str(blueprint_path)
    )
    
    print(f"\nDesign space kernels: {len(design_space.kernel_backends)}")
    for kernel_name, backends in design_space.kernel_backends:
        print(f"  - {kernel_name}: {[b.__name__ for b in backends]}")
    
    # Create a minimal blueprint config
    blueprint_config = {
        'global_config': {},
        'finn_config': {
            'board': 'V80',
            'synth_clk_period_ns': 3.33
        }
    }
    
    # Test that kernel selections are added
    from brainsmith.core.explorer.explorer import explore_execution_tree
    
    # Patch explore_execution_tree to check the config
    original_explore = explore_execution_tree
    
    def patched_explore(tree, model_path, output_dir, blueprint_config, design_space=None):
        print(f"\nIn explore_execution_tree:")
        print(f"  design_space provided: {design_space is not None}")
        if design_space:
            print(f"  design_space.kernel_backends: {len(design_space.kernel_backends)}")
        
        # Check if kernel_selections is added to finn_config
        finn_config = blueprint_config.get("finn_config", {})
        kernel_selections = finn_config.get("kernel_selections", None)
        print(f"  kernel_selections in finn_config: {kernel_selections}")
        
        return None  # Don't actually execute
    
    # Temporarily replace the function
    import brainsmith.core.explorer
    brainsmith.core.explorer.explore_execution_tree = patched_explore
    
    # Call it with design_space
    explore_execution_tree(
        tree=tree,
        model_path="dummy.onnx",
        output_dir="/tmp/test",
        blueprint_config=blueprint_config,
        design_space=design_space
    )

if __name__ == "__main__":
    test_kernel_selections()