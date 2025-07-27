#!/usr/bin/env python3
"""Test forge with branching execution tree."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import forge

def test_forge_with_branches():
    """Test forge with branching blueprint."""
    # Create dummy ONNX model
    import onnx
    from onnx import helper, TensorProto
    
    # Create simple Add node
    add = helper.make_node('Add', ['X', 'Y'], ['Z'])
    
    # Create graph
    graph = helper.make_graph(
        [add],
        'test_model',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 3])]
    )
    
    # Create model
    model = helper.make_model(graph)
    test_model = Path(__file__).parent / "test_add_branches.onnx"
    onnx.save(model, str(test_model))
    
    # Create blueprint with branches
    blueprint_path = Path(__file__).parent / "test_blueprint_branches.yaml"
    blueprint_path.write_text("""
design_space:
  steps:
    - qonnx_to_finn
    - [tidy_up, ~]  # Branch: tidy_up or skip
    - streamline
  
finn_config:
  board: Pynq-Z1
  target_fps: 1000
  synth_clk_period_ns: 10.0
  
global_config:
  output_stage: generate_reports
""")
    
    try:
        # Test forge
        print(f"Testing forge with branching blueprint...")
        
        results = forge(
            model_path=str(test_model),
            blueprint_path=str(blueprint_path)
        )
        
        print(f"\nForge completed!")
        print(f"Results: {results.stats}")
        print(f"Total time: {results.total_time:.2f}s")
        
        # Debug: print tree structure
        from brainsmith.core.execution_tree import print_tree
        print("\nExecution tree structure:")
        print_tree(results.execution_tree)
        
        # Check we have multiple segments
        assert results.stats['total'] >= 1, "Should have at least one segment"
        assert results.stats['successful'] > 0, "Should have successful builds"
        
        # Check tree structure was created
        assert hasattr(results, 'execution_tree'), "Results should have execution tree"
        assert hasattr(results, 'design_space'), "Results should have design space"
        
        print(f"\nExecution tree has {results.stats['total']} segments")
        print(f"Design space steps: {results.design_space.steps}")
        print(f"Design space has {len(results.design_space.steps)} steps")
        
        print("\n✅ Test passed!")
        
    finally:
        # Cleanup
        if test_model.exists():
            test_model.unlink()
        if blueprint_path.exists():
            blueprint_path.unlink()


if __name__ == "__main__":
    test_forge_with_branches()