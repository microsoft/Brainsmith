#!/usr/bin/env python3
"""Test the new end-to-end forge functionality."""

import os
import sys
from pathlib import Path

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import forge

def test_forge_minimal():
    """Test forge with minimal example."""
    # Use the test model from segment executor examples
    test_model = Path(__file__).parent.parent.parent / "tests" / "data" / "test_add.onnx"
    
    # Create minimal blueprint
    blueprint_path = Path(__file__).parent / "test_blueprint.yaml"
    blueprint_path.write_text("""
design_space:
  steps:
    - qonnx_to_finn
    - tidy_up
  
finn_config:
  board: Pynq-Z1
  target_fps: 1000
  synth_clk_period_ns: 10.0
  
global_config:
  output_stage: generate_reports
""")
    
    try:
        # Test forge - should create execution tree and explore it
        print(f"Testing forge with test model: {test_model}")
        print(f"Blueprint: {blueprint_path}")
        
        if test_model.exists():
            results = forge(
                model_path=str(test_model),
                blueprint_path=str(blueprint_path)
            )
            
            print(f"\nForge completed!")
            print(f"Results: {results.stats}")
            print(f"Total time: {results.total_time:.2f}s")
            
            # Check we have results
            assert results.stats['total'] > 0
            print("\n✅ Test passed!")
        else:
            print(f"Test model not found at {test_model}")
            print("Creating dummy test...")
            
            # Create dummy ONNX model
            import onnx
            import numpy as np
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
            test_model = Path(__file__).parent / "test_add.onnx"
            onnx.save(model, str(test_model))
            
            # Now test
            results = forge(
                model_path=str(test_model),
                blueprint_path=str(blueprint_path)
            )
            
            print(f"\nForge completed!")
            print(f"Results: {results.stats}")
            print(f"Total time: {results.total_time:.2f}s")
            
            # Cleanup
            test_model.unlink()
            
    finally:
        # Cleanup
        if blueprint_path.exists():
            blueprint_path.unlink()


if __name__ == "__main__":
    test_forge_minimal()