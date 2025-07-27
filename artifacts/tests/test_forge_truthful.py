#!/usr/bin/env python3
"""More truthful tests for forge functionality."""

import os
import sys
from pathlib import Path
import json

# Add parent to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from brainsmith import forge

def test_forge_with_real_verification():
    """Test forge with proper verification of outputs."""
    # Create dummy ONNX model
    import onnx
    from onnx import helper, TensorProto
    
    # Create simple Add node
    add = helper.make_node('Add', ['X', 'Y'], ['Z'])
    graph = helper.make_graph(
        [add],
        'test_model',
        [helper.make_tensor_value_info('X', TensorProto.FLOAT, [1, 3]),
         helper.make_tensor_value_info('Y', TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info('Z', TensorProto.FLOAT, [1, 3])]
    )
    model = helper.make_model(graph)
    test_model = Path(__file__).parent / "test_real.onnx"
    onnx.save(model, str(test_model))
    
    # Create blueprint with branches
    blueprint_path = Path(__file__).parent / "test_blueprint_real.yaml"
    blueprint_path.write_text("""
design_space:
  steps:
    - qonnx_to_finn
    - [tidy_up, ~]  # Branch: tidy_up or skip
    - streamline
  
finn_config:
  board: Pynq-Z1
  synth_clk_period_ns: 10.0
  save_intermediate_models: true
  
global_config:
  output_stage: generate_reports
  save_intermediate_models: true
""")
    
    output_dir = Path(__file__).parent / "test_output"
    
    try:
        # Run forge
        print("Testing forge with real verification...")
        results = forge(
            model_path=str(test_model),
            blueprint_path=str(blueprint_path),
            output_dir=str(output_dir)
        )
        
        # Verify basic results
        assert results.stats['total'] == 3, f"Expected 3 segments, got {results.stats['total']}"
        assert results.stats['successful'] == 3, f"Expected 3 successful, got {results.stats['successful']}"
        
        # Verify tree structure
        tree = results.execution_tree
        assert len(tree.children) == 2, "Root should have 2 children"
        assert "tidy_up" in tree.children, "Should have tidy_up branch"
        assert "skip_1" in tree.children, "Should have skip branch"
        
        # Verify output files exist
        for segment_id, result in results.segment_results.items():
            if result.success:
                assert result.output_model is not None, f"Segment {segment_id} should have output model"
                assert result.output_model.exists(), f"Output model for {segment_id} should exist"
                
                # Check intermediate models if saved
                if results.design_space.config.save_intermediate_models:
                    intermediate_dir = result.output_dir / "intermediate_models"
                    if intermediate_dir.exists():
                        models = list(intermediate_dir.glob("*.onnx"))
                        print(f"  {segment_id}: {len(models)} intermediate models")
        
        # Verify tree.json was saved
        tree_json = output_dir / "tree.json"
        assert tree_json.exists(), "tree.json should be saved"
        tree_data = json.loads(tree_json.read_text())
        assert "segment_id" in tree_data, "tree.json should have valid structure"
        
        # Verify summary.json
        summary_json = output_dir / "summary.json"
        assert summary_json.exists(), "summary.json should be saved"
        summary_data = json.loads(summary_json.read_text())
        assert summary_data["stats"]["total"] == 3, "Summary should match results"
        
        # Verify artifact sharing
        root_output = output_dir / "root" / "root_output.onnx"
        tidy_input = output_dir / "tidy_up" / "root_output.onnx"
        skip_input = output_dir / "skip_1" / "root_output.onnx"
        
        assert root_output.exists(), "Root should produce output"
        assert tidy_input.exists(), "Tidy_up should receive root output"
        assert skip_input.exists(), "Skip should receive root output"
        
        # Verify they're the same size (artifact was copied)
        assert root_output.stat().st_size == tidy_input.stat().st_size
        assert root_output.stat().st_size == skip_input.stat().st_size
        
        print("\n✅ All verifications passed!")
        print(f"   - Tree structure correct")
        print(f"   - Output models exist for all segments")
        print(f"   - Artifacts properly shared between branches")
        print(f"   - JSON outputs saved correctly")
        
    finally:
        # Cleanup
        if test_model.exists():
            test_model.unlink()
        if blueprint_path.exists():
            blueprint_path.unlink()
        if output_dir.exists():
            import shutil
            shutil.rmtree(output_dir)


def test_forge_failure_handling():
    """Test that forge handles failures properly."""
    # Create invalid blueprint
    blueprint_path = Path(__file__).parent / "test_blueprint_bad.yaml"
    blueprint_path.write_text("""
design_space:
  steps:
    - nonexistent_step  # This step doesn't exist
  
finn_config:
  board: Pynq-Z1
""")
    
    try:
        from brainsmith.core.blueprint_parser import BlueprintParser
        parser = BlueprintParser()
        
        # This should raise an error
        try:
            design_space, tree = parser.parse(str(blueprint_path), "dummy.onnx")
            assert False, "Should have raised error for nonexistent step"
        except ValueError as e:
            assert "not found in registry" in str(e)
            print("✅ Correctly rejected nonexistent step")
            
    finally:
        if blueprint_path.exists():
            blueprint_path.unlink()


if __name__ == "__main__":
    test_forge_with_real_verification()
    test_forge_failure_handling()