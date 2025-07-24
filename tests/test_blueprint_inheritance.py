"""Test blueprint inheritance feature."""

import pytest
import tempfile
import os

from brainsmith.core.blueprint_parser import BlueprintParser


def test_blueprint_inheritance():
    """Test that blueprint inheritance works correctly."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
global_config:
  output_stage: "generate_reports"
  max_combinations: 10000
  timeout_minutes: 30
  working_directory: "parent_work"
  
design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - RemoveUnusedTensors
  kernels:
    - MVAU: MVAU_hls
    
build_pipeline:
  steps:
    - "{cleanup}"
"""
        
        # Create child blueprint that extends parent
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
global_config:
  max_combinations: 5000  # Override parent
  working_directory: "child_work"  # Override parent
  
design_space:
  transforms:
    optimize:  # Add new stage
      - FoldConstants
      - [InferDataLayouts, ~]  # Options
  kernels:
    - MVAU: MVAU_hls  # Keep parent kernel
    - VVAU: VVAU_hls  # Add new kernel
    
build_pipeline:
  steps:
    - "{cleanup}"  # From parent
    - "{optimize}"  # New in child
"""
        
        # Write files
        parent_path = os.path.join(tmpdir, "parent.yaml")
        child_path = os.path.join(tmpdir, "child.yaml")
        model_path = os.path.join(tmpdir, "model.onnx")
        
        with open(parent_path, "w") as f:
            f.write(parent_yaml)
        with open(child_path, "w") as f:
            f.write(child_yaml)
        with open(model_path, "wb") as f:
            f.write(b"dummy_model")
            
        # Parse with inheritance
        parser = BlueprintParser()
        design_space, _ = parser.parse(child_path, model_path)
        
        # Verify inheritance worked correctly
        
        # Global config: child overrides parent
        assert design_space.global_config.max_combinations == 5000  # From child
        assert design_space.global_config.timeout_minutes == 30     # From parent
        assert design_space.global_config.working_directory == "child_work"  # From child
        
        # Transforms: merged from both
        assert "cleanup" in design_space.transform_stages  # From parent
        assert "optimize" in design_space.transform_stages  # From child
        
        # Check cleanup stage has parent's transforms
        cleanup_stage = design_space.transform_stages["cleanup"]
        assert len(cleanup_stage.transform_steps) == 2
        
        # Check optimize stage has child's transforms
        optimize_stage = design_space.transform_stages["optimize"]
        assert len(optimize_stage.transform_steps) == 2
        
        # Kernels: should have both parent and child
        kernel_names = [k[0] for k in design_space.kernel_backends]
        assert "MVAU" in kernel_names  # From parent
        assert "VVAU" in kernel_names  # From child
        
        # Pipeline: should have both stages
        assert design_space.build_pipeline == ["{cleanup}", "{optimize}"]


def test_deep_inheritance_chain():
    """Test inheritance chain: grandparent -> parent -> child."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Grandparent
        grandparent_yaml = """
version: "4.0"
name: "Grandparent"
global_config:
  max_combinations: 20000
  timeout_minutes: 60
  save_intermediate_models: false
"""
        
        # Parent extends grandparent
        parent_yaml = """
extends: grandparent.yaml
name: "Parent"
global_config:
  max_combinations: 10000  # Override
  save_intermediate_models: true  # Override
design_space:
  transforms:
    stage1:
      - Transform1
"""
        
        # Child extends parent
        child_yaml = """
extends: parent.yaml
name: "Child"
global_config:
  timeout_minutes: 30  # Override grandparent
design_space:
  transforms:
    stage2:
      - Transform2
"""
        
        # Write files
        with open(os.path.join(tmpdir, "grandparent.yaml"), "w") as f:
            f.write(grandparent_yaml)
        with open(os.path.join(tmpdir, "parent.yaml"), "w") as f:
            f.write(parent_yaml)
        child_path = os.path.join(tmpdir, "child.yaml")
        with open(child_path, "w") as f:
            f.write(child_yaml)
            
        # Test loading with deep inheritance
        parser = BlueprintParser()
        data = parser._load_with_inheritance(child_path)
        
        # Verify deep merge worked
        assert data["name"] == "Child"
        assert data["global_config"]["max_combinations"] == 10000  # From parent
        assert data["global_config"]["timeout_minutes"] == 30      # From child
        assert data["global_config"]["save_intermediate_models"] == True  # From parent
        
        # Both transform stages should be present
        assert "stage1" in data["design_space"]["transforms"]  # From parent
        assert "stage2" in data["design_space"]["transforms"]  # From child


if __name__ == "__main__":
    pytest.main([__file__, "-v"])