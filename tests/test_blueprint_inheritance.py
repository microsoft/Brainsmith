# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

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
  steps:
    - cleanup
    - cleanup_advanced
  kernels:
    - MVAU: MVAU_hls
"""
        
        # Create child blueprint that extends parent
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
global_config:
  max_combinations: 5000  # Override parent
  working_directory: "child_work"  # Override parent
  
design_space:
  steps:
    - cleanup  # From parent
    - cleanup_advanced  # From parent
    - [streamlining, ~]  # Add branching step
  kernels:
    - MVAU: MVAU_hls  # Keep parent kernel
    - VVAU: VVAU_hls  # Add new kernel
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
        design_space, execution_tree = parser.parse(child_path, model_path)
        
        # Verify inheritance worked correctly
        
        # Global config: child overrides parent
        assert design_space.global_config.max_combinations == 5000  # From child
        assert design_space.global_config.timeout_minutes == 30     # From parent
        assert design_space.global_config.working_directory == "child_work"  # From child
        
        # Steps: should include parent steps plus child's branching
        assert len(design_space.steps) == 3
        assert design_space.steps[0] == "cleanup"  # From parent
        assert design_space.steps[1] == "cleanup_advanced"  # From parent
        assert design_space.steps[2] == ["streamlining", "~"]  # From child (branching)
        
        # Kernels: should have both parent and child
        kernel_names = [k[0] for k in design_space.kernel_backends]
        assert "MVAU" in kernel_names  # From parent
        assert "VVAU" in kernel_names  # From child
        
        # Execution tree should have branching
        from brainsmith.core.execution_tree import count_leaves
        assert count_leaves(execution_tree) == 2  # Two paths due to branching


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
  steps:
    - cleanup
"""
        
        # Child extends parent
        child_yaml = """
extends: parent.yaml
name: "Child"
global_config:
  timeout_minutes: 30  # Override grandparent
design_space:
  steps:
    - cleanup  # From parent
    - quantization_preprocessing  # Add new
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
        
        # Both steps should be present
        assert data["design_space"]["steps"] == ["cleanup", "quantization_preprocessing"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])