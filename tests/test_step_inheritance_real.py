"""Test step inheritance operations with real Brainsmith steps."""

import pytest
import tempfile
import os

from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.plugins.registry import get_registry


@pytest.fixture(autouse=True, scope="module")
def ensure_plugins_loaded():
    """Ensure all plugins are loaded for these tests."""
    # Reset to ensure we have all plugins
    get_registry().reset()


def test_step_operations_after_before():
    """Test after and before operations with real steps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint with real steps
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - cleanup
    - streamlining
    - shell_metadata_handover
"""
        
        # Create child blueprint with operations
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
design_space:
  steps:
    # Insert after cleanup
    - after: "cleanup"
      insert: "cleanup_advanced"
    
    # Insert before shell_metadata_handover
    - before: "shell_metadata_handover"
      insert: "infer_hardware"
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
        
        # Verify operations worked correctly
        assert len(design_space.steps) == 5
        assert design_space.steps[0] == "cleanup"
        assert design_space.steps[1] == "cleanup_advanced"  # Inserted after cleanup
        assert design_space.steps[2] == "streamlining"
        assert design_space.steps[3] == "infer_hardware"  # Inserted before shell_metadata_handover
        assert design_space.steps[4] == "shell_metadata_handover"


def test_step_operations_replace_remove():
    """Test replace and remove operations with real steps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - onnx_preprocessing
    - cleanup
    - ["streamlining", "~"]  # Optional streamlining
    - shell_metadata_handover
"""
        
        # Create child blueprint with operations
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
design_space:
  steps:
    # Remove onnx_preprocessing
    - remove: "onnx_preprocessing"
    
    # Replace optional streamlining with always-on advanced cleanup
    - replace: ["streamlining", "~"]
      with: "cleanup_advanced"
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
        
        # Verify operations worked correctly
        assert len(design_space.steps) == 3
        assert design_space.steps[0] == "cleanup"  # onnx_preprocessing removed
        assert design_space.steps[1] == "cleanup_advanced"  # Replaced branching with single step
        assert design_space.steps[2] == "shell_metadata_handover"


def test_step_operations_at_start_at_end():
    """Test at_start and at_end operations with real steps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - cleanup
    - streamlining
"""
        
        # Create child blueprint with operations
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
design_space:
  steps:
    # Insert at start
    - at_start:
        insert: "onnx_preprocessing"
    
    # Insert at end
    - at_end:
        insert: "shell_metadata_handover"
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
        
        # Verify operations worked correctly
        assert len(design_space.steps) == 4
        assert design_space.steps[0] == "onnx_preprocessing"  # Inserted at start
        assert design_space.steps[1] == "cleanup"
        assert design_space.steps[2] == "streamlining"
        assert design_space.steps[3] == "shell_metadata_handover"  # Inserted at end


def test_step_operations_with_branching():
    """Test operations with branching steps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - cleanup
    - streamlining
"""
        
        # Create child blueprint with operations that add branching
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
design_space:
  steps:
    # Insert optional advanced cleanup after basic cleanup
    - after: "cleanup"
      insert: ["cleanup_advanced", "~"]
    
    # Replace streamlining with optional
    - replace: "streamlining"
      with: ["streamlining", "~"]
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
        
        # Verify operations worked correctly
        assert len(design_space.steps) == 3
        assert design_space.steps[0] == "cleanup"
        assert design_space.steps[1] == ["cleanup_advanced", "~"]  # Branching inserted
        assert design_space.steps[2] == ["streamlining", "~"]  # Replaced with branching
        
        # Check execution tree has correct branching
        from brainsmith.core.execution_tree import count_leaves
        assert count_leaves(execution_tree) == 4  # 2 * 2 = 4 paths


def test_step_operations_without_inheritance():
    """Test operations work in standalone blueprint."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create standalone blueprint with operations
        blueprint_yaml = """
version: "4.0"
name: "Standalone Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    # Define base steps
    - cleanup
    - streamlining
    - shell_metadata_handover
    
    # Apply operations
    - after: "cleanup"
      insert: "fix_dynamic_dimensions"
    
    - before: "shell_metadata_handover"
      insert: ["infer_hardware", "~"]
"""
        
        # Write files
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        model_path = os.path.join(tmpdir, "model.onnx")
        
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        with open(model_path, "wb") as f:
            f.write(b"dummy_model")
            
        # Parse blueprint
        parser = BlueprintParser()
        design_space, execution_tree = parser.parse(blueprint_path, model_path)
        
        # Verify operations worked correctly
        assert len(design_space.steps) == 5
        assert design_space.steps[0] == "cleanup"
        assert design_space.steps[1] == "fix_dynamic_dimensions"  # Inserted after cleanup
        assert design_space.steps[2] == "streamlining"
        assert design_space.steps[3] == ["infer_hardware", "~"]  # Inserted before shell_metadata_handover
        assert design_space.steps[4] == "shell_metadata_handover"


def test_complex_step_operations():
    """Test complex combination of operations with real steps."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent Blueprint"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - onnx_preprocessing
    - cleanup
    - ["streamlining", "~"]
    - specialize_layers
    - shell_metadata_handover
"""
        
        # Create child blueprint with multiple operations
        child_yaml = """
extends: parent.yaml
name: "Child Blueprint"
design_space:
  steps:
    # Remove onnx_preprocessing (we'll add our own)
    - remove: "onnx_preprocessing"
    
    # Add fix_dynamic_dimensions at start
    - at_start:
        insert: "fix_dynamic_dimensions"
    
    # Add advanced cleanup after basic cleanup
    - after: "cleanup"
      insert: "cleanup_advanced"
    
    # Replace optional streamlining with required + quantization prep
    - replace: ["streamlining", "~"]
      with: 
        - "streamlining"
        - "quantization_preprocessing"
    
    # Add hardware inference before specialize_layers
    - before: "specialize_layers"
      insert: "infer_hardware"
    
    # Remove shell_metadata_handover
    - remove: "shell_metadata_handover"
    
    # Add new steps at end
    - at_end:
        insert: ["generate_reference_io", "~"]
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
        
        # Verify operations worked correctly
        expected_steps = [
            "fix_dynamic_dimensions",  # Added at start
            "cleanup",
            "cleanup_advanced",  # Added after cleanup
            "streamlining",  # Replaced optional with required
            "quantization_preprocessing",  # Added after streamlining  
            "infer_hardware",  # Added before specialize_layers
            "specialize_layers",
            ["generate_reference_io", "~"]  # Added at end
        ]
        
        assert len(design_space.steps) == len(expected_steps)
        for i, (actual, expected) in enumerate(zip(design_space.steps, expected_steps)):
            assert actual == expected, f"Step {i}: expected {expected}, got {actual}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])