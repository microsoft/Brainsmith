"""Quick test to verify unified _load_with_inheritance works correctly."""

import tempfile
import os
from brainsmith.core.blueprint_parser import BlueprintParser


def test_unified_loader():
    """Test that the unified _load_with_inheritance function works correctly."""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create parent blueprint
        parent_yaml = """
version: "4.0"
name: "Parent"
global_config:
  max_combinations: 10000
  timeout_minutes: 30
"""
        
        # Create child blueprint
        child_yaml = """
extends: parent.yaml
name: "Child"
global_config:
  max_combinations: 5000  # Override
"""
        
        # Write files
        parent_path = os.path.join(tmpdir, "parent.yaml")
        child_path = os.path.join(tmpdir, "child.yaml")
        
        with open(parent_path, "w") as f:
            f.write(parent_yaml)
        with open(child_path, "w") as f:
            f.write(child_yaml)
            
        parser = BlueprintParser()
        
        # Test without return_parent
        data = parser._load_with_inheritance(child_path)
        assert isinstance(data, dict)
        assert data["name"] == "Child"
        assert data["global_config"]["max_combinations"] == 5000
        assert data["global_config"]["timeout_minutes"] == 30
        
        # Test with return_parent=False (explicit)
        data2 = parser._load_with_inheritance(child_path, return_parent=False)
        assert data2 == data
        
        # Test with return_parent=True
        merged, parent = parser._load_with_inheritance(child_path, return_parent=True)
        assert isinstance(merged, dict)
        assert isinstance(parent, dict)
        assert merged["name"] == "Child"
        assert parent["name"] == "Parent"
        assert merged["global_config"]["max_combinations"] == 5000
        assert parent["global_config"]["max_combinations"] == 10000
        
        print("✓ All tests passed!")


if __name__ == "__main__":
    test_unified_loader()