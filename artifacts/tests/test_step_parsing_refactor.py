"""Test refactored _parse_steps method with various scenarios."""

import tempfile
import os
from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.plugins import get_registry, list_steps


def test_step_parsing_scenarios():
    """Test various step parsing scenarios after refactoring."""
    
    # Ensure plugins are loaded
    get_registry().reset()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        with open(model_path, "wb") as f:
            f.write(b"dummy_model")
        
        parser = BlueprintParser()
        
        # Test 1: Direct steps only (no parent, no operations)
        print("Test 1: Direct steps only")
        yaml1 = """
version: "4.0"
name: "Test1"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - finn:tidy_up
    - [finn:streamline, ~]
    - infer_kernels
"""
        blueprint_path = os.path.join(tmpdir, "test1.yaml")
        with open(blueprint_path, "w") as f:
            f.write(yaml1)
        
        design_space, _ = parser.parse(blueprint_path, model_path)
        assert len(design_space.steps) == 3
        assert design_space.steps[0] == "finn:tidy_up"
        assert design_space.steps[1] == ["finn:streamline", "~"]
        assert design_space.steps[2] == "infer_kernels"
        print("  ✓ Direct steps parsed correctly")
        
        # Test 2: Operations with parent inheritance
        print("\nTest 2: Operations with parent inheritance")
        parent_yaml = """
version: "4.0"
name: "Parent"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - finn:tidy_up
    - finn:streamline
    - infer_kernels
"""
        child_yaml = """
extends: parent.yaml
name: "Child"
design_space:
  steps:
    - after: finn:tidy_up
      insert: bert_cleanup
    - before: infer_kernels
      insert: [finn:minimize_bit_width, ~]
"""
        parent_path = os.path.join(tmpdir, "parent.yaml")
        child_path = os.path.join(tmpdir, "child.yaml")
        with open(parent_path, "w") as f:
            f.write(parent_yaml)
        with open(child_path, "w") as f:
            f.write(child_yaml)
        
        design_space, _ = parser.parse(child_path, model_path)
        expected = ["finn:tidy_up", "bert_cleanup", "finn:streamline", ["finn:minimize_bit_width", "~"], "infer_kernels"]
        assert design_space.steps == expected
        print("  ✓ Operations applied correctly to parent steps")
        
        # Test 3: Direct steps replace parent
        print("\nTest 3: Direct steps replace parent")
        child2_yaml = """
extends: parent.yaml
name: "Child2"
design_space:
  steps:
    - bert_cleanup
    - bert_streamlining
"""
        child2_path = os.path.join(tmpdir, "child2.yaml")
        with open(child2_path, "w") as f:
            f.write(child2_yaml)
        
        design_space, _ = parser.parse(child2_path, model_path)
        assert len(design_space.steps) == 2
        assert design_space.steps == ["bert_cleanup", "bert_streamlining"]
        print("  ✓ Direct steps correctly replace parent steps")
        
        # Test 4: Mix of direct steps and operations
        print("\nTest 4: Mix of direct steps and operations")
        child3_yaml = """
extends: parent.yaml
name: "Child3"
design_space:
  steps:
    - finn:tidy_up
    - after: finn:tidy_up
      insert: bert_cleanup
    - at_end:
        insert: shell_metadata_handover
"""
        child3_path = os.path.join(tmpdir, "child3.yaml")
        with open(child3_path, "w") as f:
            f.write(child3_yaml)
        
        design_space, _ = parser.parse(child3_path, model_path)
        assert design_space.steps == ["finn:tidy_up", "bert_cleanup", "shell_metadata_handover"]
        print("  ✓ Mix of direct steps and operations handled correctly")
        
        # Test 5: Complex operations (at_start, at_end, remove, replace)
        print("\nTest 5: Complex operations")
        complex_yaml = """
version: "4.0"
name: "Complex"
finn_config:
  synth_clk_period_ns: 5.0
  board: "ZCU104"
design_space:
  steps:
    - finn:tidy_up
    - finn:streamline
    - finn:minimize_bit_width
    - infer_kernels
    - at_start:
        insert: qonnx_to_finn
    - at_end:
        insert: shell_metadata_handover
    - remove: finn:streamline
    - replace: finn:minimize_bit_width
      with: [bert_cleanup, bert_streamlining]
"""
        complex_path = os.path.join(tmpdir, "complex.yaml")
        with open(complex_path, "w") as f:
            f.write(complex_yaml)
        
        design_space, _ = parser.parse(complex_path, model_path)
        # Note: replace with a list of strings (no ~) extends the steps rather than creating a branch
        expected = ["qonnx_to_finn", "finn:tidy_up", "bert_cleanup", "bert_streamlining", "infer_kernels", "shell_metadata_handover"]
        assert design_space.steps == expected
        print("  ✓ Complex operations handled correctly")
        
        # Test 6: skip_operations parameter
        print("\nTest 6: skip_operations parameter (internal)")
        steps_data = [
            "finn:tidy_up",
            {"after": "finn:tidy_up", "insert": "should_be_ignored"},
            ["finn:streamline", "~"],
            {"remove": "finn:tidy_up"}
        ]
        result = parser._parse_steps(steps_data, skip_operations=True)
        assert len(result) == 2  # Only non-dict items
        assert result == ["finn:tidy_up", ["finn:streamline", "~"]]
        print("  ✓ skip_operations correctly ignores operation dicts")
        
        print("\n✓ All step parsing tests passed!")


if __name__ == "__main__":
    test_step_parsing_scenarios()