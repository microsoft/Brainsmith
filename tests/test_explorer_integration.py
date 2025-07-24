"""Integration tests for explorer module - testing tree structure without mocks."""

import tempfile
import json
from pathlib import Path
from brainsmith.core.execution_tree import ExecutionNode
from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.explorer import explore_execution_tree
from brainsmith.core.plugins.registry import get_registry


def test_tree_serialization():
    """Test that execution tree is properly serialized to JSON."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        
        # Create simple tree
        root = ExecutionNode(segment_steps=[{"name": "step1"}])
        child = root.add_child("branch1", [{"name": "step2"}])
        
        # Create minimal blueprint config
        blueprint_config = {
            "global_config": {"fail_fast": False},
            "finn_config": {
                "synth_clk_period_ns": 5.0,
                "board": "Pynq-Z1"
            }
        }
        
        # We can't test full execution without FINN, but we can test tree serialization
        from brainsmith.core.explorer.utils import serialize_tree
        
        output_dir.mkdir(parents=True, exist_ok=True)
        tree_json = output_dir / "tree.json"
        tree_json.write_text(serialize_tree(root))
        
        # Verify tree structure
        tree_data = json.loads(tree_json.read_text())
        assert tree_data["segment_id"] == "root"
        assert tree_data["segment_steps"] == [{"name": "step1"}]
        assert "branch1" in tree_data["children"]
        assert tree_data["children"]["branch1"]["segment_steps"] == [{"name": "step2"}]


def test_branching_tree_structure():
    """Test serialization of branching tree."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create branching tree
        root = ExecutionNode(segment_steps=[{"name": "step1"}])
        branch1 = root.add_child("option1", [{"name": "step2a"}])
        branch2 = root.add_child("option2", [{"name": "step2b"}])
        
        # Add grandchildren
        leaf1 = branch1.add_child("leaf1", [{"name": "step3"}])
        leaf2 = branch2.add_child("leaf2", [{"name": "step3"}])
        
        from brainsmith.core.explorer.utils import serialize_tree
        tree_json = output_dir / "tree.json"
        tree_json.write_text(serialize_tree(root))
        
        # Verify structure
        tree_data = json.loads(tree_json.read_text())
        assert len(tree_data["children"]) == 2
        assert "option1" in tree_data["children"]
        assert "option2" in tree_data["children"]
        
        # Check grandchildren
        assert "leaf1" in tree_data["children"]["option1"]["children"]
        assert "leaf2" in tree_data["children"]["option2"]["children"]


def test_blueprint_parser_with_real_transforms():
    """Test blueprint parser with real transforms and tree building."""
    registry = get_registry()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test model
        model_path = Path(tmpdir) / "model.onnx"
        model_path.write_bytes(b"dummy_model")
        
        # Create simple blueprint
        blueprint_path = Path(tmpdir) / "test.yaml"
        blueprint_path.write_text("""
version: "4.0"
name: "Test Blueprint"

global_config:
  output_stage: generate_reports
  fail_fast: false

finn_config:
  synth_clk_period_ns: 5.0
  board: Pynq-Z1

design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
      - ["~", RemoveUnusedTensors]
    
    optimize:
      - FoldConstants
      - InferShapes
  
  kernels: []

build_pipeline:
  steps:
    - "step_qonnx_to_finn"
    - "{cleanup}"
    - "{optimize}"
    - "step_create_dataflow_partition"
""")
        
        parser = BlueprintParser()
        design_space, tree = parser.parse(str(blueprint_path), str(model_path))
        
        # Verify design space
        assert "cleanup" in design_space.transform_stages
        assert "optimize" in design_space.transform_stages
        
        # Verify tree structure
        assert tree is not None
        assert tree.segment_id == "root"
        
        # The tree should have branches for optional RemoveUnusedTensors
        from brainsmith.core.execution_tree import count_leaves
        assert count_leaves(tree) == 2  # Two paths: with and without RemoveUnusedTensors


def test_transform_stage_wrapper_integration():
    """Test that transform stages are properly wrapped in the tree."""
    registry = get_registry()
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = Path(tmpdir) / "model.onnx"
        model_path.write_bytes(b"dummy_model")
        
        blueprint_path = Path(tmpdir) / "test.yaml"
        blueprint_path.write_text("""
version: "4.0"
name: "Wrapper Test"

global_config:
  output_stage: generate_reports

finn_config:
  synth_clk_period_ns: 5.0
  board: Pynq-Z1

design_space:
  transforms:
    multi_option:
      - [RemoveIdentityOps, RemoveUnusedTensors, GiveUniqueNodeNames]
  
  kernels: []

build_pipeline:
  steps:
    - "{multi_option}"
""")
        
        parser = BlueprintParser()
        design_space, tree = parser.parse(str(blueprint_path), str(model_path))
        
        # Should create 3 branches for the 3 transform options
        assert len(tree.children) == 3
        
        # Each child should have a finn_step_name
        for branch_name, child in tree.children.items():
            assert len(child.segment_steps) > 0
            step = child.segment_steps[0]
            assert "finn_step_name" in step
            # Should be simple numeric indices
            assert step["finn_step_name"] in ["multi_option_0", "multi_option_1", "multi_option_2"]


def test_complex_tree_stats():
    """Test tree statistics calculation with complex tree."""
    from brainsmith.core.execution_tree import get_tree_stats
    
    # Build a tree manually
    root = ExecutionNode(segment_steps=[{"name": "step1"}])
    
    # First level branches
    a1 = root.add_child("a1", [{"name": "transform_a1"}])
    a2 = root.add_child("a2", [{"name": "transform_a2"}])
    
    # Second level branches
    b1 = a1.add_child("b1", [{"name": "transform_b1"}])
    b2 = a1.add_child("b2", [{"name": "transform_b2"}])
    b3 = a2.add_child("b3", [{"name": "transform_b3"}])
    
    # Get stats
    stats = get_tree_stats(root)
    
    assert stats["total_paths"] == 3  # Three leaf nodes
    assert stats["total_segments"] == 5  # a1, a2, b1, b2, b3
    assert stats["max_depth"] == 2    # root -> a -> b
    assert stats["total_steps"] == 6  # 1 + 2 + 3 steps