"""
End-to-End Test: Blueprint YAML to Execution Tree

This test verifies the complete pipeline from parsing a blueprint YAML file
to generating an optimized execution tree with automatic prefix sharing.
"""

import pytest
import tempfile
import os
from pathlib import Path

from brainsmith.core.forge_v2 import forge_tree
from brainsmith.core.execution_tree import count_leaves, count_nodes, get_tree_stats
from brainsmith.core.plugins.registry import get_registry


def create_test_model(path: str):
    """Create a minimal ONNX model file for testing."""
    # For testing, we just need the file to exist
    with open(path, "wb") as f:
        f.write(b"dummy_onnx_model")


def test_minimal_blueprint_to_tree():
    """Test the simplest possible blueprint creates a valid tree."""
    blueprint_yaml = """
version: "4.0"
name: "Minimal Test"

design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {cleanup}
    - infer_kernels
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create files
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        # Parse blueprint to tree
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Verify structure
        assert tree.step_name == "root"
        assert tree.config["model"] == model_path
        
        # Should have: root -> cleanup -> infer_kernels
        assert count_nodes(tree) == 2
        assert count_leaves(tree) == 1
        
        # Check transforms were resolved
        cleanup_node = tree.children[0]
        assert cleanup_node.step_name == "stage_cleanup"
        assert len(cleanup_node.config["transforms"]) == 1
        assert cleanup_node.config["transforms"][0].__name__ == "RemoveIdentityOps"


def test_branching_blueprint_creates_shared_tree():
    """Test that branching in blueprint creates proper tree with sharing."""
    blueprint_yaml = """
version: "4.0"
name: "Branching Test"

design_space:
  transforms:
    # Stage A: single transform (shared prefix)
    imports:
      - ConvertQONNXtoFINN
    
    # Stage B: two mutually exclusive options
    cleanup:
      - [RemoveIdentityOps, RemoveUnusedTensors]
    
    # Stage C: optional transform
    optimize:
      - FoldConstants
      - ["~", InferShapes]
  
  kernels:
    - MVAU: [MVAU_hls, MVAU_rtl]
    - Thresholding: Thresholding_hls

build_pipeline:
  steps:
    - {imports}
    - {cleanup}
    - {optimize}
    - infer_kernels
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Calculate expected paths:
        # imports: 1 option
        # cleanup: 2 options
        # optimize: 2 combinations (FoldConstants, FoldConstants+InferShapes)
        # Total: 1 * 2 * 2 = 4 paths
        assert count_leaves(tree) == 4
        
        # Verify sharing
        stats = get_tree_stats(tree)
        assert stats['sharing_factor'] > 1.0
        
        # The imports stage should be shared (only one node)
        imports_count = 0
        def count_imports(node):
            nonlocal imports_count
            if node.step_name == "stage_imports":
                imports_count += 1
            for child in node.children:
                count_imports(child)
        
        count_imports(tree)
        assert imports_count == 1  # Shared across all paths


def test_complex_finn_pipeline():
    """Test a complex FINN-style pipeline with multiple stages."""
    blueprint_yaml = """
version: "4.0"
name: "FINN Pipeline Test"

global_config:
  output_stage: "synthesize_bitstream"
  working_directory: "work"
  save_intermediate_models: true
  max_combinations: 1000

design_space:
  transforms:
    # Import and initial cleanup
    init:
      - ConvertQONNXtoFINN
      - RemoveIdentityOps
      - ["~", RemoveUnusedTensors]
    
    # Streamlining with multiple choices
    streamline:
      - [
          AbsorbSignBiasIntoMultiThreshold,
          AbsorbAddIntoMultiThreshold,
          AbsorbMulIntoMultiThreshold
        ]
      - ["~", RoundAndClipThresholds]
      - [CollapseRepeatedOp, "~"]
    
    # Convert to HW layers
    convert:
      - InferQuantizedMatrixVectorActivation
      - ["~", InferThresholdingLayer]
      - InferConvInpGen
    
    # Optimization
    optimize:
      - ["~", MinimizeAccumulatorWidth]
      - ["~", MinimizeWeightBitWidth]
  
  kernels:
    - MVAU: [MVAU_hls, MVAU_rtl]
    - Thresholding: Thresholding_hls
    - ConvolutionInputGenerator: ConvolutionInputGenerator_hls

build_pipeline:
  steps:
    - step_prepare
    - {init}
    - {streamline}
    - {convert}
    - infer_kernels
    - {optimize}
    - step_create_dataflow
    - step_synthesize
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Verify global config was parsed
        assert design_space.global_config.working_directory == "work"
        assert design_space.global_config.save_intermediate_models == True
        assert design_space.global_config.max_combinations == 1000
        
        # Calculate paths:
        # init: 1 * 1 * 2 = 2
        # streamline: 3 * 2 * 2 = 12
        # convert: 1 * 2 * 1 = 2
        # optimize: 2 * 2 = 4
        # Total: 2 * 12 * 2 * 4 = 192 paths
        assert count_leaves(tree) == 192
        
        # With this many paths, sharing should be significant
        stats = get_tree_stats(tree)
        assert stats['sharing_factor'] > 2.0
        
        # Verify kernels are properly attached
        kernel_nodes = []
        def find_kernel_nodes(node, results):
            if node.step_name == "infer_kernels":
                results.append(node)
            for child in node.children:
                find_kernel_nodes(child, results)
        
        find_kernel_nodes(tree, kernel_nodes)
        
        # Should have multiple kernel nodes due to branching before
        assert len(kernel_nodes) > 1
        
        # But all should have same kernel config
        first_kernels = kernel_nodes[0].config["kernel_backends"]
        assert len(first_kernels) == 3  # MVAU, Thresholding, ConvolutionInputGenerator
        
        for node in kernel_nodes:
            assert node.config["kernel_backends"] == first_kernels


def test_invalid_transform_fails_early():
    """Test that invalid transforms are caught during parsing."""
    blueprint_yaml = """
version: "4.0"
name: "Invalid Transform Test"

design_space:
  transforms:
    bad_stage:
      - ThisTransformDoesNotExist
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {bad_stage}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        # Should fail during parsing
        with pytest.raises(ValueError) as exc_info:
            forge_tree(model_path, blueprint_path)
        
        assert "Transform 'ThisTransformDoesNotExist' not found" in str(exc_info.value)


def test_invalid_backend_fails_early():
    """Test that invalid backends are caught during parsing."""
    blueprint_yaml = """
version: "4.0"
name: "Invalid Backend Test"

design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
  
  kernels:
    - MVAU: NonExistentBackend

build_pipeline:
  steps:
    - {cleanup}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        with pytest.raises(ValueError) as exc_info:
            forge_tree(model_path, blueprint_path)
        
        assert "Backend 'NonExistentBackend' not found" in str(exc_info.value)


def test_size_limit_enforcement():
    """Test that design space size limits are enforced."""
    blueprint_yaml = """
version: "4.0"
name: "Size Limit Test"

global_config:
  max_combinations: 10  # Very low limit

design_space:
  transforms:
    # This will create too many combinations
    stage1:
      - [A, B, C, D]
    stage2:
      - [E, F, G, H]
    stage3:
      - [I, J, K, L]
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {stage1}
    - {stage2}
    - {stage3}
"""
    
    # First register some dummy transforms
    registry = get_registry()
    for name in "ABCDEFGHIJKL":
        class DummyTransform:
            pass
        DummyTransform.__name__ = name
        registry.register_transform(name, DummyTransform, stage="test")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        # Should fail due to size limit (4 * 4 * 4 = 64 > 10)
        with pytest.raises(ValueError) as exc_info:
            forge_tree(model_path, blueprint_path)
        
        assert "exceeds limit" in str(exc_info.value)


def test_empty_stages_handled_correctly():
    """Test that stages with only skip options don't create nodes."""
    blueprint_yaml = """
version: "4.0"
name: "Empty Stage Test"

design_space:
  transforms:
    # Stage that might be completely skipped
    optional_stage:
      - "~"
    
    # Normal stage after
    required_stage:
      - FoldConstants
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {optional_stage}
    - {required_stage}
    - infer_kernels
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Should skip the optional stage entirely
        # Tree: root -> required_stage -> infer_kernels
        assert count_nodes(tree) == 2
        
        # First child should be required_stage, not optional_stage
        assert tree.children[0].step_name == "stage_required_stage"


def test_pipeline_step_validation():
    """Test that pipeline steps referencing non-existent stages fail."""
    blueprint_yaml = """
version: "4.0"
name: "Pipeline Validation Test"

design_space:
  transforms:
    cleanup:
      - RemoveIdentityOps
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {cleanup}
    - {this_stage_does_not_exist}  # Error here
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        with pytest.raises(ValueError) as exc_info:
            forge_tree(model_path, blueprint_path)
        
        assert "Pipeline references stage 'this_stage_does_not_exist'" in str(exc_info.value)


def test_tree_execution_order():
    """Test that execution order respects dependencies."""
    blueprint_yaml = """
version: "4.0"
name: "Execution Order Test"

design_space:
  transforms:
    A:
      - RemoveIdentityOps
    B:
      - [FoldConstants, RemoveUnusedTensors]
    C:
      - InferShapes
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {A}
    - {B}
    - {C}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Get execution order
        from brainsmith.core.tree_builder import get_execution_order
        order = get_execution_order(tree)
        
        # Verify stages execute in correct order
        stage_positions = {}
        for i, node in enumerate(order):
            if node.step_name.startswith("stage_"):
                stage_name = node.step_name.replace("stage_", "")
                if stage_name not in stage_positions:
                    stage_positions[stage_name] = []
                stage_positions[stage_name].append(i)
        
        # A must come before B, B before C
        assert all(a_pos < b_pos 
                  for a_pos in stage_positions["A"] 
                  for b_pos in stage_positions["B"])
        assert all(b_pos < c_pos 
                  for b_pos in stage_positions["B"] 
                  for c_pos in stage_positions["C"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])