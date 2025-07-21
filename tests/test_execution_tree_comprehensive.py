"""
Comprehensive Execution Tree Test

This test exercises the full complexity of the execution tree system with:
- Multiple stages with various branching patterns
- Multiple kernels with different backend options
- Deep pipelines that create significant sharing opportunities
"""

import pytest
import tempfile
import os

from brainsmith.core.forge_v2 import forge_tree
from brainsmith.core.execution_tree import count_leaves, count_nodes, get_tree_stats, print_tree
from brainsmith.core.tree_builder import get_execution_order


def create_test_model(path: str):
    """Create a minimal ONNX model file for testing."""
    with open(path, "wb") as f:
        f.write(b"dummy_onnx_model")


def test_comprehensive_finn_hardware_pipeline():
    """
    Test a comprehensive hardware compilation pipeline that demonstrates:
    1. Multiple stages with different branching patterns
    2. Multiple kernels with various backend implementations
    3. Significant prefix sharing opportunities
    4. Realistic FINN/QONNX transform sequences
    """
    blueprint_yaml = """
version: "4.0"
name: "Comprehensive Hardware Acceleration Pipeline"

global_config:
  output_stage: "synthesize_bitstream"
  working_directory: "work"
  save_intermediate_models: true
  max_combinations: 50000

design_space:
  transforms:
    # Stage 1: Import and initial preparation (no branching)
    imports:
      - ConvertQONNXtoFINN
      - GiveUniqueNodeNames
      - GiveReadableTensorNames
    
    # Stage 2: Cleanup with optional transforms (2 options)
    cleanup:
      - RemoveIdentityOps
      - ["~", RemoveStaticGraphInputs]
      - ["~", RemoveUnusedTensors]
      - GiveUniqueParameterTensors
    
    # Stage 3: Graph transformations with choices (3 options)
    graph_transform:
      - ["~", BatchNormToAffine]
      - [GemmToMatMul, LowerConvsToMatMul, "~"]
      - FoldConstants
    
    # Stage 4: Streamlining with multiple exclusive options (6 options)
    streamline_absorb:
      - [
          AbsorbSignBiasIntoMultiThreshold,
          AbsorbAddIntoMultiThreshold,
          AbsorbMulIntoMultiThreshold
        ]
      - ["~", Absorb1BitMulIntoMatMul]
      - ["~", AbsorbTransposeIntoMultiThreshold]
    
    # Stage 5: Reordering transforms (4 options)
    streamline_reorder:
      - ["~", MoveScalarMulPastMatMul, MoveScalarAddPastMatMul]
      - ["~", MoveAddPastMul]
      - CollapseRepeatedOp
    
    # Stage 6: Convert to hardware layers (8 options)
    convert_to_hw:
      - InferQuantizedMatrixVectorActivation
      - ["~", InferThresholdingLayer]
      - ["~", InferConvInpGen]
      - ["~", InferStreamingMaxPool]
      - InferChannelwiseLinearLayer
    
    # Stage 7: Hardware optimizations (4 options)
    hw_optimize:
      - ["~", MinimizeAccumulatorWidth]
      - ["~", MinimizeWeightBitWidth]
      - InsertFIFO
      - SetFolding
    
    # Stage 8: Final preparation (2 options)
    final_prep:
      - ["~", InsertDWC]
      - InsertTLastMarker
      - PrepareIP
  
  # Multiple kernels with different backend options
  kernels:
    # Matrix/Vector operations
    - MVAU: [MVAU_hls, MVAU_rtl]
    - VVAU: [VVAU_hls, VVAU_rtl]
    
    # Activation functions
    - Thresholding: [Thresholding_hls, Thresholding_rtl]
    - ChannelwiseOp: ChannelwiseOp_hls
    
    # Memory and data movement
    - ConvolutionInputGenerator: [ConvolutionInputGenerator_hls, ConvolutionInputGenerator_rtl]
    - StreamingFIFO: StreamingFIFO_rtl
    - StreamingDataWidthConverter: [StreamingDataWidthConverter_hls, StreamingDataWidthConverter_rtl]
    
    # Pooling and aggregation
    - StreamingMaxPool: StreamingMaxPool_hls
    - GlobalAccPool: GlobalAccPool_hls

build_pipeline:
  steps:
    - step_load_onnx
    - {imports}
    - {cleanup}
    - {graph_transform}
    - {streamline_absorb}
    - {streamline_reorder}
    - step_save_intermediate
    - {convert_to_hw}
    - infer_kernels
    - {hw_optimize}
    - {final_prep}
    - step_generate_verilog
    - step_synthesize_bitstream
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        # Parse blueprint to tree
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Calculate expected combinations:
        # imports: 1 (no branching)
        # cleanup: 1 * 2 * 2 * 1 = 4
        # graph_transform: 2 * 3 * 1 = 6
        # streamline_absorb: 3 * 2 * 2 = 12
        # streamline_reorder: 3 * 2 * 1 = 6
        # convert_to_hw: 1 * 2 * 2 * 2 * 1 = 8
        # hw_optimize: 2 * 2 * 1 * 1 = 4
        # final_prep: 2 * 1 * 1 = 2
        # Total paths: 1 * 4 * 6 * 12 * 6 * 8 * 4 * 2 = 221,184
        
        # But we set max_combinations to 50,000
        # So the tree building should succeed but be limited
        total_paths = count_leaves(tree)
        assert total_paths <= 50000
        
        # Verify significant sharing is happening
        stats = get_tree_stats(tree)
        print(f"\nTree Statistics:")
        print(f"  Total paths: {stats['total_paths']:,}")
        print(f"  Total nodes: {stats['total_nodes']:,}")
        print(f"  Max depth: {stats['max_depth']}")
        print(f"  Sharing factor: {stats['sharing_factor']}x")
        print(f"  Saved nodes: {stats['saved_nodes']:,}")
        
        # With this many stages, sharing should be very significant
        assert stats['sharing_factor'] > 5.0
        
        # Verify all 9 kernels are present
        kernel_nodes = []
        def find_kernel_nodes(node, results):
            if node.step_name == "infer_kernels":
                results.append(node)
            for child in node.children:
                find_kernel_nodes(child, results)
        
        find_kernel_nodes(tree, kernel_nodes)
        assert len(kernel_nodes) > 0
        
        # Check that all kernel nodes have all 9 kernels
        first_kernels = kernel_nodes[0].config["kernel_backends"]
        assert len(first_kernels) == 9
        
        kernel_names = [k[0] for k in first_kernels]
        expected_kernels = {
            "MVAU", "VVAU", "Thresholding", "ChannelwiseOp",
            "ConvolutionInputGenerator", "StreamingFIFO",
            "StreamingDataWidthConverter", "StreamingMaxPool", "GlobalAccPool"
        }
        assert set(kernel_names) == expected_kernels
        
        # Verify early stages are maximally shared
        imports_nodes = []
        def count_stage_nodes(node, stage_name, results):
            if node.step_name == f"stage_{stage_name}":
                results.append(node)
            for child in node.children:
                count_stage_nodes(child, stage_name, results)
        
        count_stage_nodes(tree, "imports", imports_nodes)
        # imports has no branching, so should have exactly 1 node
        assert len(imports_nodes) == 1
        
        # Verify pipeline steps execute in order
        execution_order = get_execution_order(tree)
        step_names = [node.step_name for node in execution_order]
        
        # Check that stages appear in pipeline order
        stage_order = [
            "stage_imports", "stage_cleanup", "stage_graph_transform",
            "stage_streamline_absorb", "stage_streamline_reorder",
            "stage_convert_to_hw", "infer_kernels", "stage_hw_optimize",
            "stage_final_prep"
        ]
        
        # Get first occurrence of each stage
        stage_positions = {}
        for i, name in enumerate(step_names):
            if name in stage_order and name not in stage_positions:
                stage_positions[name] = i
        
        # Verify order is preserved
        for i in range(len(stage_order) - 1):
            if stage_order[i] in stage_positions and stage_order[i+1] in stage_positions:
                assert stage_positions[stage_order[i]] < stage_positions[stage_order[i+1]]


def test_kernel_backend_variations():
    """Test that different kernel backend selections work correctly."""
    blueprint_yaml = """
version: "4.0"
name: "Kernel Backend Variation Test"

design_space:
  transforms:
    # Simple stages to focus on kernel testing
    prep:
      - ConvertQONNXtoFINN
      - InferQuantizedMatrixVectorActivation
    
    optimize:
      - ["~", MinimizeAccumulatorWidth]
  
  # Test various kernel backend specifications
  kernels:
    # Single backend specified
    - MVAU: MVAU_hls
    
    # Multiple backends specified as list
    - Thresholding: [Thresholding_hls, Thresholding_rtl]
    
    # All available backends (when not specified)
    - StreamingDataWidthConverter
    
    # RTL-only kernel
    - StreamingFIFO: StreamingFIFO_rtl
    
    # HLS-only kernel  
    - ChannelwiseOp: ChannelwiseOp_hls

build_pipeline:
  steps:
    - {prep}
    - infer_kernels
    - {optimize}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Find a kernel node
        kernel_node = None
        def find_first_kernel_node(node):
            nonlocal kernel_node
            if node.step_name == "infer_kernels":
                kernel_node = node
                return
            for child in node.children:
                find_first_kernel_node(child)
        
        find_first_kernel_node(tree)
        assert kernel_node is not None
        
        kernels = kernel_node.config["kernel_backends"]
        assert len(kernels) == 5
        
        # Check each kernel has correct backends
        kernel_dict = {k[0]: k[1] for k in kernels}
        
        # MVAU: single HLS backend
        assert len(kernel_dict["MVAU"]) == 1
        assert kernel_dict["MVAU"][0].__name__ == "MVAU_hls"
        
        # Thresholding: both HLS and RTL
        assert len(kernel_dict["Thresholding"]) == 2
        backend_names = {b.__name__ for b in kernel_dict["Thresholding"]}
        assert backend_names == {"Thresholding_hls", "Thresholding_rtl"}
        
        # StreamingDataWidthConverter: should have multiple backends
        assert len(kernel_dict["StreamingDataWidthConverter"]) >= 2
        
        # StreamingFIFO: RTL only
        assert len(kernel_dict["StreamingFIFO"]) == 1
        assert kernel_dict["StreamingFIFO"][0].__name__ == "StreamingFIFO_rtl"
        
        # ChannelwiseOp: HLS only
        assert len(kernel_dict["ChannelwiseOp"]) == 1
        assert kernel_dict["ChannelwiseOp"][0].__name__ == "ChannelwiseOp_hls"


def test_deep_branching_with_convergence():
    """Test a pipeline that branches early and converges late."""
    blueprint_yaml = """
version: "4.0"
name: "Deep Branching Test"

design_space:
  transforms:
    # Early branching point - creates 3 paths
    early_branch:
      - [
          ConvertQONNXtoFINN,
          QCDQToQuant,
          QuantToQCDQ
        ]
    
    # Multiple stages that maintain branches
    stage_a:
      - ["~", RemoveIdentityOps]
      - FoldConstants
    
    stage_b:
      - ["~", BatchNormToAffine]
      - ["~", GemmToMatMul]
    
    stage_c:
      - [AbsorbSignBiasIntoMultiThreshold, "~"]
      - RoundAndClipThresholds
    
    # Convergence point - no branching
    converge:
      - InferShapes
      - InferDataTypes
    
    # Another branching after convergence
    late_branch:
      - ["~", MinimizeAccumulatorWidth, MinimizeWeightBitWidth]
  
  kernels:
    - MVAU: [MVAU_hls, MVAU_rtl]
    - Thresholding: Thresholding_hls
    - VVAU: VVAU_hls

build_pipeline:
  steps:
    - {early_branch}
    - {stage_a}
    - {stage_b}
    - {stage_c}
    - {converge}
    - infer_kernels
    - {late_branch}
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Calculate paths:
        # early_branch: 3
        # stage_a: 2 * 1 = 2
        # stage_b: 2 * 2 = 4
        # stage_c: 2 * 1 = 2
        # converge: 1 (no branching)
        # late_branch: 3
        # Total: 3 * 2 * 4 * 2 * 1 * 3 = 144 paths
        
        assert count_leaves(tree) == 144
        
        # The converge stage should have fewer nodes than leaf count
        # because many paths converge there
        converge_nodes = []
        def find_converge_nodes(node):
            if node.step_name == "stage_converge":
                converge_nodes.append(node)
            for child in node.children:
                find_converge_nodes(child)
        
        find_converge_nodes(tree)
        # Should be 3 * 2 * 4 * 2 = 48 converge nodes (before convergence)
        assert len(converge_nodes) == 48
        
        # But after converge, we branch again with late_branch
        # so we should see the branching factor increase again
        stats = get_tree_stats(tree)
        assert stats['sharing_factor'] > 2.0


def test_transform_step_ordering():
    """Test that transform steps within a stage maintain order."""
    blueprint_yaml = """
version: "4.0"
name: "Transform Step Order Test"

design_space:
  transforms:
    # Multi-step stage with specific ordering requirements
    ordered_stage:
      - RemoveIdentityOps         # Step 1: Always first
      - ["~", FoldConstants]      # Step 2: Optional
      - InferShapes              # Step 3: Always after folding
      - ["~", InferDataTypes]    # Step 4: Optional
      - GiveUniqueNodeNames      # Step 5: Always last
  
  kernels:
    - MVAU: MVAU_hls

build_pipeline:
  steps:
    - {ordered_stage}
    - infer_kernels
"""
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "model.onnx")
        blueprint_path = os.path.join(tmpdir, "blueprint.yaml")
        
        create_test_model(model_path)
        with open(blueprint_path, "w") as f:
            f.write(blueprint_yaml)
        
        design_space, tree = forge_tree(model_path, blueprint_path)
        
        # Get all stage nodes
        stage_nodes = []
        def collect_stage_nodes(node):
            if node.step_name == "stage_ordered_stage":
                stage_nodes.append(node)
            for child in node.children:
                collect_stage_nodes(child)
        
        collect_stage_nodes(tree)
        
        # Should have 2 * 2 = 4 combinations
        assert len(stage_nodes) == 4
        
        # Check that transform order is preserved in all variants
        for node in stage_nodes:
            transforms = node.config["transforms"]
            transform_names = [t.__name__ for t in transforms]
            
            # RemoveIdentityOps should always be first
            assert transform_names[0] == "RemoveIdentityOps"
            
            # InferShapes should always be present
            assert "InferShapes" in transform_names
            
            # GiveUniqueNodeNames should always be last
            assert transform_names[-1] == "GiveUniqueNodeNames"
            
            # If FoldConstants is present, it should come before InferShapes
            if "FoldConstants" in transform_names:
                fold_idx = transform_names.index("FoldConstants")
                shapes_idx = transform_names.index("InferShapes")
                assert fold_idx < shapes_idx
            
            # If InferDataTypes is present, it should come after InferShapes
            if "InferDataTypes" in transform_names:
                shapes_idx = transform_names.index("InferShapes")
                types_idx = transform_names.index("InferDataTypes")
                assert shapes_idx < types_idx


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])  # -s to see print output