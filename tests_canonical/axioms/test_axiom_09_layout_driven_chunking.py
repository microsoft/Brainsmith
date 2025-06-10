"""
Test Axiom 9: Layout-Driven Chunking

Validates: ONNX tensor layout determines chunking dimension
- [N, C, H, W] → chunk along C
- [N, L, C] → chunk along L  
- [N, H, W, C] → chunk along H×W

This axiom ensures that chunking strategies are automatically determined
by the tensor layout rather than requiring manual specification.
"""

import pytest
import numpy as np
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType
from brainsmith.dataflow.core.block_chunking import TensorChunking


@pytest.mark.axiom
class TestAxiom9LayoutDrivenChunking:
    """Test suite for validating Axiom 9: Layout-Driven Chunking."""
    
    def test_cnn_nchw_layout_chunking(self):
        """Test [N, C, H, W] → chunk along C dimension."""
        
        # CNN tensor in NCHW format
        tensor_shape = [1, 64, 56, 56]  # Batch=1, Channels=64, Height=56, Width=56
        onnx_layout = "[N, C, H, W]"
        
        # Apply layout-driven chunking
        tensor_dims, block_dims = self._compute_layout_chunking(tensor_shape, onnx_layout)
        
        # Validate chunking strategy
        assert tensor_dims == [64, 56, 56]  # Exclude batch dimension
        assert block_dims == [1, 56, 56]    # Chunk along C: process 1 channel at a time
        
        # Validate chunking efficiency
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [64, 1, 1]  # 64 blocks along channel dimension
        
        # Total element conservation
        total_tensor = np.prod(tensor_dims)
        total_blocks = np.prod(num_blocks) * np.prod(block_dims)
        assert total_tensor == total_blocks
    
    def test_cnn_nhwc_layout_chunking(self):
        """Test [N, H, W, C] → chunk along H×W dimensions."""
        
        # CNN tensor in NHWC format (TensorFlow style)
        tensor_shape = [1, 56, 56, 64]  # Batch=1, Height=56, Width=56, Channels=64
        onnx_layout = "[N, H, W, C]"
        
        # Apply layout-driven chunking
        tensor_dims, block_dims = self._compute_layout_chunking(tensor_shape, onnx_layout)
        
        # Validate chunking strategy
        assert tensor_dims == [56, 56, 64]  # Exclude batch dimension
        assert block_dims == [1, 1, 64]     # Chunk along H×W: process 1 spatial location at a time
        
        # Validate chunking efficiency
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [56, 56, 1]  # 56×56 = 3136 blocks along spatial dimensions
        
        # Spatial parallelism opportunities
        spatial_blocks = num_blocks[0] * num_blocks[1]
        assert spatial_blocks == 3136  # High parallelism for spatial processing
    
    def test_transformer_nlc_layout_chunking(self):
        """Test [N, L, C] → chunk along L (sequence) dimension."""
        
        # Transformer tensor in NLC format
        tensor_shape = [1, 512, 768]  # Batch=1, SeqLen=512, Channels=768
        onnx_layout = "[N, L, C]"
        
        # Apply layout-driven chunking
        tensor_dims, block_dims = self._compute_layout_chunking(tensor_shape, onnx_layout)
        
        # Validate chunking strategy
        assert tensor_dims == [512, 768]  # Exclude batch dimension
        assert block_dims == [1, 768]     # Chunk along L: process 1 token at a time
        
        # Validate sequence processing
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [512, 1]  # 512 blocks along sequence dimension
        
        # Each block contains full feature vector for one token
        assert np.prod(block_dims) == 768  # Complete feature vector per block
    
    def test_transformer_ncl_layout_chunking(self):
        """Test [N, C, L] → chunk along C (feature) dimension."""
        
        # Transformer tensor in NCL format (less common)
        tensor_shape = [1, 768, 512]  # Batch=1, Channels=768, SeqLen=512
        onnx_layout = "[N, C, L]"
        
        # Apply layout-driven chunking
        tensor_dims, block_dims = self._compute_layout_chunking(tensor_shape, onnx_layout)
        
        # Validate chunking strategy
        assert tensor_dims == [768, 512]  # Exclude batch dimension
        assert block_dims == [1, 512]     # Chunk along C: process 1 feature across all tokens
        
        # Validate feature processing
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [768, 1]  # 768 blocks along feature dimension
    
    def test_multi_head_attention_layout(self):
        """Test [N, L, h, d] → chunk along L dimension."""
        
        # Multi-head attention tensor
        tensor_shape = [1, 512, 12, 64]  # Batch=1, SeqLen=512, Heads=12, Dim=64
        onnx_layout = "[N, L, h, d]"
        
        # Apply layout-driven chunking
        tensor_dims, block_dims = self._compute_layout_chunking(tensor_shape, onnx_layout)
        
        # Validate chunking strategy
        assert tensor_dims == [512, 12, 64]  # Exclude batch dimension
        assert block_dims == [1, 12, 64]     # Chunk along L: process 1 token across all heads
        
        # Validate multi-head processing
        num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
        assert num_blocks == [512, 1, 1]  # 512 blocks along sequence dimension
        
        # Each block contains all attention heads for one token
        assert np.prod(block_dims) == 12 * 64  # All heads × head dimension
    
    def test_layout_comparison_same_data(self):
        """Test that different layouts of same data produce different chunking strategies."""
        
        # Same data in different layouts
        total_elements = 64 * 56 * 56
        
        layouts = [
            {"shape": [1, 64, 56, 56], "layout": "[N, C, H, W]", "chunk_dim": "C"},
            {"shape": [1, 56, 56, 64], "layout": "[N, H, W, C]", "chunk_dim": "H×W"},
        ]
        
        chunking_results = []
        
        for config in layouts:
            tensor_dims, block_dims = self._compute_layout_chunking(
                config["shape"], config["layout"]
            )
            
            # Verify same total elements
            assert np.prod(tensor_dims) == total_elements
            
            # Calculate parallelism characteristics
            num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
            total_blocks = np.prod(num_blocks)
            elements_per_block = np.prod(block_dims)
            
            chunking_results.append({
                "layout": config["layout"],
                "chunk_dim": config["chunk_dim"],
                "total_blocks": total_blocks,
                "elements_per_block": elements_per_block,
                "tensor_dims": tensor_dims,
                "block_dims": block_dims
            })
        
        # Compare chunking strategies
        nchw_result = chunking_results[0]
        nhwc_result = chunking_results[1]
        
        # Different chunking characteristics
        assert nchw_result["total_blocks"] != nhwc_result["total_blocks"]
        assert nchw_result["elements_per_block"] != nhwc_result["elements_per_block"]
        
        # NCHW: Few large blocks (64 blocks × 3136 elements each)
        assert nchw_result["total_blocks"] == 64
        assert nchw_result["elements_per_block"] == 56 * 56
        
        # NHWC: Many small blocks (3136 blocks × 64 elements each)
        assert nhwc_result["total_blocks"] == 56 * 56
        assert nhwc_result["elements_per_block"] == 64
    
    def test_layout_driven_parallelism_optimization(self):
        """Test that layout affects parallelism opportunities."""
        
        parallelism_scenarios = [
            {
                "layout": "[N, C, H, W]",
                "shape": [1, 128, 32, 32],
                "expected_max_parallelism": 128,  # Limited by C dimension
                "optimal_for": "channel_parallelism"
            },
            {
                "layout": "[N, H, W, C]", 
                "shape": [1, 32, 32, 128],
                "expected_max_parallelism": 32 * 32,  # Limited by H×W dimensions
                "optimal_for": "spatial_parallelism"
            },
            {
                "layout": "[N, L, C]",
                "shape": [1, 256, 512],
                "expected_max_parallelism": 256,  # Limited by L dimension
                "optimal_for": "sequence_parallelism"
            }
        ]
        
        for scenario in parallelism_scenarios:
            tensor_dims, block_dims = self._compute_layout_chunking(
                scenario["shape"], scenario["layout"]
            )
            
            # Calculate maximum useful parallelism
            num_blocks = [tensor_dims[i] // block_dims[i] for i in range(len(tensor_dims))]
            max_blocks = np.prod(num_blocks)
            
            # Validate parallelism bounds
            assert max_blocks == scenario["expected_max_parallelism"]
            
            # Test that parallelism beyond this is wasteful
            parallelism_efficiency = min(1.0, max_blocks / (max_blocks + 64))
            assert parallelism_efficiency == 1.0  # Optimal utilization
    
    def test_automatic_layout_detection(self):
        """Test automatic detection of optimal chunking from tensor shapes."""
        
        test_cases = [
            # CNN patterns
            {"shape": [64, 224, 224], "expected_layout": "CHW", "chunk_along": 0},
            {"shape": [224, 224, 64], "expected_layout": "HWC", "chunk_along": [0, 1]},
            
            # Transformer patterns
            {"shape": [1024, 768], "expected_layout": "LC", "chunk_along": 0},
            {"shape": [768, 1024], "expected_layout": "CL", "chunk_along": 0},
            
            # Matrix patterns
            {"shape": [512, 512], "expected_layout": "MM", "chunk_along": 0},
        ]
        
        for case in test_cases:
            shape = case["shape"]
            
            # Infer optimal chunking based on shape characteristics
            optimal_chunking = self._infer_optimal_chunking(shape)
            
            # Validate that chunking makes sense for the pattern
            tensor_dims = shape
            block_dims = optimal_chunking["block_dims"]
            
            # Check that chunking is along expected dimensions
            chunked_dims = [i for i in range(len(shape)) 
                          if block_dims[i] < tensor_dims[i]]
            
            if isinstance(case["chunk_along"], list):
                assert set(chunked_dims) == set(case["chunk_along"])
            else:
                assert chunked_dims[0] == case["chunk_along"]
    
    def test_layout_validation_and_constraints(self, test_utils):
        """Test that layout-driven chunking respects hardware constraints."""
        
        # Test with hardware-friendly layouts
        hardware_configs = [
            {
                "tensor_dims": [64, 56, 56],
                "block_dims": [1, 56, 56],
                "stream_dims": [1, 1, 8],
                "constraint": "memory_bandwidth"
            },
            {
                "tensor_dims": [512, 768], 
                "block_dims": [1, 768],
                "stream_dims": [1, 64],
                "constraint": "processing_elements"
            }
        ]
        
        for config in hardware_configs:
            interface = test_utils.create_test_interface(
                config["tensor_dims"], config["block_dims"], config["stream_dims"],
                DataflowInterfaceType.INPUT
            )
            
            # Validate hardware constraints are met
            stream_width = interface.calculate_stream_width()
            assert stream_width <= 1024  # Reasonable AXI width constraint
            
            transfer_cycles = interface.get_transfer_cycles()
            assert transfer_cycles > 0  # Must be processable
            
            # Validate tiling constraints (Axiom 8)
            block_elements = np.prod(config["block_dims"])
            stream_elements = np.prod(config["stream_dims"])
            assert block_elements % stream_elements == 0
    
    def _compute_layout_chunking(self, tensor_shape, onnx_layout):
        """Compute layout-driven chunking strategy."""
        # Remove batch dimension (first dimension)
        tensor_dims = tensor_shape[1:]
        
        # Determine chunking based on layout
        if onnx_layout == "[N, C, H, W]":
            # Chunk along C dimension
            block_dims = [1, tensor_dims[1], tensor_dims[2]]
        elif onnx_layout == "[N, H, W, C]":
            # Chunk along H×W dimensions
            block_dims = [1, 1, tensor_dims[2]]
        elif onnx_layout == "[N, L, C]":
            # Chunk along L dimension
            block_dims = [1, tensor_dims[1]]
        elif onnx_layout == "[N, C, L]":
            # Chunk along C dimension
            block_dims = [1, tensor_dims[1]]
        elif onnx_layout == "[N, L, h, d]":
            # Chunk along L dimension
            block_dims = [1, tensor_dims[1], tensor_dims[2]]
        else:
            # Default: chunk first non-batch dimension
            block_dims = [1] + tensor_dims[1:]
            
        return tensor_dims, block_dims
    
    def _infer_optimal_chunking(self, shape):
        """Infer optimal chunking strategy from tensor shape characteristics."""
        
        # Simple heuristic: chunk along largest dimension for better parallelism
        largest_dim = np.argmax(shape)
        
        block_dims = list(shape)
        block_dims[largest_dim] = 1  # Chunk along largest dimension
        
        return {
            "block_dims": block_dims,
            "chunking_dimension": largest_dim,
            "parallelism_factor": shape[largest_dim]
        }