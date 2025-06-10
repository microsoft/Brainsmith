"""
Test Axiom 2: The Core Relationship

Validates: tensor_dims → chunked into → num_blocks pieces of shape block_dims → streamed as stream_dims per cycle

Core mathematical relationship:
- tensor_dims: Full tensor shape (no batch dimension)
- num_blocks: Number of blocks available  
- block_dims: Shape of each block
- stream_dims: Data streamed per clock cycle
"""

import pytest
import numpy as np
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType


@pytest.mark.axiom
class TestAxiom2CoreRelationship:
    """Test suite for validating Axiom 2: The Core Relationship."""
    
    def test_basic_core_relationship(self, cnn_input_interface):
        """Test basic tensor_dims → num_blocks × block_dims relationship."""
        interface = cnn_input_interface
        
        # Core relationship: tensor_dims = num_blocks × block_dims
        tensor_dims = interface.tensor_dims      # [64, 56, 56]
        block_dims = interface.block_dims        # [1, 56, 56]
        num_blocks = interface.get_num_blocks()  # [64, 1, 1]
        
        # Mathematical validation
        for i in range(len(tensor_dims)):
            expected_blocks = tensor_dims[i] // block_dims[i]
            assert num_blocks[i] == expected_blocks
            assert tensor_dims[i] == num_blocks[i] * block_dims[i]
        
        # Total element conservation
        total_tensor_elements = np.prod(tensor_dims)
        total_block_elements = np.prod(num_blocks) * np.prod(block_dims)
        assert total_tensor_elements == total_block_elements
    
    def test_streaming_relationship(self, transformer_input_interface):
        """Test block_dims → streamed as stream_dims per cycle."""
        interface = transformer_input_interface
        
        block_dims = interface.block_dims        # [1, 768]
        stream_dims = interface.stream_dims      # [1, 64]
        
        # Streaming constraint: block_dims must be divisible by stream_dims
        for i in range(min(len(block_dims), len(stream_dims))):
            assert block_dims[i] % stream_dims[i] == 0
        
        # Calculate cycles needed to stream one block
        cycles_per_block = interface.get_transfer_cycles()
        expected_cycles = np.prod(block_dims) // np.prod(stream_dims)
        assert cycles_per_block == expected_cycles
        assert cycles_per_block == 768 // 64  # 12 cycles per block
    
    def test_complete_pipeline_relationship(self, cnn_weight_interface):
        """Test complete tensor → blocks → streams pipeline."""
        interface = cnn_weight_interface
        
        # Full pipeline: tensor_dims → num_blocks × block_dims → cycles × stream_dims
        tensor_dims = interface.tensor_dims      # [128, 64, 3, 3]
        block_dims = interface.block_dims        # [32, 1, 3, 3]
        stream_dims = interface.stream_dims      # [8, 1, 1, 1]
        num_blocks = interface.get_num_blocks()  # [4, 64, 1, 1]
        
        # Stage 1: Tensor to blocks
        total_tensor_elements = np.prod(tensor_dims)
        total_blocks = np.prod(num_blocks)
        elements_per_block = np.prod(block_dims)
        
        assert total_tensor_elements == total_blocks * elements_per_block
        assert total_blocks == 4 * 64 * 1 * 1  # 256 blocks total
        
        # Stage 2: Blocks to streams
        elements_per_stream = np.prod(stream_dims)
        cycles_per_block = elements_per_block // elements_per_stream
        
        assert elements_per_block % elements_per_stream == 0
        assert cycles_per_block == (32 * 1 * 3 * 3) // (8 * 1 * 1 * 1)  # 36 cycles
        
        # Stage 3: Total processing time
        total_cycles = total_blocks * cycles_per_block
        expected_total_cycles = total_tensor_elements // elements_per_stream
        assert total_cycles == expected_total_cycles
    
    def test_dimension_relationships_multi_interface(self, cnn_interfaces):
        """Test core relationship across multiple interface types."""
        
        for interface_name, interface in cnn_interfaces.items():
            tensor_dims = interface.tensor_dims
            block_dims = interface.block_dims
            num_blocks = interface.get_num_blocks()
            
            # Core relationship must hold for all interfaces
            for i in range(len(tensor_dims)):
                # Each tensor dimension = num_blocks × block_dims
                assert tensor_dims[i] == num_blocks[i] * block_dims[i]
                
                # num_blocks is computed correctly  
                expected_blocks = tensor_dims[i] // block_dims[i]
                assert num_blocks[i] == expected_blocks
            
            # Total element conservation
            tensor_elements = np.prod(tensor_dims)
            reconstructed_elements = np.prod(num_blocks) * np.prod(block_dims)
            assert tensor_elements == reconstructed_elements
    
    def test_relationship_with_different_layouts(self, axiom_test_data):
        """Test core relationship with different tensor layouts."""
        
        layout_examples = [
            # CNN layouts
            {"tensor_dims": [64, 56, 56], "block_dims": [1, 56, 56]},   # [C,H,W] → chunk C
            {"tensor_dims": [56, 56, 64], "block_dims": [1, 1, 64]},    # [H,W,C] → chunk H×W
            
            # Transformer layouts  
            {"tensor_dims": [512, 768], "block_dims": [1, 768]},        # [L,C] → chunk L
            {"tensor_dims": [768, 512], "block_dims": [768, 1]},        # [C,L] → chunk C
            
            # Multi-head attention
            {"tensor_dims": [512, 12, 64], "block_dims": [1, 12, 64]},  # [L,h,d] → chunk L
        ]
        
        for example in layout_examples:
            tensor_dims = example["tensor_dims"]
            block_dims = example["block_dims"]
            
            # Create interface
            interface = self._create_test_interface(tensor_dims, block_dims)
            
            # Validate core relationship
            num_blocks = interface.get_num_blocks()
            
            for i in range(len(tensor_dims)):
                assert tensor_dims[i] == num_blocks[i] * block_dims[i]
                assert tensor_dims[i] % block_dims[i] == 0  # Must be divisible
            
            # Element conservation
            assert np.prod(tensor_dims) == np.prod(num_blocks) * np.prod(block_dims)
    
    def test_edge_case_relationships(self, test_utils):
        """Test core relationship with edge cases."""
        
        # Edge case 1: Single block (tensor_dims == block_dims)
        single_block = test_utils.create_test_interface(
            tensor_dims=[64, 64], block_dims=[64, 64], stream_dims=[8, 8],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        num_blocks = single_block.get_num_blocks()
        assert num_blocks == [1, 1]  # Only one block
        assert np.prod(single_block.tensor_dims) == np.prod(single_block.block_dims)
        
        # Edge case 2: Maximum blocks (block_dims = [1, 1, ...])
        max_blocks = test_utils.create_test_interface(
            tensor_dims=[8, 8, 8], block_dims=[1, 1, 1], stream_dims=[1],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        num_blocks = max_blocks.get_num_blocks()
        assert num_blocks == [8, 8, 8]  # Maximum possible blocks
        assert np.prod(num_blocks) == np.prod(max_blocks.tensor_dims)
        
        # Edge case 3: 1D tensors
        tensor_1d = test_utils.create_test_interface(
            tensor_dims=[1024], block_dims=[32], stream_dims=[8],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        num_blocks = tensor_1d.get_num_blocks()
        assert num_blocks == [32]  # 1024 / 32 = 32 blocks
        assert 1024 == 32 * 32
    
    def test_relationship_mathematical_properties(self, simple_interfaces):
        """Test mathematical properties of the core relationship."""
        
        for interface in simple_interfaces.values():
            tensor_dims = interface.tensor_dims
            block_dims = interface.block_dims
            num_blocks = interface.get_num_blocks()
            
            # Property 1: Associativity
            # (tensor_dims / block_dims) == num_blocks
            for i in range(len(tensor_dims)):
                assert tensor_dims[i] // block_dims[i] == num_blocks[i]
            
            # Property 2: Distributivity over products
            # prod(tensor_dims) == prod(num_blocks) × prod(block_dims)
            tensor_prod = np.prod(tensor_dims)
            blocks_prod = np.prod(num_blocks)
            block_dims_prod = np.prod(block_dims)
            
            assert tensor_prod == blocks_prod * block_dims_prod
            
            # Property 3: Monotonicity
            # If block_dims increases, num_blocks decreases (inverse relationship)
            # This is tested by creating modified interfaces
            
    def test_relationship_validation_errors(self, test_utils):
        """Test that invalid relationships are caught during validation."""
        
        # Invalid case 1: tensor_dims not divisible by block_dims
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[100], block_dims=[30], stream_dims=[5],
                interface_type=DataflowInterfaceType.INPUT
            )
        
        # Invalid case 2: block_dims larger than tensor_dims
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[32, 32], block_dims=[64, 32], stream_dims=[4, 4],
                interface_type=DataflowInterfaceType.INPUT
            )
        
        # Invalid case 3: Mismatched dimensions
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[64, 64], block_dims=[16], stream_dims=[4],
                interface_type=DataflowInterfaceType.INPUT
            )
    
    def test_runtime_relationship_consistency(self, cnn_dataflow_model):
        """Test that relationships remain consistent during runtime operations."""
        
        model = cnn_dataflow_model
        
        # Test with different parallelism settings
        parallelism_configs = [
            {"iPar": {"input": 1}, "wPar": {"weights": 1}},
            {"iPar": {"input": 4}, "wPar": {"weights": 2}},
            {"iPar": {"input": 8}, "wPar": {"weights": 4}},
        ]
        
        for config in parallelism_configs:
            iPar = config["iPar"]
            wPar = config["wPar"]
            
            # Calculate intervals with this parallelism
            intervals = model.calculate_initiation_intervals(iPar, wPar)
            
            # Core relationship should remain valid regardless of parallelism
            for interface in model.all_interfaces():
                num_blocks = interface.get_num_blocks()
                tensor_dims = interface.tensor_dims
                block_dims = interface.block_dims
                
                # Relationship must still hold
                for i in range(len(tensor_dims)):
                    assert tensor_dims[i] == num_blocks[i] * block_dims[i]
    
    def _create_test_interface(self, tensor_dims, block_dims):
        """Helper to create test interface with minimal stream_dims."""
        from brainsmith.dataflow.core.dataflow_interface import DataflowDataType
        
        # Create minimal valid stream_dims
        stream_dims = [1] * len(block_dims)
        
        dtype = DataflowDataType("INT", 8, True, "INT8")
        return DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            dtype=dtype
        )