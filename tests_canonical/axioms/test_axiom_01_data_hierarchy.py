"""
Test Axiom 1: Data Hierarchy

Validates: Tensor → Block → Stream → Element
- Tensor: Complete data (entire hidden state/weight)
- Block: Minimum data for one calculation  
- Stream: Data per clock cycle
- Element: Single value
"""

import pytest
import numpy as np
from brainsmith.dataflow.core.dataflow_interface import DataflowInterface, DataflowInterfaceType


@pytest.mark.axiom
class TestAxiom1DataHierarchy:
    """Test suite for validating Axiom 1: Data Hierarchy."""
    
    def test_tensor_to_block_relationship(self, cnn_input_interface):
        """Test that blocks properly tile into tensors."""
        interface = cnn_input_interface
        
        # Tensor level: Complete data
        tensor_elements = np.prod(interface.tensor_dims)
        assert tensor_elements == 64 * 56 * 56  # Total tensor elements
        
        # Block level: Minimum data for calculation
        block_elements = np.prod(interface.block_dims)
        assert block_elements == 1 * 56 * 56   # Elements per block
        
        # Relationship: tensor_dims = num_blocks × block_dims
        num_blocks = interface.get_num_blocks()
        total_block_elements = np.prod(num_blocks) * block_elements
        
        assert tensor_elements == total_block_elements
        assert num_blocks == [64, 1, 1]  # 64 blocks in C dimension
    
    def test_block_to_stream_relationship(self, cnn_input_interface):
        """Test that streams properly tile into blocks."""
        interface = cnn_input_interface
        
        # Block level: Minimum data for calculation
        block_elements = np.prod(interface.block_dims)
        assert block_elements == 1 * 56 * 56
        
        # Stream level: Data per clock cycle
        stream_elements = np.prod(interface.stream_dims)
        assert stream_elements == 1 * 1 * 8  # 8 elements per cycle
        
        # Relationship: blocks must be divisible by streams
        cycles_per_block = block_elements // stream_elements
        assert cycles_per_block == (1 * 56 * 56) // (1 * 1 * 8)
        assert cycles_per_block == 392  # Clock cycles to process one block
        
        # Validate tiling constraint
        assert block_elements % stream_elements == 0
    
    def test_stream_to_element_relationship(self, cnn_input_interface):
        """Test that elements properly compose streams."""
        interface = cnn_input_interface
        
        # Stream level: Data per clock cycle
        stream_elements = np.prod(interface.stream_dims)
        assert stream_elements == 8
        
        # Element level: Single value with specific bitwidth
        element_bitwidth = interface.dtype.bitwidth
        assert element_bitwidth == 8
        
        # Stream width calculation
        stream_width_bits = stream_elements * element_bitwidth
        calculated_width = interface.calculate_stream_width()
        
        # Should be aligned to 8-bit boundaries
        assert calculated_width >= stream_width_bits
        assert calculated_width % 8 == 0
        assert calculated_width == 64  # 8 elements × 8 bits = 64 bits
    
    def test_complete_hierarchy_consistency(self, transformer_input_interface):
        """Test complete hierarchy for transformer example."""
        interface = transformer_input_interface
        
        # Tensor: 512×768 = 393,216 total elements
        tensor_elements = np.prod(interface.tensor_dims)
        assert tensor_elements == 512 * 768
        
        # Block: 1×768 = 768 elements per block
        block_elements = np.prod(interface.block_dims)
        assert block_elements == 1 * 768
        
        # Stream: 1×64 = 64 elements per cycle
        stream_elements = np.prod(interface.stream_dims)
        assert stream_elements == 1 * 64
        
        # Hierarchy validation
        num_blocks = interface.get_num_blocks()
        assert np.prod(num_blocks) * block_elements == tensor_elements
        assert block_elements % stream_elements == 0
        
        # Timing calculation
        cycles_per_block = block_elements // stream_elements
        total_cycles = cycles_per_block * np.prod(num_blocks)
        assert cycles_per_block == 768 // 64  # 12 cycles per block
        assert total_cycles == 12 * 512      # 6144 total cycles
    
    def test_hierarchy_with_different_layouts(self, axiom_test_data):
        """Test hierarchy consistency across different tensor layouts."""
        for example in axiom_test_data["hierarchy_examples"]:
            tensor_dims = example["tensor_dims"]
            block_dims = example["block_dims"]
            stream_dims = example["stream_dims"]
            
            # Create test interface
            interface = self._create_test_interface(
                tensor_dims, block_dims, stream_dims
            )
            
            # Validate hierarchy at each level
            tensor_elements = np.prod(tensor_dims)
            block_elements = np.prod(block_dims)
            stream_elements = np.prod(stream_dims)
            
            # Level 1: Tensor to Block tiling
            num_blocks = interface.get_num_blocks()
            assert np.prod(num_blocks) * block_elements == tensor_elements
            
            # Level 2: Block to Stream tiling
            assert block_elements % stream_elements == 0
            
            # Level 3: Element consistency
            stream_width = interface.calculate_stream_width()
            expected_width = stream_elements * example["element_bitwidth"]
            assert stream_width >= expected_width
    
    def test_hierarchy_edge_cases(self, test_utils):
        """Test hierarchy with edge cases and boundary conditions."""
        
        # Edge case 1: Single element tensors
        single_element = test_utils.create_test_interface(
            tensor_dims=[1], block_dims=[1], stream_dims=[1],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        assert np.prod(single_element.tensor_dims) == 1
        assert np.prod(single_element.block_dims) == 1
        assert np.prod(single_element.stream_dims) == 1
        assert single_element.get_num_blocks() == [1]
        
        # Edge case 2: Large tensors with small blocks
        large_tensor = test_utils.create_test_interface(
            tensor_dims=[1024, 1024], block_dims=[1, 1], stream_dims=[1],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        total_elements = 1024 * 1024
        num_blocks = large_tensor.get_num_blocks()
        assert np.prod(num_blocks) == total_elements
        
        # Edge case 3: Equal tensor and block dimensions
        equal_dims = test_utils.create_test_interface(
            tensor_dims=[64, 64], block_dims=[64, 64], stream_dims=[8, 8],
            interface_type=DataflowInterfaceType.INPUT
        )
        
        assert equal_dims.get_num_blocks() == [1, 1]
        assert np.prod(equal_dims.block_dims) % np.prod(equal_dims.stream_dims) == 0
    
    def test_hierarchy_mathematical_properties(self, cnn_interfaces):
        """Test mathematical properties of the hierarchy."""
        
        for interface_name, interface in cnn_interfaces.items():
            # Property 1: Associativity of decomposition
            # tensor_elements = num_blocks × block_elements
            tensor_elements = np.prod(interface.tensor_dims)
            num_blocks = interface.get_num_blocks()
            block_elements = np.prod(interface.block_dims)
            
            assert tensor_elements == np.prod(num_blocks) * block_elements
            
            # Property 2: Stream decomposition
            # block_elements = cycles_per_block × stream_elements
            stream_elements = np.prod(interface.stream_dims)
            if block_elements % stream_elements == 0:
                cycles_per_block = block_elements // stream_elements
                assert cycles_per_block * stream_elements == block_elements
            
            # Property 3: Total processing time
            # total_cycles = num_blocks × cycles_per_block
            if block_elements % stream_elements == 0:
                cycles_per_block = block_elements // stream_elements
                total_cycles = np.prod(num_blocks) * cycles_per_block
                
                # This should equal tensor_elements / stream_elements
                expected_cycles = tensor_elements // stream_elements
                assert total_cycles == expected_cycles
    
    def _create_test_interface(self, tensor_dims, block_dims, stream_dims):
        """Helper to create test interface."""
        from brainsmith.dataflow.core.dataflow_interface import DataflowDataType
        
        dtype = DataflowDataType("INT", 8, True, "INT8")
        return DataflowInterface(
            name="test",
            interface_type=DataflowInterfaceType.INPUT,
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            stream_dims=stream_dims,
            dtype=dtype
        )
    
    def test_hierarchy_validation_failures(self, test_utils):
        """Test that hierarchy validation catches invalid configurations."""
        
        # Invalid case 1: Block larger than tensor
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[32], block_dims=[64], stream_dims=[4],
                interface_type=DataflowInterfaceType.INPUT
            )
        
        # Invalid case 2: Stream not tiling into block
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[64], block_dims=[15], stream_dims=[4],
                interface_type=DataflowInterfaceType.INPUT
            )
        
        # Invalid case 3: Non-divisible relationships
        with pytest.raises(ValueError):
            test_utils.create_test_interface(
                tensor_dims=[100], block_dims=[30], stream_dims=[7],
                interface_type=DataflowInterfaceType.INPUT
            )