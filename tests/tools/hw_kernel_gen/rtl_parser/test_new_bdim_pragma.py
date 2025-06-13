#!/usr/bin/env python3
"""
Tests for the new simplified BDIM pragma system.

Tests the new BDIM pragma format and BlockChunkingStrategy functionality.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../../..'))

from brainsmith.tools.hw_kernel_gen.rtl_parser.data import BDimPragma, PragmaType
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestNewBDimPragmaFormat:
    """Test the new simplified BDIM pragma format."""
    
    def test_basic_shape_parsing(self):
        """Test basic shape parsing with parameter name."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[PE]"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["block_shape"] == ["PE"]
        assert pragma.parsed_data["rindex"] == 0
    
    def test_multi_dimension_shape(self):
        """Test multi-dimension shape parsing with parameter names."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[SIMD,PE]"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["block_shape"] == ["SIMD", "PE"]
        assert pragma.parsed_data["rindex"] == 0
    
    def test_full_dimension_colon(self):
        """Test ':' for full dimensions."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[:,:,PE]"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["block_shape"] == [":", ":", "PE"]
        assert pragma.parsed_data["rindex"] == 0
    
    def test_parameter_names(self):
        """Test parameter names in shape."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["weights", "[TILE_SIZE,PE]"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "weights"
        assert pragma.parsed_data["block_shape"] == ["TILE_SIZE", "PE"]
        assert pragma.parsed_data["rindex"] == 0
    
    def test_rindex_parameter(self):
        """Test RINDEX parameter."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[SIMD,PE]", "RINDEX=2"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["block_shape"] == ["SIMD", "PE"]
        assert pragma.parsed_data["rindex"] == 2
    
    def test_mixed_shape_elements(self):
        """Test mixed shape elements."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[PE,:,SIMD]", "RINDEX=1"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["block_shape"] == ["PE", ":", "SIMD"]
        assert pragma.parsed_data["rindex"] == 1


class TestNewBDimPragmaValidation:
    """Test validation and error handling."""
    
    def test_missing_shape(self):
        """Test error when shape is missing."""
        with pytest.raises(Exception):  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0"],
                line_number=10
            )
    
    def test_invalid_shape_format(self):
        """Test error for invalid shape format."""
        with pytest.raises(Exception):  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "8,4"],  # Missing brackets
                line_number=10
            )
    
    def test_empty_shape(self):
        """Test error for empty shape."""
        with pytest.raises(Exception):  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[]"],
                line_number=10
            )
    
    def test_invalid_rindex(self):
        """Test error for invalid RINDEX."""
        with pytest.raises(Exception):  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[16]", "RINDEX=-1"],
                line_number=10
            )
    
    def test_invalid_shape_element(self):
        """Test error for invalid shape element."""
        with pytest.raises(Exception):  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[PE*2+1]"],  # Invalid expression
                line_number=10
            )
    
    def test_magic_numbers_rejected(self):
        """Test that magic numbers are explicitly rejected."""
        # Single magic number
        with pytest.raises(Exception) as exc_info:  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[16]"],  # Magic number
                line_number=10
            )
        assert "Magic numbers not allowed" in str(exc_info.value)
        
        # Multiple magic numbers
        with pytest.raises(Exception) as exc_info:  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[8,4]"],  # Magic numbers
                line_number=10
            )
        assert "Magic numbers not allowed" in str(exc_info.value)
        
        # Mixed with valid parameter
        with pytest.raises(Exception) as exc_info:  # Should raise PragmaError
            BDimPragma(
                type=PragmaType.BDIM,
                inputs=["in0", "[PE,16]"],  # Valid param + magic number
                line_number=10
            )
        assert "Magic numbers not allowed" in str(exc_info.value)


class TestBlockChunkingStrategy:
    """Test the new BlockChunkingStrategy."""
    
    def test_basic_chunking(self):
        """Test basic block chunking with parameter name."""
        strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, 64, 128, "PE"]  # Last dimension chunked with parameter
    
    def test_multi_dimension_chunking(self):
        """Test multi-dimension chunking with parameter names."""
        strategy = BlockChunkingStrategy(block_shape=["SIMD", "PE"], rindex=0)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, 64, "SIMD", "PE"]  # Last two dimensions chunked with parameters
    
    def test_rindex_positioning(self):
        """Test RINDEX positioning with parameter names."""
        strategy = BlockChunkingStrategy(block_shape=["SIMD", "PE"], rindex=1)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, "SIMD", "PE", 256]  # Middle dimensions chunked with parameters
    
    def test_full_dimension_colon(self):
        """Test ':' preserves full dimension."""
        strategy = BlockChunkingStrategy(block_shape=[":", "PE"], rindex=0)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, 64, 128, "PE"]  # Third dim preserved, fourth chunked with parameter
    
    def test_parameter_name_preservation(self):
        """Test parameter names are preserved."""
        strategy = BlockChunkingStrategy(block_shape=["PE", ":"], rindex=0)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, 64, "PE", 256]  # Parameter name preserved
    
    def test_parameter_resolution(self):
        """Test that parameter names are preserved for runtime resolution."""
        strategy = BlockChunkingStrategy(block_shape=["LARGE_PARAM"], rindex=0)
        tensor_shape = [32, 64, 128, 256]
        
        tensor_dims, block_dims = strategy.compute_chunking(tensor_shape, "test_interface")
        
        assert tensor_dims == [32, 64, 128, 256]
        assert block_dims == [32, 64, 128, "LARGE_PARAM"]  # Parameter preserved for runtime resolution
    
    def test_rindex_bounds_checking(self):
        """Test RINDEX bounds checking."""
        strategy = BlockChunkingStrategy(block_shape=["SIMD", "PE"], rindex=10)  # Too large
        tensor_shape = [32, 64, 128, 256]
        
        with pytest.raises(ValueError):
            strategy.compute_chunking(tensor_shape, "test_interface")


class TestBDimPragmaIntegration:
    """Test BDIM pragma integration with InterfaceMetadata."""
    
    def test_pragma_application(self):
        """Test BDIM pragma application to InterfaceMetadata."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["test_interface", "[SIMD,PE]", "RINDEX=1"],
            line_number=10
        )
        
        base_metadata = InterfaceMetadata(
            name="test_interface",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[],
            chunking_strategy=None
        )
        
        # Test that pragma applies to interface
        assert pragma.applies_to_interface_metadata(base_metadata)
        
        # Apply pragma
        updated_metadata = pragma.apply_to_metadata(base_metadata)
        
        # Check that chunking strategy was updated
        assert isinstance(updated_metadata.chunking_strategy, BlockChunkingStrategy)
        assert updated_metadata.chunking_strategy.block_shape == ["SIMD", "PE"]
        assert updated_metadata.chunking_strategy.rindex == 1
    
    def test_pragma_name_matching(self):
        """Test pragma name matching patterns."""
        pragma = BDimPragma(
            type=PragmaType.BDIM,
            inputs=["in0", "[PE]"],
            line_number=10
        )
        
        # Test exact match
        metadata1 = InterfaceMetadata(
            name="in0",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[],
            chunking_strategy=None
        )
        assert pragma.applies_to_interface_metadata(metadata1)
        
        # Test AXI naming pattern match
        metadata2 = InterfaceMetadata(
            name="in0_V_data_V",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[],
            chunking_strategy=None
        )
        assert pragma.applies_to_interface_metadata(metadata2)
        
        # Test non-matching
        metadata3 = InterfaceMetadata(
            name="out0",
            interface_type=InterfaceType.OUTPUT,
            allowed_datatypes=[],
            chunking_strategy=None
        )
        assert not pragma.applies_to_interface_metadata(metadata3)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])