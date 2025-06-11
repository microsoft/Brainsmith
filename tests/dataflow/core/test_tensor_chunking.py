"""
Tests for simplified tensor chunking system with per-interface strategies.

Tests the new architecture where each interface has its own chunking strategy
instead of a global override system.
"""

import pytest
from unittest.mock import Mock
from brainsmith.dataflow.core.block_chunking import TensorChunking
from brainsmith.dataflow.core.block_chunking import (
    default_chunking, index_chunking, last_dim_chunking, spatial_chunking,
    DefaultChunkingStrategy, IndexBasedChunkingStrategy, FullTensorChunkingStrategy,
    ChunkingType
)
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType


class TestSimplifiedTensorChunking:
    """Test the simplified tensor chunking system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.chunker = TensorChunking()
        self.mock_onnx_node = Mock()
        self.mock_onnx_node.input = ["input_tensor"]
        self.mock_onnx_node.output = ["output_tensor"]
    
    def test_compute_chunking_for_interface_with_default_strategy(self):
        """Test chunking computation with default strategy."""
        interface_metadata = InterfaceMetadata(
            name="test_interface",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=default_chunking()
        )
        
        # Mock shape extraction to return known shape
        self.chunker.extract_tensor_shape_from_input = Mock(return_value=[1, 8, 32, 32])
        
        qDim, tDim = self.chunker.compute_chunking_for_interface(interface_metadata, self.mock_onnx_node)
        
        # In new architecture: qDim preserves original shape, tDim = processing shape
        assert qDim == [1, 8, 32, 32]  # Original tensor shape preserved
        assert tDim == [1, 8, 32, 32]  # Default: process entire tensor
    
    def test_compute_chunking_for_interface_with_index_strategy(self):
        """Test chunking computation with index-based strategy."""
        interface_metadata = InterfaceMetadata(
            name="test_interface",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=index_chunking(-1, [16])
        )
        
        # Mock specific tensor shape extraction
        self.chunker.extract_tensor_shape_from_input = Mock(return_value=[1, 8, 32, 32])
        
        qDim, tDim = self.chunker.compute_chunking_for_interface(interface_metadata, self.mock_onnx_node)
        
        # In new architecture: qDim preserves original, tDim shows chunk size
        assert qDim == [1, 8, 32, 32]  # Original tensor shape preserved
        assert tDim[3] == 16  # Last dimension chunked to size 16
        assert tDim[:3] == [1, 8, 32]  # Other dimensions unchanged
    
    def test_extract_tensor_shape_from_input(self):
        """Test tensor shape extraction from ONNX node."""
        interface_name = "in0_V_data_V"
        
        # Mock model wrapper with shape extraction
        mock_wrapper = Mock()
        mock_wrapper.get_tensor_shape.return_value = [1, 8, 32, 32]
        self.chunker.set_model_wrapper(mock_wrapper)
        
        shape = self.chunker.extract_tensor_shape_from_input(interface_name, self.mock_onnx_node)
        assert shape == [1, 8, 32, 32]
    
    def test_extract_tensor_shape_fallback(self):
        """Test tensor shape extraction requires ModelWrapper."""
        interface_name = "in0_V_data_V"
        
        # No model wrapper - should raise error (strict tensor shape requirement)
        with pytest.raises(RuntimeError, match="ModelWrapper required"):
            self.chunker.extract_tensor_shape_from_input(interface_name, self.mock_onnx_node)
    
    def test_interface_name_mapping(self):
        """Test interface name to input index mapping."""
        assert self.chunker._map_interface_to_input_index("in0_V_data_V") == 0
        assert self.chunker._map_interface_to_input_index("in1_V_data_V") == 1
        assert self.chunker._map_interface_to_input_index("weights") == 1
        assert self.chunker._map_interface_to_input_index("bias") == 2
        assert self.chunker._map_interface_to_input_index("unknown") == 0
    
    def test_default_shapes_for_interfaces(self):
        """Test that default shapes are deprecated and require ModelWrapper."""
        # Default shapes are deprecated - should raise error
        with pytest.raises(RuntimeError, match="Default shapes not allowed"):
            self.chunker._get_default_shape_for_interface("weights")
        with pytest.raises(RuntimeError, match="Default shapes not allowed"):
            self.chunker._get_default_shape_for_interface("bias")
        with pytest.raises(RuntimeError, match="Default shapes not allowed"):
            self.chunker._get_default_shape_for_interface("config")


class TestChunkingStrategies:
    """Test individual chunking strategies."""
    
    def test_default_chunking_strategy(self):
        """Test default chunking strategy."""
        strategy = default_chunking()
        
        assert strategy.chunking_type == ChunkingType.DEFAULT
        
        # In new architecture: qDim preserves original, tDim = processing shape
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim == [1, 8, 32, 32]  # Default: process entire tensor
        
        # Test with 2D tensor
        qDim, tDim = strategy.compute_chunking([128, 64], "test_interface")
        assert qDim == [128, 64]  # Original shape preserved
        assert tDim == [128, 64]  # Default: process entire tensor
    
    def test_index_based_chunking_strategy(self):
        """Test index-based chunking strategy."""
        strategy = index_chunking(-1, [16])
        
        assert strategy.chunking_type == ChunkingType.INDEX_BASED
        assert strategy.start_index == -1
        assert strategy.shape == [16]
        
        # In new architecture: qDim preserves original, tDim shows chunk size
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[3] == 16  # Last dimension chunked to size 16
    
    def test_index_based_chunking_with_full_tensor(self):
        """Test index-based chunking with full tensor shape."""
        strategy = index_chunking(0, [":"])
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # Full tensor - no chunking, both preserve original
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim == [1, 8, 32, 32]  # Process entire tensor
    
    def test_index_based_chunking_multidimensional(self):
        """Test index-based chunking with multi-dimensional shapes."""
        strategy = index_chunking(2, [16, 16])  # Spatial chunking
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # In new architecture: qDim preserves original, tDim shows chunk sizes
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[2] == 16  # Spatial dimension chunked to 16
        assert tDim[3] == 16  # Spatial dimension chunked to 16
        assert tDim[:2] == [1, 8]  # Unchanged
    
    def test_index_based_chunking_negative_indices(self):
        """Test negative index handling."""
        strategy = index_chunking(-2, [16])  # Second to last dimension
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # In new architecture: qDim preserves original, tDim shows chunk size
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[2] == 16  # Second to last dimension chunked to 16
    
    def test_index_based_chunking_out_of_bounds(self):
        """Test handling of out-of-bounds indices."""
        strategy = index_chunking(10, [16])  # Invalid index
        
        with pytest.raises(ValueError, match="out of bounds"):
            strategy.compute_chunking([1, 8, 32, 32], "test_interface")
    
    def test_index_based_chunking_string_parameters(self):
        """Test string parameter resolution."""
        strategy = index_chunking(1, ["tdim1", "tdim2"])
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test_interface")
        
        # String parameters resolve to 1 (HWKG resolves them)
        assert tDim[1] == 1
        assert tDim[2] == 1


class TestConvenienceFunctions:
    """Test convenience chunking functions."""
    
    def test_last_dim_chunking(self):
        """Test last dimension chunking convenience function."""
        strategy = last_dim_chunking(8)
        
        assert isinstance(strategy, IndexBasedChunkingStrategy)
        assert strategy.start_index == -1
        assert strategy.shape == [8]
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test")
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[3] == 8  # Last dimension chunked to 8
    
    def test_spatial_chunking(self):
        """Test spatial chunking convenience function."""
        strategy = spatial_chunking(16, 16)
        
        assert isinstance(strategy, IndexBasedChunkingStrategy)
        assert strategy.start_index == 2  # Spatial dimensions start at index 2 in NCHW
        assert strategy.shape == [16, 16]
        
        qDim, tDim = strategy.compute_chunking([1, 8, 32, 32], "test")
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[2] == 16  # Spatial dimension chunked to 16
        assert tDim[3] == 16  # Spatial dimension chunked to 16


class TestChunkingStrategyValidation:
    """Test chunking strategy validation."""
    
    def test_invalid_start_index(self):
        """Test invalid start index validation."""
        with pytest.raises(ValueError, match="start_index must be an integer"):
            index_chunking("invalid", [16])
    
    def test_invalid_shape(self):
        """Test invalid shape validation."""
        with pytest.raises(ValueError, match="shape must be a list"):
            index_chunking(0, "invalid")
    
    def test_edge_cases(self):
        """Test edge cases in chunking."""
        strategy = default_chunking()
        
        # Empty tensor shape
        qDim, tDim = strategy.compute_chunking([], "test")
        assert qDim == [1]
        assert tDim == [1]
        
        # Single dimension
        qDim, tDim = strategy.compute_chunking([256], "test")
        assert qDim == [256]  # Original shape preserved
        assert tDim == [256]  # Default: process entire tensor


class TestIntegrationWithInterfaceMetadata:
    """Test integration between chunking strategies and interface metadata."""
    
    def test_interface_with_default_strategy(self):
        """Test interface metadata with default chunking strategy."""
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)]
            # chunking_strategy defaults to default_chunking()
        )
        
        assert isinstance(metadata.chunking_strategy, DefaultChunkingStrategy)
        assert metadata.chunking_strategy.chunking_type == ChunkingType.DEFAULT
    
    def test_interface_with_custom_strategy(self):
        """Test interface metadata with custom chunking strategy."""
        custom_strategy = index_chunking(-1, [32])
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=custom_strategy
        )
        
        assert metadata.chunking_strategy == custom_strategy
        assert isinstance(metadata.chunking_strategy, IndexBasedChunkingStrategy)
    
    def test_chunking_delegation(self):
        """Test chunking computation delegation to interface strategy."""
        chunker = TensorChunking()
        
        # Create interface with specific strategy
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=last_dim_chunking(16)
        )
        
        # Mock ONNX node
        mock_node = Mock()
        mock_node.input = ["tensor"]
        
        # Mock shape extraction to return known shape
        chunker.extract_tensor_shape_from_input = Mock(return_value=[1, 8, 32, 32])
        
        # Compute chunking - should delegate to interface strategy
        qDim, tDim = chunker.compute_chunking_for_interface(metadata, mock_node)
        
        # Verify delegation worked - new architecture
        assert qDim == [1, 8, 32, 32]  # Original shape preserved
        assert tDim[3] == 16  # Last dimension chunked to 16