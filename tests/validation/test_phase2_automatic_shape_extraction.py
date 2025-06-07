"""
Comprehensive Phase 2 validation tests for automatic tensor shape extraction 
and zero-configuration features.

This module validates all Phase 2 functionality including automatic shape extraction,
ModelWrapper integration, smart layout inference, HWKG pragma integration, and 
zero-configuration FINN workflow.
"""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock

from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint, DataflowInterfaceType
from brainsmith.dataflow.core.chunking_strategy import (
    default_chunking, index_chunking, last_dim_chunking, spatial_chunking, FullTensorChunkingStrategy
)
from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.tensor_chunking import TensorChunking
from brainsmith.tools.hw_kernel_gen.pragma_to_strategy import PragmaToStrategyConverter


class MockModelWrapper:
    """Mock FINN ModelWrapper for testing."""
    
    def __init__(self, input_shapes, tensor_names=None):
        self.input_shapes = input_shapes
        self.tensor_names = tensor_names or [f"input_{i}" for i in range(len(input_shapes))]
        self.shape_map = dict(zip(self.tensor_names, input_shapes))
        
        self.graph = Mock()
        self.graph.input = []
        
        for i, (name, shape) in enumerate(zip(self.tensor_names, input_shapes)):
            input_mock = Mock()
            input_mock.name = name
            self.graph.input.append(input_mock)
    
    def get_tensor_shape(self, tensor_name):
        """Get tensor shape by name."""
        return self.shape_map.get(tensor_name, [1, 8, 32, 32])  # Return mapped shape or default


class TestAutomaticShapeExtraction:
    """Test automatic tensor shape extraction functionality."""
    
    def test_extract_shape_with_model_wrapper(self):
        """Test shape extraction with ModelWrapper integration."""
        # Setup
        chunker = TensorChunking()
        model_wrapper = MockModelWrapper(
            [[1, 8, 32, 32], [64, 8, 3, 3], [64]],
            ["input_0", "input_1", "input_2"]
        )
        chunker.set_model_wrapper(model_wrapper)
        
        # Create mock ONNX node
        onnx_node = Mock()
        onnx_node.input = ["input_0", "input_1", "input_2"]
        
        # Test shape extraction for different interfaces
        input_shape = chunker.extract_tensor_shape_from_input("in0_V_data_V", onnx_node)
        weight_shape = chunker.extract_tensor_shape_from_input("weights", onnx_node)
        bias_shape = chunker.extract_tensor_shape_from_input("bias", onnx_node)
        
        assert input_shape == [1, 8, 32, 32]
        assert weight_shape == [64, 8, 3, 3]
        assert bias_shape == [64]
    
    def test_extract_shape_without_model_wrapper(self):
        """Test shape extraction fallback without ModelWrapper."""
        chunker = TensorChunking()
        onnx_node = Mock()
        onnx_node.input = ["input_tensor"]
        
        # Should fall back to defaults
        input_shape = chunker.extract_tensor_shape_from_input("in0_V_data_V", onnx_node)
        weight_shape = chunker.extract_tensor_shape_from_input("weights", onnx_node)
        bias_shape = chunker.extract_tensor_shape_from_input("bias", onnx_node)
        
        assert input_shape == [1, 8, 32, 32]  # Default input
        assert weight_shape == [64, 64]       # Default weight
        assert bias_shape == [64]             # Default bias
    
    def test_interface_name_mapping(self):
        """Test interface name to input index mapping."""
        chunker = TensorChunking()
        
        # Test various interface naming patterns
        assert chunker._map_interface_to_input_index("in0_V_data_V") == 0
        assert chunker._map_interface_to_input_index("in1_V_data_V") == 1
        assert chunker._map_interface_to_input_index("in2_V_data_V") == 2
        assert chunker._map_interface_to_input_index("weights") == 1
        assert chunker._map_interface_to_input_index("bias") == 2
        assert chunker._map_interface_to_input_index("input") == 0
        assert chunker._map_interface_to_input_index("unknown_interface") == 0  # Default
    
    def test_shape_extraction_edge_cases(self):
        """Test shape extraction with edge cases."""
        chunker = TensorChunking()
        
        # Test with empty model wrapper
        chunker.set_model_wrapper(None)
        onnx_node = Mock()
        onnx_node.input = []
        
        shape = chunker.extract_tensor_shape_from_input("test_interface", onnx_node)
        assert len(shape) == 4  # Should get default shape
        
        # Test with model wrapper that raises exceptions
        broken_wrapper = Mock()
        broken_wrapper.get_tensor_shape.side_effect = Exception("Connection failed")
        chunker.set_model_wrapper(broken_wrapper)
        
        shape = chunker.extract_tensor_shape_from_input("test_interface", onnx_node)
        assert len(shape) >= 1  # Should fall back gracefully


class TestSmartLayoutInference:
    """Test smart layout inference functionality."""
    
    def test_layout_inference_for_common_shapes(self):
        """Test layout inference for common tensor shapes."""
        chunker = TensorChunking()
        
        # Test 4D tensor (NCHW)
        layout = chunker.infer_layout_from_shape([1, 8, 32, 32])
        assert layout == "NCHW"
        
        # Test 3D tensor (CHW)
        layout = chunker.infer_layout_from_shape([8, 32, 32])
        assert layout == "CHW"
        
        # Test 2D tensor (NC)
        layout = chunker.infer_layout_from_shape([1, 1000])
        assert layout == "NC"
        
        # Test 1D tensor (C)
        layout = chunker.infer_layout_from_shape([256])
        assert layout == "C"
        
        # Test unusual dimensions
        layout = chunker.infer_layout_from_shape([1, 2, 3, 4, 5])
        assert layout == "DIM5"
    
    def test_layout_aware_chunking(self):
        """Test layout-aware chunking strategies."""
        chunker = TensorChunking()
        
        # Test NCHW layout chunking
        qDim, tDim = chunker.get_layout_aware_chunking([1, 8, 32, 32], "NCHW")
        assert len(qDim) == 4
        assert len(tDim) == 4
        assert qDim[3] == 32  # Stream on width
        assert tDim[3] == 1   # One element per cycle
        
        # Test CHW layout chunking
        qDim, tDim = chunker.get_layout_aware_chunking([8, 32, 32], "CHW")
        assert len(qDim) == 3
        assert qDim[2] == 32  # Stream on width
        assert tDim[2] == 1   # One element per cycle
        
        # Test NC layout chunking
        qDim, tDim = chunker.get_layout_aware_chunking([64, 1000], "NC")
        assert len(qDim) == 2
        assert qDim[1] == 1000  # Stream on channels
        assert tDim[1] == 1     # One element per cycle
        
        # Test unknown layout (conservative default)
        qDim, tDim = chunker.get_layout_aware_chunking([10, 20, 30], "UNKNOWN")
        assert qDim == [1, 1, 1]      # Conservative chunking
        assert tDim == [10, 20, 30]   # Full tensor


class TestModelWrapperIntegration:
    """Test ModelWrapper integration with AutoHWCustomOp."""
    
    def test_model_wrapper_setting_and_invalidation(self):
        """Test ModelWrapper setting and model invalidation."""
        # Create AutoHWCustomOp
        onnx_node = Mock()
        onnx_node.input = ["input"]
        onnx_node.output = ["output"]
        
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=default_chunking()
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        
        # Initially no ModelWrapper
        assert op.get_model_wrapper() is None
        
        # Build model once
        model1 = op.dataflow_model
        assert model1 is not None
        
        # Set ModelWrapper - should invalidate model
        model_wrapper = MockModelWrapper([[1, 16, 64, 64]])
        op.set_model_wrapper(model_wrapper)
        
        assert op.get_model_wrapper() is model_wrapper
        
        # Should rebuild with new shape information
        model2 = op.dataflow_model
        assert model2 is not model1  # Should be different object
    
    def test_automatic_shape_extraction_in_dataflow_building(self):
        """Test automatic shape extraction during DataflowModel building."""
        # Setup with ModelWrapper containing specific shapes
        model_wrapper = MockModelWrapper(
            [[2, 16, 128, 128]],  # Different from default
            ["input_tensor"]      # Match ONNX node input
        )
        
        onnx_node = Mock()
        onnx_node.input = ["input_tensor"]
        onnx_node.output = ["output"]
        
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=last_dim_chunking(16)
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        op.set_model_wrapper(model_wrapper)
        
        # Build DataflowModel - should extract actual shapes
        dataflow_model = op.dataflow_model
        
        # Verify shapes were extracted and stored
        interface = dataflow_model.interfaces["in0_V_data_V"]
        assert hasattr(interface, '_tensor_shape')
        assert interface._tensor_shape == [2, 16, 128, 128]  # Extracted shape
        assert hasattr(interface, '_inferred_layout')
        assert interface._inferred_layout == "NCHW"  # Inferred layout


class TestHWKGPragmaIntegration:
    """Test HWKG pragma integration and conversion."""
    
    def test_pragma_parsing_index_based(self):
        """Test parsing of index-based TDIM pragmas."""
        converter = PragmaToStrategyConverter()
        
        # Test simple index pragma
        pragma_str = "@brainsmith TDIM in0_V_data_V -1 [16]"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['interface_name'] == 'in0_V_data_V'
        assert parsed['type'] == 'index'
        assert parsed['start_index'] == -1
        assert parsed['shape'] == [16]
        
        # Test multi-dimensional index pragma
        pragma_str = "@brainsmith TDIM weights 2 [3, 3]"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['interface_name'] == 'weights'
        assert parsed['type'] == 'index'
        assert parsed['start_index'] == 2
        assert parsed['shape'] == [3, 3]
    
    def test_pragma_parsing_spatial(self):
        """Test parsing of spatial TDIM pragmas."""
        converter = PragmaToStrategyConverter()
        
        # Test spatial pragma
        pragma_str = "@brainsmith TDIM weights spatial 8x8"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['interface_name'] == 'weights'
        assert parsed['type'] == 'spatial'
        assert parsed['height'] == 8
        assert parsed['width'] == 8
        
        # Test square spatial pragma
        pragma_str = "@brainsmith TDIM weights spatial 16"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['type'] == 'spatial'
        assert parsed['height'] == 16
        assert parsed['width'] == 16
    
    def test_pragma_parsing_special_cases(self):
        """Test parsing of special case pragmas."""
        converter = PragmaToStrategyConverter()
        
        # Test none pragma
        pragma_str = "@brainsmith TDIM bias none"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['interface_name'] == 'bias'
        assert parsed['type'] == 'none'
        
        # Test last_dim pragma
        pragma_str = "@brainsmith TDIM output last_dim 32"
        parsed = converter.parse_enhanced_tdim_pragma(pragma_str)
        
        assert parsed['interface_name'] == 'output'
        assert parsed['type'] == 'last_dim'
        assert parsed['chunk_size'] == 32
    
    def test_pragma_to_strategy_conversion(self):
        """Test conversion of parsed pragmas to chunking strategies."""
        converter = PragmaToStrategyConverter()
        
        # Test index-based conversion
        pragma_data = {
            'type': 'index',
            'start_index': -1,
            'shape': [16]
        }
        strategy = converter.convert_tdim_pragma(pragma_data)
        assert strategy.start_index == -1
        assert strategy.shape == [16]
        
        # Test spatial conversion
        pragma_data = {
            'type': 'spatial',
            'height': 8,
            'width': 8
        }
        strategy = converter.convert_tdim_pragma(pragma_data)
        assert strategy.start_index == 2  # Spatial chunking starts at dim 2
        assert strategy.shape == [8, 8]
        
        # Test none conversion
        pragma_data = {'type': 'none'}
        strategy = converter.convert_tdim_pragma(pragma_data)
        assert isinstance(strategy, FullTensorChunkingStrategy)
    
    def test_pragma_error_handling(self):
        """Test error handling in pragma parsing."""
        converter = PragmaToStrategyConverter()
        
        # Test invalid pragma format
        with pytest.raises(ValueError, match="Invalid TDIM pragma format"):
            converter.parse_enhanced_tdim_pragma("invalid pragma")
        
        # Test missing parameters
        with pytest.raises(ValueError, match="Index-based TDIM pragma missing shape"):
            converter.parse_enhanced_tdim_pragma("@brainsmith TDIM in0 -1")
        
        # Test unknown pragma type
        with pytest.raises(ValueError, match="Unknown TDIM pragma type"):
            converter.parse_enhanced_tdim_pragma("@brainsmith TDIM in0 unknown_type")


class TestZeroConfigurationWorkflow:
    """Test zero-configuration FINN workflow integration."""
    
    def test_zero_config_node_creation(self):
        """Test zero-configuration node creation workflow."""
        # Simulate ONNX node creation (zero config)
        onnx_node = Mock()
        onnx_node.op_type = "ThresholdingAxi"
        onnx_node.input = ["input_tensor"]
        onnx_node.output = ["output_tensor"]
        onnx_node.attribute = []
        
        # Create interface metadata with strategies (from HWKG pragma parsing)
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=index_chunking(-1, [16])
            ),
            InterfaceMetadata(
                name="out0_V_data_V", 
                interface_type=DataflowInterfaceType.OUTPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=last_dim_chunking(32)
            )
        ]
        
        # Create AutoHWCustomOp (should work without manual configuration)
        op = AutoHWCustomOp(onnx_node, metadata)
        
        # Set ModelWrapper for shape extraction
        model_wrapper = MockModelWrapper([[1, 8, 32, 32]])
        op.set_model_wrapper(model_wrapper)
        
        # Access DataflowModel - should build automatically
        dataflow_model = op.dataflow_model
        
        # Verify automatic configuration
        assert len(dataflow_model.interfaces) == 2
        assert "in0_V_data_V" in dataflow_model.interfaces
        assert "out0_V_data_V" in dataflow_model.interfaces
        
        # Verify shapes were extracted and chunking applied
        input_interface = dataflow_model.interfaces["in0_V_data_V"]
        assert hasattr(input_interface, '_tensor_shape')
        assert input_interface._tensor_shape == [1, 8, 32, 32]
    
    def test_backward_compatibility_with_manual_config(self):
        """Test that manual configuration still works alongside automatic features."""
        onnx_node = Mock()
        onnx_node.input = ["input"]
        onnx_node.output = ["output"]
        
        # Create with explicit chunking strategy that should produce different results
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=index_chunking(-1, [8])  # Explicit strategy: chunk last dim with size 8
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        dataflow_model = op.dataflow_model
        
        # Should use explicit strategy, not defaults
        interface = dataflow_model.interfaces["in0_V_data_V"]
        # With index_chunking(-1, [8]) on default shape [1,8,32,32], expect different chunking
        assert interface.tDim[-1] == 8  # Should chunk last dimension to size 8
        assert interface.qDim[-1] == 4  # Should have 32/8 = 4 chunks
    
    def test_mixed_automatic_and_manual_interfaces(self):
        """Test mixing automatic and manually configured interfaces."""
        onnx_node = Mock()
        onnx_node.input = ["input", "weights"]
        onnx_node.output = ["output"]
        
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=default_chunking()  # Automatic
            ),
            InterfaceMetadata(
                name="weights",
                interface_type=DataflowInterfaceType.WEIGHT,
                allowed_datatypes=[DataTypeConstraint(finn_type='INT8', bit_width=8)],
                chunking_strategy=spatial_chunking(3, 3)  # Explicit
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        model_wrapper = MockModelWrapper([[1, 8, 32, 32], [64, 8, 3, 3]])
        op.set_model_wrapper(model_wrapper)
        
        dataflow_model = op.dataflow_model
        
        # Both should work correctly
        assert len(dataflow_model.interfaces) == 2
        assert "in0_V_data_V" in dataflow_model.interfaces
        assert "weights" in dataflow_model.interfaces


class TestPerformanceValidation:
    """Test performance aspects of Phase 2 features."""
    
    def test_lazy_building_performance(self):
        """Test that lazy building provides performance benefits."""
        import time
        
        onnx_node = Mock()
        onnx_node.input = ["input"] * 20  # Many inputs
        onnx_node.output = ["output"]
        
        # Create many interfaces
        metadata = []
        for i in range(20):
            metadata.append(
                InterfaceMetadata(
                    name=f"interface_{i}",
                    interface_type=DataflowInterfaceType.INPUT,
                    allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                    chunking_strategy=index_chunking(-1, [16])
                )
            )
        
        # Measure construction time
        start_time = time.time()
        op = AutoHWCustomOp(onnx_node, metadata)
        construction_time = time.time() - start_time
        
        # Should be very fast (lazy)
        assert construction_time < 0.01  # Less than 10ms
        
        # Measure first build time
        start_time = time.time()
        dataflow_model = op.dataflow_model
        build_time = time.time() - start_time
        
        # Should still be reasonable
        assert build_time < 1.0  # Less than 1 second
        
        # Measure cached access time
        start_time = time.time()
        dataflow_model2 = op.dataflow_model
        cached_time = time.time() - start_time
        
        # Should be instant (cached)
        assert cached_time < 0.001  # Less than 1ms
        assert dataflow_model2 is dataflow_model  # Same object
    
    def test_model_wrapper_change_invalidation(self):
        """Test that changing ModelWrapper properly invalidates cache."""
        onnx_node = Mock()
        onnx_node.input = ["input"]
        onnx_node.output = ["output"]
        
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=default_chunking()
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        
        # Build initial model
        model1 = op.dataflow_model
        
        # Change ModelWrapper with matching tensor name
        model_wrapper = MockModelWrapper([[2, 16, 64, 64]], ["input"])
        op.set_model_wrapper(model_wrapper)
        
        # Should rebuild with new shapes
        model2 = op.dataflow_model
        assert model2 is not model1  # Different objects
        
        # Verify new shape was used
        interface = model2.interfaces["in0_V_data_V"]
        assert hasattr(interface, '_tensor_shape')
        assert interface._tensor_shape == [2, 16, 64, 64]


class TestEdgeCasesAndErrorHandling:
    """Test edge cases and error handling for Phase 2 features."""
    
    def test_empty_interfaces(self):
        """Test handling of empty interface lists."""
        onnx_node = Mock()
        onnx_node.input = []
        onnx_node.output = []
        
        with pytest.raises(ValueError, match="No interface metadata available"):
            AutoHWCustomOp(onnx_node, [])
    
    def test_malformed_model_wrapper(self):
        """Test handling of malformed ModelWrapper."""
        onnx_node = Mock()
        onnx_node.input = ["input"]
        onnx_node.output = ["output"]
        
        metadata = [
            InterfaceMetadata(
                name="in0_V_data_V",
                interface_type=DataflowInterfaceType.INPUT,
                allowed_datatypes=[DataTypeConstraint(finn_type='UINT8', bit_width=8)],
                chunking_strategy=default_chunking()
            )
        ]
        
        op = AutoHWCustomOp(onnx_node, metadata)
        
        # Set broken ModelWrapper
        broken_wrapper = Mock()
        broken_wrapper.get_tensor_shape.side_effect = Exception("Broken")
        op.set_model_wrapper(broken_wrapper)
        
        # Should still work with fallback
        dataflow_model = op.dataflow_model
        assert len(dataflow_model.interfaces) == 1
    
    def test_inconsistent_interface_names(self):
        """Test handling of inconsistent interface names."""
        chunker = TensorChunking()
        
        # Test with interface name that doesn't match ONNX inputs
        onnx_node = Mock()
        onnx_node.input = ["different_name"]
        
        # Should fall back gracefully
        shape = chunker.extract_tensor_shape_from_input("nonexistent_interface", onnx_node)
        assert len(shape) >= 1  # Should get some default shape
    
    def test_shape_extraction_with_none_values(self):
        """Test shape extraction when ModelWrapper returns None."""
        chunker = TensorChunking()
        
        model_wrapper = Mock()
        model_wrapper.get_tensor_shape.return_value = None
        model_wrapper.graph.input = [Mock()]
        model_wrapper.graph.input[0].name = "input_0"
        chunker.set_model_wrapper(model_wrapper)
        
        onnx_node = Mock()
        onnx_node.input = ["input"]
        
        # Should fall back to defaults
        shape = chunker.extract_tensor_shape_from_input("in0_V_data_V", onnx_node)
        assert len(shape) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])