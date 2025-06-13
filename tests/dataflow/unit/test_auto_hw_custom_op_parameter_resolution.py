#!/usr/bin/env python3
"""
Test parameter resolution integration in AutoHWCustomOp.

Tests the new parameter resolution bridge between symbolic BDIM parameters
and concrete dataflow modeling.
"""

import pytest
import sys
import os
import onnx.helper
from typing import List

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata
from brainsmith.dataflow.core.interface_types import InterfaceType
from brainsmith.dataflow.core.qonnx_types import DatatypeConstraintGroup
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class MockParameterTestOp(AutoHWCustomOp):
    """Mock AutoHWCustomOp for parameter resolution testing."""
    
    def __init__(self, onnx_node, interface_metadata_list):
        self._test_interface_metadata = interface_metadata_list
        super().__init__(onnx_node)
    
    def get_nodeattr_types(self):
        """Define node attributes for testing."""
        my_attrs = {
            "PE": ("i", False, 8),
            "SIMD": ("i", False, 4), 
            "CHANNELS": ("i", False, 32),
            "test_input_dtype": ("s", True, ""),
            "weights_dtype": ("s", True, ""),
            "output_dtype": ("s", True, ""),
        }
        my_attrs.update(super().get_enhanced_nodeattr_types())
        return my_attrs
    
    def get_interface_metadata(self) -> List[InterfaceMetadata]:
        """Return test interface metadata."""
        return self._test_interface_metadata


class TestParameterResolutionIntegration:
    """Test parameter resolution bridge in AutoHWCustomOp."""
    
    def test_parameter_resolution_basic(self):
        """Test basic parameter resolution with runtime parameters."""
        # Create metadata with symbolic BDIM pragma for last dimension
        chunking_strategy = BlockChunkingStrategy(block_shape=[":", ":", ":", "PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            PE=8,
            test_input_dtype="UINT8"
        )
        
        auto_op = MockParameterTestOp(node, [metadata])
        
        # Check that dataflow model was created successfully
        assert auto_op.dataflow_model is not None
        
        # Check that input interface has resolved block_dims
        input_interfaces = auto_op.dataflow_model.input_interfaces
        assert len(input_interfaces) == 1
        
        input_iface = input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 128, 8]  # Last dim: PE parameter resolved to 8
    
    def test_parameter_resolution_multi_dimension(self):
        """Test parameter resolution with multiple dimensions."""
        # Create metadata with multi-dimension symbolic BDIM pragma
        chunking_strategy = BlockChunkingStrategy(block_shape=["SIMD", "PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            SIMD=4,
            PE=8,
            test_input_dtype="UINT8"
        )
        
        auto_op = MockParameterTestOp(node, [metadata])
        
        # Check resolved block_dims
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 4, 8]  # Last two dims: SIMD=4, PE=8
    
    def test_parameter_resolution_with_colon(self):
        """Test parameter resolution with ':' (full dimension)."""
        chunking_strategy = BlockChunkingStrategy(block_shape=[":", "PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            PE=16,
            test_input_dtype="UINT8"
        )
        
        auto_op = MockParameterTestOp(node, [metadata])
        
        # Check that ':' was resolved to tensor dimension and PE to parameter value
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        # Last two dims: ':' preserves tensor dim (128), PE=16
        assert input_iface.block_dims == [1, 128, 128, 16]
    
    def test_parameter_resolution_defaults(self):
        """Test parameter resolution with default values when no runtime parameters provided."""
        chunking_strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with just the required datatype, PE will use default
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            test_input_dtype="UINT8"
            # PE will use default value from get_nodeattr_types
        )
        
        auto_op = MockParameterTestOp(node, [metadata])
        
        # Check that parameter was resolved to default value (8)
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 128, 8]  # Last dim: PE resolved to default value 8
    
    def test_parameter_resolution_missing_parameter_error(self):
        """Test error when required parameter is missing."""
        chunking_strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with missing required parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            SIMD=4,  # PE is missing but needed for block_shape
            test_input_dtype="UINT8"
        )
        
        with pytest.raises(ValueError) as exc_info:
            MockParameterTestOp(node, [metadata])
        
        assert "Parameter 'PE' not found" in str(exc_info.value)
    
    def test_parameter_resolution_multiple_interfaces(self):
        """Test parameter resolution with multiple interfaces."""
        # Input interface with PE parameter
        input_chunking = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        input_metadata = InterfaceMetadata(
            name="input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=input_chunking
        )
        
        # Weight interface with SIMD parameter
        weight_chunking = BlockChunkingStrategy(block_shape=["SIMD"], rindex=0)
        weight_metadata = InterfaceMetadata(
            name="weights",
            interface_type=InterfaceType.WEIGHT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=weight_chunking
        )
        
        # Output interface with mixed parameters
        output_chunking = BlockChunkingStrategy(block_shape=["CHANNELS", "PE"], rindex=0)
        output_metadata = InterfaceMetadata(
            name="output",
            interface_type=InterfaceType.OUTPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=output_chunking
        )
        
        # Create ONNX node with all parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            PE=8,
            SIMD=4,
            CHANNELS=32,
            input_dtype="UINT8",
            weights_dtype="UINT8",
            output_dtype="UINT8"
        )
        
        auto_op = MockParameterTestOp(node, [input_metadata, weight_metadata, output_metadata])
        
        # Check all interfaces have resolved block_dims
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        weight_iface = auto_op.dataflow_model.weight_interfaces[0]
        output_iface = auto_op.dataflow_model.output_interfaces[0]
        
        assert input_iface.block_dims == [1, 128, 128, 8]  # Last dim: PE=8
        assert weight_iface.block_dims == [1, 128, 128, 4]  # Last dim: SIMD=4
        assert output_iface.block_dims == [1, 128, 32, 8]  # Last two dims: CHANNELS=32, PE=8
    
    def test_parameter_resolution_preserves_integers(self):
        """Test that existing integer block_dims are preserved."""
        # Create chunking strategy that returns integers (old style)
        chunking_strategy = BlockChunkingStrategy(block_shape=[16], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            datatype_constraints=[DatatypeConstraintGroup("UINT", 8, 8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create ONNX node with parameters
        node = onnx.helper.make_node(
            'MockParameterTestOp',
            ['input'],
            ['output'],
            PE=8,
            test_input_dtype="UINT8"
        )
        
        auto_op = MockParameterTestOp(node, [metadata])
        
        # Check that integer dimensions are preserved
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 128, 16]  # Last dim: Integer preserved as-is


if __name__ == "__main__":
    pytest.main([__file__, "-v"])