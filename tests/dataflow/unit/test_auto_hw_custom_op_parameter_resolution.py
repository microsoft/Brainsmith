#!/usr/bin/env python3
"""
Test parameter resolution integration in AutoHWCustomOp.

Tests the new parameter resolution bridge between symbolic BDIM parameters
and concrete dataflow modeling.
"""

import pytest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../..'))

from brainsmith.dataflow.core.auto_hw_custom_op import AutoHWCustomOp
from brainsmith.dataflow.core.interface_metadata import InterfaceMetadata, DataTypeConstraint
from brainsmith.dataflow.core.interface_types import InterfaceType
# DataflowDataType removed - using QONNX DataType directly
from brainsmith.dataflow.core.block_chunking import BlockChunkingStrategy


class TestParameterResolutionIntegration:
    """Test parameter resolution bridge in AutoHWCustomOp."""
    
    def test_parameter_resolution_basic(self):
        """Test basic parameter resolution with runtime parameters."""
        # Create metadata with symbolic BDIM pragma
        chunking_strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create AutoHWCustomOp with runtime parameters
        runtime_parameters = {"PE": 8}
        
        auto_op = AutoHWCustomOp(
            onnx_node=None,  # Mock for testing
            interface_metadata=[metadata],
            runtime_parameters=runtime_parameters
        )
        
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
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        # Create AutoHWCustomOp with runtime parameters
        runtime_parameters = {"SIMD": 4, "PE": 8}
        
        auto_op = AutoHWCustomOp(
            onnx_node=None,
            interface_metadata=[metadata],
            runtime_parameters=runtime_parameters
        )
        
        # Check resolved block_dims
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 4, 8]  # Last two dims: SIMD=4, PE=8
    
    def test_parameter_resolution_with_colon(self):
        """Test parameter resolution with ':' (full dimension)."""
        chunking_strategy = BlockChunkingStrategy(block_shape=[":", "PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        runtime_parameters = {"PE": 16}
        
        auto_op = AutoHWCustomOp(
            onnx_node=None,
            interface_metadata=[metadata],
            runtime_parameters=runtime_parameters
        )
        
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
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        # No runtime parameters provided
        auto_op = AutoHWCustomOp(
            onnx_node=None,
            interface_metadata=[metadata]
        )
        
        # Check that parameter was resolved to default value (1)
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 128, 1]  # Last dim: PE resolved to default value 1
    
    def test_parameter_resolution_missing_parameter_error(self):
        """Test error when required parameter is missing."""
        chunking_strategy = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        
        metadata = InterfaceMetadata(
            name="test_input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        # Provide some runtime parameters but not the required one
        runtime_parameters = {"SIMD": 4}  # PE is missing
        
        with pytest.raises(ValueError) as exc_info:
            AutoHWCustomOp(
                onnx_node=None,
                interface_metadata=[metadata],
                runtime_parameters=runtime_parameters
            )
        
        assert "Parameter 'PE' not found in runtime_parameters" in str(exc_info.value)
        assert "Available parameters: ['SIMD']" in str(exc_info.value)
    
    def test_parameter_resolution_multiple_interfaces(self):
        """Test parameter resolution with multiple interfaces."""
        # Input interface with PE parameter
        input_chunking = BlockChunkingStrategy(block_shape=["PE"], rindex=0)
        input_metadata = InterfaceMetadata(
            name="input",
            interface_type=InterfaceType.INPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=input_chunking
        )
        
        # Weight interface with SIMD parameter
        weight_chunking = BlockChunkingStrategy(block_shape=["SIMD"], rindex=0)
        weight_metadata = InterfaceMetadata(
            name="weights",
            interface_type=InterfaceType.WEIGHT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=weight_chunking
        )
        
        # Output interface with mixed parameters
        output_chunking = BlockChunkingStrategy(block_shape=["CHANNELS", "PE"], rindex=0)
        output_metadata = InterfaceMetadata(
            name="output",
            interface_type=InterfaceType.OUTPUT,
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=output_chunking
        )
        
        runtime_parameters = {"PE": 8, "SIMD": 4, "CHANNELS": 32}
        
        auto_op = AutoHWCustomOp(
            onnx_node=None,
            interface_metadata=[input_metadata, weight_metadata, output_metadata],
            runtime_parameters=runtime_parameters
        )
        
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
            allowed_datatypes=[DataTypeConstraint(finn_type="UINT8", bit_width=8)],
            chunking_strategy=chunking_strategy
        )
        
        auto_op = AutoHWCustomOp(
            onnx_node=None,
            interface_metadata=[metadata],
            runtime_parameters={"PE": 8}
        )
        
        # Check that integer dimensions are preserved
        input_iface = auto_op.dataflow_model.input_interfaces[0]
        assert input_iface.block_dims == [1, 128, 128, 16]  # Last dim: Integer preserved as-is


if __name__ == "__main__":
    pytest.main([__file__, "-v"])