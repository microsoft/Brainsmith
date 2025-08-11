"""
Tests for kernel_integrator converters.
"""

import pytest
from pathlib import Path

from brainsmith.core.dataflow.types import InterfaceType, Shape
from brainsmith.core.dataflow.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.qonnx_types import DataType

from brainsmith.tools.kernel_integrator.types.metadata import (
    KernelMetadata, InterfaceMetadata, DatatypeMetadata
)
from brainsmith.tools.kernel_integrator.types.rtl import Parameter
from brainsmith.tools.kernel_integrator.converters import (
    metadata_to_kernel_definition,
    kernel_definition_to_metadata,
    validate_conversion
)


class TestConverters:
    """Test conversion between KernelMetadata and KernelDefinition."""
    
    def test_simple_kernel_conversion(self):
        """Test converting a simple kernel with input/output."""
        # Create kernel metadata
        metadata = KernelMetadata(
            name="simple_add",
            source_file="simple_add.sv",
            interfaces=[
                InterfaceMetadata(
                    name="input0",
                    interface_type=InterfaceType.INPUT,
                    description="First input"
                ),
                InterfaceMetadata(
                    name="output0",
                    interface_type=InterfaceType.OUTPUT,
                    description="Sum output"
                )
            ],
            parameters=[
                Parameter(name="WIDTH", default_value="32", param_type="integer")
            ],
            exposed_parameters=["WIDTH"]
        )
        
        # Convert to kernel definition
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Verify conversion
        assert kernel_def.name == "simple_add"
        assert len(kernel_def.input_definitions) == 1
        assert len(kernel_def.output_definitions) == 1
        assert kernel_def.input_definitions[0].name == "input0"
        assert kernel_def.output_definitions[0].name == "output0"
        
        # Verify metadata preserved
        assert kernel_def.metadata["source_file"] == "simple_add.sv"
        assert kernel_def.metadata["kernel_type"] == "rtl"
        assert "WIDTH" in kernel_def.metadata["parameters"]
        assert kernel_def.metadata["exposed_parameters"] == ["WIDTH"]
    
    def test_weight_interface_conversion(self):
        """Test that weight interfaces are converted to inputs."""
        metadata = KernelMetadata(
            name="matmul",
            source_file="matmul.sv",
            interfaces=[
                InterfaceMetadata(
                    name="activations",
                    interface_type=InterfaceType.INPUT
                ),
                InterfaceMetadata(
                    name="weights",
                    interface_type=InterfaceType.WEIGHT,
                    is_weight=True,
                    weight_file="weights.dat"
                ),
                InterfaceMetadata(
                    name="results",
                    interface_type=InterfaceType.OUTPUT
                )
            ],
            parameters=[]
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Weight should be converted to input
        assert len(kernel_def.input_definitions) == 2
        assert len(kernel_def.output_definitions) == 1
        
        # Find weight input
        weight_input = next(i for i in kernel_def.input_definitions if i.name == "weights")
        assert weight_input.is_weight is True
        assert weight_input.metadata["weight_file"] == "weights.dat"
    
    def test_special_interfaces_conversion(self):
        """Test CONFIG and CONTROL interfaces go to metadata."""
        metadata = KernelMetadata(
            name="configurable",
            source_file="configurable.sv",
            interfaces=[
                InterfaceMetadata(
                    name="data_in",
                    interface_type=InterfaceType.INPUT
                ),
                InterfaceMetadata(
                    name="data_out",
                    interface_type=InterfaceType.OUTPUT
                ),
                InterfaceMetadata(
                    name="s_axilite",
                    interface_type=InterfaceType.CONFIG,
                    description="AXI-Lite config"
                ),
                InterfaceMetadata(
                    name="global_ctrl",
                    interface_type=InterfaceType.CONTROL,
                    description="Clock and reset"
                )
            ],
            parameters=[]
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Only data interfaces should be in definitions
        assert len(kernel_def.input_definitions) == 1
        assert len(kernel_def.output_definitions) == 1
        
        # Special interfaces in metadata
        assert "special_interfaces" in kernel_def.metadata
        assert "s_axilite" in kernel_def.metadata["special_interfaces"]
        assert "global_ctrl" in kernel_def.metadata["special_interfaces"]
        
        assert kernel_def.metadata["special_interfaces"]["s_axilite"]["type"] == "config"
        assert kernel_def.metadata["special_interfaces"]["global_ctrl"]["type"] == "control"
    
    def test_shape_conversion(self):
        """Test conversion of shape information."""
        metadata = KernelMetadata(
            name="tiled_conv",
            source_file="tiled_conv.sv",
            interfaces=[
                InterfaceMetadata(
                    name="input0",
                    interface_type=InterfaceType.INPUT,
                    bdim_shape=[16, 16, 3],
                    sdim_shape=[1, 1, "CHANNELS"],
                    bdim_params=["TILE_H", "TILE_W", "3"],
                    sdim_params=["1", "1", "CHANNELS"]
                ),
                InterfaceMetadata(
                    name="output0",
                    interface_type=InterfaceType.OUTPUT,
                    bdim_shape=[14, 14, 32],
                    sdim_shape=[1, 1, "FILTERS"]
                )
            ],
            parameters=[
                Parameter(name="TILE_H", default_value="16"),
                Parameter(name="TILE_W", default_value="16"),
                Parameter(name="CHANNELS", default_value="3"),
                Parameter(name="FILTERS", default_value="32")
            ]
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Check input shape conversion
        input_def = kernel_def.input_definitions[0]
        assert input_def.block_tiling == [16, 16, 3]
        assert input_def.stream_tiling == [1, 1, "CHANNELS"]
        
        # Check parameter mappings stored in metadata
        assert input_def.metadata["bdim_params"] == ["TILE_H", "TILE_W", "3"]
        assert input_def.metadata["sdim_params"] == ["1", "1", "CHANNELS"]
    
    def test_datatype_metadata_conversion(self):
        """Test conversion of datatype metadata."""
        dt_meta = DatatypeMetadata(
            name="input0",
            width="INPUT_WIDTH",
            signed="INPUT_SIGNED"
        )
        
        metadata = KernelMetadata(
            name="typed_kernel",
            source_file="typed.sv",
            interfaces=[
                InterfaceMetadata(
                    name="input0",
                    interface_type=InterfaceType.INPUT,
                    datatype_metadata=dt_meta
                ),
                InterfaceMetadata(
                    name="output0",
                    interface_type=InterfaceType.OUTPUT
                )
            ],
            parameters=[
                Parameter(name="INPUT_WIDTH", default_value="16"),
                Parameter(name="INPUT_SIGNED", default_value="1")
            ]
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Check datatype metadata preserved
        input_def = kernel_def.input_definitions[0]
        assert input_def.metadata["datatype_metadata"] == dt_meta
        assert dt_meta.width == "INPUT_WIDTH"
        assert dt_meta.signed == "INPUT_SIGNED"
    
    def test_round_trip_conversion(self):
        """Test converting to KernelDefinition and back."""
        original = KernelMetadata(
            name="round_trip",
            source_file="round_trip.sv",
            interfaces=[
                InterfaceMetadata(
                    name="in0",
                    interface_type=InterfaceType.INPUT,
                    description="Input stream"
                ),
                InterfaceMetadata(
                    name="out0", 
                    interface_type=InterfaceType.OUTPUT,
                    description="Output stream"
                ),
                InterfaceMetadata(
                    name="ctrl",
                    interface_type=InterfaceType.CONTROL,
                    description="Control signals"
                )
            ],
            parameters=[
                Parameter(name="WIDTH", default_value="32"),
                Parameter(name="DEPTH", default_value="1024")
            ],
            exposed_parameters=["WIDTH"]
        )
        
        # Convert to kernel definition
        kernel_def = metadata_to_kernel_definition(original)
        
        # Convert back
        restored = kernel_definition_to_metadata(kernel_def, Path("round_trip.sv"))
        
        # Verify basic properties preserved
        assert restored.name == original.name
        assert restored.source_file == str(original.source_file)
        assert len(restored.interfaces) == len(original.interfaces)
        assert len(restored.parameters) == len(original.parameters)
        assert set(restored.exposed_parameters) == set(original.exposed_parameters)
        
        # Verify interface types preserved
        original_types = {i.interface_type for i in original.interfaces}
        restored_types = {i.interface_type for i in restored.interfaces}
        assert original_types == restored_types
    
    def test_validate_conversion(self):
        """Test conversion validation."""
        metadata = KernelMetadata(
            name="test_kernel",
            source_file="test.sv",
            interfaces=[
                InterfaceMetadata(name="in0", interface_type=InterfaceType.INPUT),
                InterfaceMetadata(name="out0", interface_type=InterfaceType.OUTPUT)
            ],
            parameters=[
                Parameter(name="WIDTH", default_value="32")
            ]
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Should validate successfully
        is_valid, errors = validate_conversion(metadata, kernel_def)
        assert is_valid is True
        assert len(errors) == 0
        
        # Break something and revalidate
        kernel_def.name = "wrong_name"
        is_valid, errors = validate_conversion(metadata, kernel_def)
        assert is_valid is False
        assert any("Name mismatch" in e for e in errors)
    
    def test_relationship_preservation(self):
        """Test that relationships are preserved."""
        from brainsmith.core.dataflow.relationships import DimensionRelationship, RelationType
        
        relationships = [
            DimensionRelationship(
                source_interface="input0",
                target_interface="output0",
                relation=RelationType.EQUAL
            )
        ]
        
        metadata = KernelMetadata(
            name="related",
            source_file="related.sv",
            interfaces=[
                InterfaceMetadata(name="input0", interface_type=InterfaceType.INPUT),
                InterfaceMetadata(name="output0", interface_type=InterfaceType.OUTPUT)
            ],
            parameters=[],
            relationships=relationships
        )
        
        kernel_def = metadata_to_kernel_definition(metadata)
        
        # Relationships should be preserved
        assert len(kernel_def.relationships) == 1
        assert kernel_def.relationships[0].source_interface == "input0"
        assert kernel_def.relationships[0].target_interface == "output0"
        assert kernel_def.relationships[0].relation == RelationType.EQUAL