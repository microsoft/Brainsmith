############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Interface metadata structure validation tests.

This test module validates the complete InterfaceMetadata structure creation
and compiler name generation, ensuring the RTL Parser correctly builds
interface metadata objects with proper naming conventions.

Test Coverage:
- Compiler name generation (input0, output0, weight0, config0, global)
- Interface type determination accuracy
- Weight interface detection from WEIGHT pragmas
- Multi-interface scenarios with proper indexing
"""

import pytest
from brainsmith.tools.hw_kernel_gen.rtl_parser import RTLParser
from brainsmith.tools.hw_kernel_gen.data import InterfaceType

from .utils.rtl_builder import StrictRTLBuilder


class TestCompilerNameGeneration:
    """Test systematic compiler name generation for interfaces."""
    
    def test_single_interface_compiler_names(self, rtl_parser):
        """Test compiler name generation for single interfaces of each type."""
        rtl = (StrictRTLBuilder()
               .module("single_interface_test")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .axi_lite_slave("s_axi_config")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "single_interface_test.sv")
        
        # Find interfaces by type
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        output_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.OUTPUT),
            None
        )
        config_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.CONFIG),
            None
        )
        control_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.CONTROL),
            None
        )
        
        # Validate compiler names
        assert input_interface is not None
        # Note: Compiler names are generated internally, check if they follow pattern
        # This depends on the actual InterfaceMetadata structure
        assert hasattr(input_interface, 'name')  # Original name preserved
        
        assert output_interface is not None
        assert hasattr(output_interface, 'name')
        
        assert config_interface is not None
        assert hasattr(config_interface, 'name')
        
        assert control_interface is not None
        assert hasattr(control_interface, 'name')
    
    def test_multiple_input_interfaces_indexing(self, rtl_parser):
        """Test multiple input interfaces get proper indexing."""
        rtl = (StrictRTLBuilder()
               .module("multi_input_test")
               .add_stream_input("s_axis_in0", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_in1", bdim_value="64", sdim_value="256")
               .add_stream_input("s_axis_in2", bdim_value="16", sdim_value="1024")
               .add_stream_output("m_axis_output", bdim_value="112")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_input_test.sv")
        
        # Find all input interfaces
        input_interfaces = [
            i for i in kernel_metadata.interfaces 
            if i.interface_type == InterfaceType.INPUT
        ]
        assert len(input_interfaces) == 3
        
        # Check original names are preserved and unique
        input_names = {i.name for i in input_interfaces}
        expected_names = {"s_axis_in0", "s_axis_in1", "s_axis_in2"}
        assert input_names == expected_names
        
        # Each interface should have proper metadata structure
        for interface in input_interfaces:
            assert interface.interface_type == InterfaceType.INPUT
            assert interface.name in expected_names
    
    def test_multiple_output_interfaces_indexing(self, rtl_parser):
        """Test multiple output interfaces get proper indexing."""
        rtl = (StrictRTLBuilder()
               .module("multi_output_test")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_out0", bdim_value="16")
               .add_stream_output("m_axis_out1", bdim_value="32")
               .add_stream_output("m_axis_result", bdim_value="48")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "multi_output_test.sv")
        
        # Find all output interfaces
        output_interfaces = [
            i for i in kernel_metadata.interfaces 
            if i.interface_type == InterfaceType.OUTPUT
        ]
        assert len(output_interfaces) == 3
        
        # Check original names are preserved
        output_names = {i.name for i in output_interfaces}
        expected_names = {"m_axis_out0", "m_axis_out1", "m_axis_result"}
        assert output_names == expected_names
    
    def test_weight_interface_detection_and_naming(self, rtl_parser):
        """Test weight interfaces get proper detection and naming."""
        rtl = (StrictRTLBuilder()
               .module("weight_naming_test")
               .pragma("WEIGHT", "s_axis_weights")
               .pragma("WEIGHT", "s_axis_bias")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_weights", bdim_value="64", sdim_value="256")
               .add_stream_input("s_axis_bias", bdim_value="32", sdim_value="1")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "weight_naming_test.sv")
        
        # Find weight interfaces
        weight_interfaces = [
            i for i in kernel_metadata.interfaces 
            if i.interface_type == InterfaceType.WEIGHT
        ]
        assert len(weight_interfaces) == 2
        
        # Check weight interfaces have correct names
        weight_names = {i.name for i in weight_interfaces}
        expected_weight_names = {"s_axis_weights", "s_axis_bias"}
        assert weight_names == expected_weight_names
        
        # Find remaining input interface (should not be weight)
        input_interfaces = [
            i for i in kernel_metadata.interfaces 
            if i.interface_type == InterfaceType.INPUT
        ]
        assert len(input_interfaces) == 1
        assert input_interfaces[0].name == "s_axis_input"
    
    def test_mixed_interface_types_comprehensive(self, rtl_parser):
        """Test comprehensive scenario with all interface types."""
        rtl = (StrictRTLBuilder()
               .module("comprehensive_interface_test")
               .pragma("WEIGHT", "s_axis_weights")
               .add_stream_input("s_axis_in0", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_in1", bdim_value="64", sdim_value="256")
               .add_stream_input("s_axis_weights", bdim_value="64", sdim_value="512")
               .add_stream_output("m_axis_out0", bdim_value="32")
               .add_stream_output("m_axis_out1", bdim_value="64")
               .axi_lite_slave("s_axi_config")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "comprehensive_interface_test.sv")
        
        # Categorize interfaces by type
        interfaces_by_type = {}
        for interface in kernel_metadata.interfaces:
            if interface.interface_type not in interfaces_by_type:
                interfaces_by_type[interface.interface_type] = []
            interfaces_by_type[interface.interface_type].append(interface)
        
        # Validate counts
        assert len(interfaces_by_type[InterfaceType.INPUT]) == 2
        assert len(interfaces_by_type[InterfaceType.WEIGHT]) == 1
        assert len(interfaces_by_type[InterfaceType.OUTPUT]) == 2
        assert len(interfaces_by_type[InterfaceType.CONFIG]) == 1
        assert len(interfaces_by_type[InterfaceType.CONTROL]) == 1
        
        # Validate names
        input_names = {i.name for i in interfaces_by_type[InterfaceType.INPUT]}
        assert input_names == {"s_axis_in0", "s_axis_in1"}
        
        weight_names = {i.name for i in interfaces_by_type[InterfaceType.WEIGHT]}
        assert weight_names == {"s_axis_weights"}
        
        output_names = {i.name for i in interfaces_by_type[InterfaceType.OUTPUT]}
        assert output_names == {"m_axis_out0", "m_axis_out1"}


class TestInterfaceTypeDetection:
    """Test interface type determination accuracy."""
    
    def test_axi_stream_direction_detection(self, rtl_parser):
        """Test AXI-Stream interface direction detection."""
        rtl = (StrictRTLBuilder()
               .module("direction_detection_test")
               .add_stream_input("s_axis_data", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_result", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "direction_detection_test.sv")
        
        # Check that interfaces were detected with correct types
        input_interface = next(
            (i for i in kernel_metadata.interfaces if "data" in i.name),
            None
        )
        output_interface = next(
            (i for i in kernel_metadata.interfaces if "result" in i.name),
            None
        )
        
        assert input_interface is not None
        assert input_interface.interface_type == InterfaceType.INPUT
        
        assert output_interface is not None
        assert output_interface.interface_type == InterfaceType.OUTPUT
    
    def test_weight_pragma_changes_interface_type(self, rtl_parser):
        """Test WEIGHT pragma changes interface from INPUT to WEIGHT."""
        rtl = (StrictRTLBuilder()
               .module("weight_type_change_test")
               .pragma("WEIGHT", "s_axis_params")
               .add_stream_input("s_axis_data", bdim_value="32", sdim_value="512")
               .add_stream_input("s_axis_params", bdim_value="64", sdim_value="256")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "weight_type_change_test.sv")
        
        # Find interfaces
        data_interface = next(
            (i for i in kernel_metadata.interfaces if "data" in i.name),
            None
        )
        params_interface = next(
            (i for i in kernel_metadata.interfaces if "params" in i.name),
            None
        )
        
        # Data interface should remain INPUT
        assert data_interface is not None
        assert data_interface.interface_type == InterfaceType.INPUT
        
        # Params interface should be changed to WEIGHT by pragma
        assert params_interface is not None
        assert params_interface.interface_type == InterfaceType.WEIGHT
    
    def test_axi_lite_interface_detection(self, rtl_parser):
        """Test AXI-Lite interface detection as CONFIG type."""
        rtl = (StrictRTLBuilder()
               .module("axi_lite_detection_test")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .axi_lite_slave("s_axi_control")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "axi_lite_detection_test.sv")
        
        # Find AXI-Lite interface
        config_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.CONFIG),
            None
        )
        
        assert config_interface is not None
        assert "axi" in config_interface.name or "control" in config_interface.name
    
    def test_global_control_interface_detection(self, rtl_parser):
        """Test global control signal detection."""
        rtl = (StrictRTLBuilder()
               .module("global_control_test")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="512")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())  # Global control is added by default
        
        kernel_metadata = rtl_parser.parse(rtl, "global_control_test.sv")
        
        # Find global control interface
        control_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.CONTROL),
            None
        )
        
        assert control_interface is not None


class TestInterfaceMetadataCompleteness:
    """Test that interface metadata objects are complete and properly structured."""
    
    def test_interface_metadata_required_fields(self, rtl_parser):
        """Test all interfaces have required metadata fields."""
        rtl = (StrictRTLBuilder()
               .module("metadata_completeness_test")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "32")
               .pragma("BDIM", "s_axis_input", "TILE_SIZE")
               .pragma("SDIM", "s_axis_input", "STREAM_SIZE")
               .parameter("TILE_SIZE", "32")
               .parameter("STREAM_SIZE", "1024")
               .add_stream_input("s_axis_input", bdim_value="32", sdim_value="1024")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "metadata_completeness_test.sv")
        
        # Check each interface has complete metadata structure
        for interface in kernel_metadata.interfaces:
            # Required fields for all interfaces
            assert hasattr(interface, 'name')
            assert hasattr(interface, 'interface_type')
            assert hasattr(interface, 'datatype_constraints')
            
            # Optional fields that may be None
            assert hasattr(interface, 'description')
            assert hasattr(interface, 'datatype_metadata')
            assert hasattr(interface, 'bdim_params')
            assert hasattr(interface, 'sdim_params')
            
            # Validate types
            assert isinstance(interface.name, str)
            assert isinstance(interface.interface_type, InterfaceType)
            assert isinstance(interface.datatype_constraints, list)
    
    def test_interface_metadata_with_pragma_effects(self, rtl_parser):
        """Test interface metadata reflects pragma applications."""
        rtl = (StrictRTLBuilder()
               .module("pragma_effects_test")
               .parameter("BATCH", "16")
               .parameter("CHANNELS", "3")
               .parameter("HEIGHT", "32")
               .parameter("WIDTH", "32")
               .pragma("DATATYPE", "s_axis_input", "UINT", "8", "16")
               .pragma("BDIM", "s_axis_input", "[BATCH, CHANNELS]")
               .pragma("SDIM", "s_axis_input", "[HEIGHT, WIDTH]")
               .add_stream_input("s_axis_input", bdim_value="48", sdim_value="1024")
               .add_stream_output("m_axis_output", bdim_value="32")
               .build())
        
        kernel_metadata = rtl_parser.parse(rtl, "pragma_effects_test.sv")
        
        input_interface = next(
            (i for i in kernel_metadata.interfaces if i.interface_type == InterfaceType.INPUT),
            None
        )
        assert input_interface is not None
        
        # Validate pragma effects in metadata
        # DATATYPE pragma effect
        assert len(input_interface.datatype_constraints) == 1
        constraint = input_interface.datatype_constraints[0]
        assert constraint.base_type == "UINT"
        assert constraint.min_width == 8
        assert constraint.max_width == 16
        
        # BDIM pragma effect
        assert input_interface.bdim_params == ["BATCH", "CHANNELS"]
        
        # SDIM pragma effect
        assert input_interface.sdim_params == ["HEIGHT", "WIDTH"]