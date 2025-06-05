"""
Tests for RTL to Dataflow Interface conversion pipeline.

This module tests the interface conversion system that bridges RTL Parser
and the Dataflow Framework.
"""

import pytest
from unittest.mock import Mock, patch
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import Interface as RTLInterface, InterfaceType as RTLInterfaceType
from brainsmith.dataflow.integration.rtl_conversion import RTLInterfaceConverter, validate_conversion_result
from brainsmith.dataflow.core.dataflow_interface import DataflowInterfaceType, DataflowDataType, DataTypeConstraint


class TestRTLInterfaceConverter:
    """Test RTL Interface to DataflowInterface conversion."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.converter = RTLInterfaceConverter()
        
    def test_interface_type_mapping_axi_stream_input(self):
        """Test mapping AXI-Stream to INPUT interface type."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "in0"
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {}
        
        interface_type = self.converter._map_interface_type(rtl_interface)
        assert interface_type == DataflowInterfaceType.INPUT
    
    def test_interface_type_mapping_axi_stream_output(self):
        """Test mapping AXI-Stream to OUTPUT interface type."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "out0"
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {}
        
        interface_type = self.converter._map_interface_type(rtl_interface)
        assert interface_type == DataflowInterfaceType.OUTPUT
    
    def test_interface_type_mapping_axi_stream_weight(self):
        """Test mapping AXI-Stream with weight metadata to WEIGHT interface type."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "weights"
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {"is_weight": True}
        
        interface_type = self.converter._map_interface_type(rtl_interface)
        assert interface_type == DataflowInterfaceType.WEIGHT
    
    def test_interface_type_mapping_axi_lite(self):
        """Test mapping AXI-Lite to CONFIG interface type."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "s_axi_control"
        rtl_interface.type = RTLInterfaceType.AXI_LITE
        rtl_interface.metadata = {}
        
        interface_type = self.converter._map_interface_type(rtl_interface)
        assert interface_type == DataflowInterfaceType.CONFIG
    
    def test_interface_type_mapping_global_control(self):
        """Test mapping Global Control to CONTROL interface type."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "global"
        rtl_interface.type = RTLInterfaceType.GLOBAL_CONTROL
        rtl_interface.metadata = {}
        
        interface_type = self.converter._map_interface_type(rtl_interface)
        assert interface_type == DataflowInterfaceType.CONTROL
    
    def test_dimension_extraction_tdim_override(self):
        """Test dimension extraction with TDIM pragma override."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "in0"
        rtl_interface.metadata = {"tdim_override": [64, 1]}
        
        parameters = {"PE": 4, "CHANNELS": 16}
        qDim, tDim = self.converter._extract_dimensions(rtl_interface, parameters)
        
        assert tDim == [64, 1]
        assert len(qDim) > 0  # qDim should be inferred
    
    def test_dimension_extraction_onnx_metadata(self):
        """Test dimension extraction from ONNX metadata."""
        converter = RTLInterfaceConverter(onnx_metadata={
            "in0_layout": "[N, C, H, W]",
            "in0_shape": [1, 16, 32, 32]
        })
        
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "in0"
        rtl_interface.metadata = {}
        
        with patch.object(converter.tensor_chunking, 'infer_dimensions', return_value=([16], [1024])):
            qDim, tDim = converter._extract_dimensions(rtl_interface, {})
            
        assert qDim == [16]
        assert tDim == [1024]
    
    def test_dimension_extraction_defaults(self):
        """Test default dimension extraction for interfaces without metadata."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "in0"
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {}
        
        qDim, tDim = self.converter._extract_dimensions(rtl_interface, {})
        
        # Should get default dimensions for AXI-Stream
        assert qDim == [16]
        assert tDim == [8]
    
    def test_datatype_extraction_default(self):
        """Test default datatype extraction."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {}
        
        dtype = self.converter._extract_datatype(rtl_interface)
        
        assert isinstance(dtype, DataflowDataType)
        assert dtype.base_type == "UINT"
        assert dtype.bitwidth == 8
        assert dtype.signed == False
    
    def test_datatype_extraction_explicit(self):
        """Test explicit datatype extraction from metadata."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.metadata = {
            "explicit_datatype": {
                "base_type": "INT",
                "bitwidth": 16,
                "signed": True
            }
        }
        
        dtype = self.converter._extract_datatype(rtl_interface)
        
        assert dtype.base_type == "INT"
        assert dtype.bitwidth == 16
        assert dtype.signed == True
    
    def test_datatype_constraints_from_pragma(self):
        """Test datatype constraint extraction from DATATYPE pragma."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "test_interface"  # Add missing name attribute
        rtl_interface.metadata = {
            "datatype_constraints": {
                "base_types": ["INT", "UINT"],
                "min_bitwidth": 4,
                "max_bitwidth": 16
            }
        }
        
        constraints = self.converter._extract_datatype_constraints(rtl_interface)
        
        assert len(constraints) == 1
        constraint = constraints[0]
        assert isinstance(constraint, DataTypeConstraint)
        assert constraint.base_types == ["INT", "UINT"]
        assert constraint.min_bitwidth == 4
        assert constraint.max_bitwidth == 16
    
    def test_datatype_constraints_default(self):
        """Test default datatype constraint generation."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {}
        
        constraints = self.converter._extract_datatype_constraints(rtl_interface)
        
        assert len(constraints) == 1
        constraint = constraints[0]
        assert "INT" in constraint.base_types
        assert "UINT" in constraint.base_types
        assert constraint.min_bitwidth >= 1
        assert constraint.max_bitwidth >= constraint.min_bitwidth
    
    def test_axi_metadata_extraction(self):
        """Test AXI metadata extraction."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {
            "data_width": 64,
            "has_tlast": True,
            "has_tkeep": False
        }
        
        axi_metadata = self.converter._extract_axi_metadata(rtl_interface)
        
        assert axi_metadata["data_width"] == 64
        assert axi_metadata["has_tlast"] == True
        assert axi_metadata["has_tkeep"] == False
    
    def test_convert_single_interface_complete(self):
        """Test complete single interface conversion."""
        rtl_interface = Mock(spec=RTLInterface)
        rtl_interface.name = "in0"
        rtl_interface.type = RTLInterfaceType.AXI_STREAM
        rtl_interface.metadata = {
            "tdim_override": [32],
            "datatype_constraints": {
                "base_types": ["INT"],
                "min_bitwidth": 8,
                "max_bitwidth": 8
            }
        }
        
        dataflow_interface = self.converter._convert_single_interface(rtl_interface, {"PE": 4})
        
        assert dataflow_interface is not None
        assert dataflow_interface.name == "in0"
        assert dataflow_interface.interface_type == DataflowInterfaceType.INPUT
        assert dataflow_interface.tDim == [32]
        assert len(dataflow_interface.allowed_datatypes) == 1
        assert dataflow_interface.allowed_datatypes[0].base_types == ["INT"]
    
    def test_convert_interfaces_multiple(self):
        """Test conversion of multiple interfaces."""
        rtl_interfaces = {}
        
        # Create input interface
        input_interface = Mock(spec=RTLInterface)
        input_interface.name = "in0"
        input_interface.type = RTLInterfaceType.AXI_STREAM
        input_interface.metadata = {}
        rtl_interfaces["in0"] = input_interface
        
        # Create output interface
        output_interface = Mock(spec=RTLInterface)
        output_interface.name = "out0"
        output_interface.type = RTLInterfaceType.AXI_STREAM
        output_interface.metadata = {}
        rtl_interfaces["out0"] = output_interface
        
        # Create config interface
        config_interface = Mock(spec=RTLInterface)
        config_interface.name = "s_axi_control"
        config_interface.type = RTLInterfaceType.AXI_LITE
        config_interface.metadata = {}
        rtl_interfaces["config"] = config_interface
        
        dataflow_interfaces = self.converter.convert_interfaces(rtl_interfaces)
        
        assert len(dataflow_interfaces) == 3
        
        # Verify interface types
        interface_types = [iface.interface_type for iface in dataflow_interfaces]
        assert DataflowInterfaceType.INPUT in interface_types
        assert DataflowInterfaceType.OUTPUT in interface_types
        assert DataflowInterfaceType.CONFIG in interface_types


class TestConversionValidation:
    """Test conversion result validation."""
    
    def test_validate_conversion_result_valid(self):
        """Test validation of valid conversion result."""
        dataflow_interfaces = []
        
        # Create input interface
        input_interface = Mock()
        input_interface.interface_type = DataflowInterfaceType.INPUT
        input_interface.validate_constraints.return_value = []
        dataflow_interfaces.append(input_interface)
        
        # Create output interface
        output_interface = Mock()
        output_interface.interface_type = DataflowInterfaceType.OUTPUT
        output_interface.validate_constraints.return_value = []
        dataflow_interfaces.append(output_interface)
        
        errors = validate_conversion_result(dataflow_interfaces)
        assert len(errors) == 0
    
    def test_validate_conversion_result_missing_input(self):
        """Test validation when INPUT interface is missing."""
        dataflow_interfaces = []
        
        # Only create output interface
        output_interface = Mock()
        output_interface.interface_type = DataflowInterfaceType.OUTPUT
        output_interface.validate_constraints.return_value = []
        dataflow_interfaces.append(output_interface)
        
        errors = validate_conversion_result(dataflow_interfaces)
        
        # Should have warning about missing INPUT
        input_warnings = [e for e in errors if "INPUT" in e.message]
        assert len(input_warnings) == 1
    
    def test_validate_conversion_result_missing_output(self):
        """Test validation when OUTPUT interface is missing."""
        dataflow_interfaces = []
        
        # Only create input interface
        input_interface = Mock()
        input_interface.interface_type = DataflowInterfaceType.INPUT
        input_interface.validate_constraints.return_value = []
        dataflow_interfaces.append(input_interface)
        
        errors = validate_conversion_result(dataflow_interfaces)
        
        # Should have warning about missing OUTPUT
        output_warnings = [e for e in errors if "OUTPUT" in e.message]
        assert len(output_warnings) == 1
    
    def test_validate_conversion_result_interface_errors(self):
        """Test validation when individual interfaces have errors."""
        from brainsmith.dataflow.core.validation import ValidationError, ValidationSeverity
        
        dataflow_interfaces = []
        
        # Create interface with validation errors
        input_interface = Mock()
        input_interface.interface_type = DataflowInterfaceType.INPUT
        input_interface.validate_constraints.return_value = [
            ValidationError(
                component="test_interface",
                error_type="test_error",
                message="Test validation error",
                severity=ValidationSeverity.ERROR,
                context={}
            )
        ]
        dataflow_interfaces.append(input_interface)
        
        errors = validate_conversion_result(dataflow_interfaces)
        
        # Should include interface validation errors
        interface_errors = [e for e in errors if e.error_type == "test_error"]
        assert len(interface_errors) == 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])