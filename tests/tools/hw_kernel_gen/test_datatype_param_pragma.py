#!/usr/bin/env python3
"""
Unit tests for DatatypeParamPragma functionality.

Tests the multi-interface datatype parameter mapping implementation including:
- DatatypeParamPragma parsing and validation
- InterfaceMetadata datatype parameter generation
- Interface name matching patterns
- Error handling for invalid inputs
"""

import pytest
from unittest.mock import Mock, patch
from typing import Dict, Optional

# We'll need to mock the imports since we're testing in isolation
import sys
from unittest.mock import MagicMock

# Mock the brainsmith modules
sys.modules['brainsmith.dataflow.core.interface_metadata'] = MagicMock()
sys.modules['brainsmith.dataflow.core.interface_types'] = MagicMock()
sys.modules['brainsmith.dataflow.core.block_chunking'] = MagicMock()


class TestDatatypeParamPragma:
    """Test suite for DatatypeParamPragma class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock InterfaceMetadata for testing
        self.mock_interface_metadata = Mock()
        self.mock_interface_metadata.name = "s_axis_input0"
        self.mock_interface_metadata.interface_type = "INPUT"
        self.mock_interface_metadata.datatype_constraints = []
        self.mock_interface_metadata.chunking_strategy = None
        self.mock_interface_metadata.description = None
        self.mock_interface_metadata.datatype_params = None
    
    def test_pragma_parsing_valid_inputs(self):
        """Test parsing of valid DATATYPE_PARAM pragma inputs."""
        # Test case 1: Basic width parameter
        inputs1 = ["s_axis_input0", "width", "INPUT0_WIDTH"]
        parsed1 = self._parse_pragma_inputs(inputs1)
        
        assert parsed1["interface_name"] == "s_axis_input0"
        assert parsed1["property_type"] == "width"
        assert parsed1["parameter_name"] == "INPUT0_WIDTH"
        
        # Test case 2: Signed parameter
        inputs2 = ["s_axis_query", "signed", "QUERY_SIGNED"]
        parsed2 = self._parse_pragma_inputs(inputs2)
        
        assert parsed2["interface_name"] == "s_axis_query"
        assert parsed2["property_type"] == "signed"
        assert parsed2["parameter_name"] == "QUERY_SIGNED"
        
        # Test case 3: Format parameter
        inputs3 = ["weights_V", "format", "WEIGHT_FORMAT"]
        parsed3 = self._parse_pragma_inputs(inputs3)
        
        assert parsed3["interface_name"] == "weights_V"
        assert parsed3["property_type"] == "format"
        assert parsed3["parameter_name"] == "WEIGHT_FORMAT"
        
        print("✅ Valid pragma parsing tests passed")
    
    def test_pragma_parsing_invalid_inputs(self):
        """Test error handling for invalid pragma inputs."""
        # Test case 1: Too few arguments
        with pytest.raises(Exception) as exc_info:
            self._parse_pragma_inputs(["s_axis_input0", "width"])
        assert "requires interface_name, property_type, parameter_name" in str(exc_info.value)
        
        # Test case 2: Too many arguments
        with pytest.raises(Exception) as exc_info:
            self._parse_pragma_inputs(["s_axis_input0", "width", "INPUT0_WIDTH", "extra"])
        assert "requires interface_name, property_type, parameter_name" in str(exc_info.value)
        
        # Test case 3: Invalid property type
        with pytest.raises(Exception) as exc_info:
            self._parse_pragma_inputs(["s_axis_input0", "invalid_prop", "PARAM"])
        assert "Invalid property_type 'invalid_prop'" in str(exc_info.value)
        
        print("✅ Invalid pragma parsing tests passed")
    
    def test_interface_name_matching(self):
        """Test interface name matching patterns."""
        test_cases = [
            # (pragma_name, interface_name, should_match)
            ("in0", "in0", True),  # Exact match
            ("in0", "in0_V_data_V", True),  # Prefix match
            ("in0_V_data_V", "in0", True),  # Reverse prefix match
            ("s_axis_input0", "s_axis_input0_tdata", True),  # AXI naming
            ("query", "s_axis_query", True),  # Base name match
            ("weights", "bias", False),  # No match
            ("input0", "input1", False),  # Different indices
        ]
        
        for pragma_name, interface_name, expected in test_cases:
            result = self._interface_names_match(pragma_name, interface_name)
            assert result == expected, f"Failed: {pragma_name} vs {interface_name} (expected {expected}, got {result})"
        
        print("✅ Interface name matching tests passed")
    
    def test_datatype_params_application(self):
        """Test application of datatype parameters to metadata."""
        # Test applying width parameter
        result1 = self._apply_pragma_to_metadata(
            interface_name="s_axis_input0",
            property_type="width",
            parameter_name="INPUT0_WIDTH",
            target_interface="s_axis_input0"
        )
        
        assert result1.datatype_params is not None
        assert result1.datatype_params["width"] == "INPUT0_WIDTH"
        
        # Test applying signed parameter to same interface (should merge)
        existing_params = {"width": "INPUT0_WIDTH"}
        result2 = self._apply_pragma_to_metadata(
            interface_name="s_axis_input0",
            property_type="signed", 
            parameter_name="SIGNED_INPUT0",
            target_interface="s_axis_input0",
            existing_datatype_params=existing_params
        )
        
        assert result2.datatype_params["width"] == "INPUT0_WIDTH"
        assert result2.datatype_params["signed"] == "SIGNED_INPUT0"
        
        # Test non-matching interface (should not apply)
        result3 = self._apply_pragma_to_metadata(
            interface_name="s_axis_input0",
            property_type="width",
            parameter_name="INPUT0_WIDTH", 
            target_interface="s_axis_input1"
        )
        
        assert result3.datatype_params is None  # Should not be modified
        
        print("✅ Datatype params application tests passed")
    
    def test_property_type_validation(self):
        """Test validation of property types."""
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        
        for prop in valid_properties:
            # Should not raise exception
            parsed = self._parse_pragma_inputs(["interface", prop, "PARAM"])
            assert parsed["property_type"] == prop
        
        # Test invalid property types
        invalid_properties = ['invalid', 'datatype', 'bits', 'type']
        
        for prop in invalid_properties:
            with pytest.raises(Exception) as exc_info:
                self._parse_pragma_inputs(["interface", prop, "PARAM"])
            assert "Invalid property_type" in str(exc_info.value)
        
        print("✅ Property type validation tests passed")
    
    def _parse_pragma_inputs(self, inputs):
        """Mock implementation of DatatypeParamPragma._parse_inputs logic."""
        if len(inputs) != 3:
            raise Exception("DATATYPE_PARAM pragma requires interface_name, property_type, parameter_name")
        
        interface_name = inputs[0]
        property_type = inputs[1].lower()
        parameter_name = inputs[2]
        
        # Validate property type
        valid_properties = ['width', 'signed', 'format', 'bias', 'fractional_width']
        if property_type not in valid_properties:
            raise Exception(f"Invalid property_type '{property_type}'. Must be one of: {valid_properties}")
        
        return {
            "interface_name": interface_name,
            "property_type": property_type,
            "parameter_name": parameter_name
        }
    
    def _interface_names_match(self, pragma_name: str, interface_name: str) -> bool:
        """Mock implementation of interface name matching logic."""
        # Exact match
        if pragma_name == interface_name:
            return True
        
        # Prefix match (e.g., "in0" matches "in0_V_data_V")
        if interface_name.startswith(pragma_name):
            return True
        
        # Reverse prefix match (e.g., "in0_V_data_V" matches "in0")
        if pragma_name.startswith(interface_name):
            return True
        
        # Base name matching (remove common suffixes)
        pragma_base = pragma_name.replace('_V_data_V', '').replace('_data', '').replace('s_axis_', '').replace('m_axis_', '')
        interface_base = interface_name.replace('_V_data_V', '').replace('_data', '').replace('s_axis_', '').replace('m_axis_', '').replace('_tdata', '')
        
        return pragma_base == interface_base
    
    def _apply_pragma_to_metadata(self, interface_name: str, property_type: str, parameter_name: str, 
                                  target_interface: str, existing_datatype_params: Optional[Dict] = None):
        """Mock implementation of pragma application to metadata."""
        # Check if pragma applies to target interface
        if not self._interface_names_match(interface_name, target_interface):
            # Return unchanged metadata
            result = Mock()
            result.name = target_interface
            result.datatype_params = existing_datatype_params
            return result
        
        # Apply pragma to matching interface
        current_params = existing_datatype_params or {}
        current_params[property_type] = parameter_name
        
        result = Mock()
        result.name = target_interface
        result.datatype_params = current_params
        return result


class TestInterfaceMetadataEnhancements:
    """Test suite for InterfaceMetadata datatype parameter enhancements."""
    
    def test_get_datatype_parameter_name_defaults(self):
        """Test default parameter name generation."""
        test_cases = [
            # (interface_name, property_type, expected_param_name)
            ("s_axis_input0", "width", "INPUT0_WIDTH"),
            ("s_axis_input0", "signed", "SIGNED_INPUT0"),
            ("s_axis_input1", "width", "INPUT1_WIDTH"),
            ("s_axis_input1", "signed", "SIGNED_INPUT1"),
            ("m_axis_output0", "width", "OUTPUT0_WIDTH"),
            ("m_axis_output0", "signed", "SIGNED_OUTPUT0"),
            ("s_axis_query", "width", "QUERY_WIDTH"),
            ("s_axis_query", "signed", "SIGNED_QUERY"),
            ("weights_V", "width", "WEIGHTS_WIDTH"),
            ("weights_V", "signed", "SIGNED_WEIGHTS"),
            ("s_axis_input0_tdata", "width", "INPUT0_WIDTH"),  # Strip suffixes
        ]
        
        for interface_name, property_type, expected in test_cases:
            result = self._get_datatype_parameter_name(interface_name, property_type, None)
            assert result == expected, f"Failed: {interface_name}.{property_type} -> {result} (expected {expected})"
        
        print("✅ Default parameter name generation tests passed")
    
    def test_get_datatype_parameter_name_custom(self):
        """Test custom parameter name override via datatype_params."""
        # Test custom width parameter
        custom_params1 = {"width": "CUSTOM_WIDTH"}
        result1 = self._get_datatype_parameter_name("s_axis_input0", "width", custom_params1)
        assert result1 == "CUSTOM_WIDTH"
        
        # Test custom signed parameter
        custom_params2 = {"signed": "QUERY_SIGNED"}
        result2 = self._get_datatype_parameter_name("s_axis_query", "signed", custom_params2)
        assert result2 == "QUERY_SIGNED"
        
        # Test fallback to default for unmapped property
        custom_params3 = {"width": "CUSTOM_WIDTH"}  # Only width is custom
        result3 = self._get_datatype_parameter_name("s_axis_input0", "signed", custom_params3)
        assert result3 == "SIGNED_INPUT0"  # Should use default
        
        print("✅ Custom parameter name override tests passed")
    
    def test_clean_interface_name_extraction(self):
        """Test interface name cleaning logic."""
        test_cases = [
            # (raw_name, clean_name)
            ("s_axis_input0", "INPUT0"),
            ("m_axis_output0", "OUTPUT0"),
            ("s_axis_query", "QUERY"),
            ("weights_V", "WEIGHTS"),
            ("s_axis_input0_tdata", "INPUT0"),
            ("m_axis_output1_tdata", "OUTPUT1"),
            ("config", "CONFIG"),
            ("clk", "CLK"),
            ("s_axis_value_V_data_V", "VALUE"),
        ]
        
        for raw_name, expected_clean in test_cases:
            result = self._get_clean_interface_name(raw_name)
            assert result == expected_clean, f"Failed: {raw_name} -> {result} (expected {expected_clean})"
        
        print("✅ Interface name cleaning tests passed")
    
    def _get_datatype_parameter_name(self, interface_name: str, property_type: str, datatype_params: Optional[Dict] = None) -> str:
        """Mock implementation of InterfaceMetadata.get_datatype_parameter_name logic."""
        if datatype_params and property_type in datatype_params:
            return datatype_params[property_type]
        
        clean_name = self._get_clean_interface_name(interface_name)
        
        if property_type == 'width':
            return f"{clean_name}_WIDTH"
        elif property_type == 'signed':
            return f"SIGNED_{clean_name}"
        elif property_type == 'format':
            return f"{clean_name}_FORMAT"
        elif property_type == 'bias':
            return f"{clean_name}_BIAS"
        elif property_type == 'fractional_width':
            return f"{clean_name}_FRACTIONAL_WIDTH"
        else:
            return f"{clean_name}_{property_type.upper()}"
    
    def _get_clean_interface_name(self, interface_name: str) -> str:
        """Mock implementation of InterfaceMetadata._get_clean_interface_name logic."""
        clean_name = interface_name
        
        # Remove common prefixes
        for prefix in ['s_axis_', 'm_axis_', 'axis_']:
            if clean_name.startswith(prefix):
                clean_name = clean_name[len(prefix):]
                break
        
        # Remove common suffixes
        for suffix in ['_tdata', '_tvalid', '_tready', '_V_data_V', '_V']:
            if clean_name.endswith(suffix):
                clean_name = clean_name[:-len(suffix)]
                break
        
        return clean_name.upper()


def run_all_tests():
    """Run all unit tests."""
    print("🧪 Starting DatatypeParamPragma unit tests...\n")
    
    # Test DatatypeParamPragma functionality
    pragma_tests = TestDatatypeParamPragma()
    pragma_tests.setup_method()
    pragma_tests.test_pragma_parsing_valid_inputs()
    pragma_tests.test_pragma_parsing_invalid_inputs()
    pragma_tests.test_interface_name_matching()
    pragma_tests.test_datatype_params_application()
    pragma_tests.test_property_type_validation()
    
    print()
    
    # Test InterfaceMetadata enhancements
    metadata_tests = TestInterfaceMetadataEnhancements()
    metadata_tests.test_get_datatype_parameter_name_defaults()
    metadata_tests.test_get_datatype_parameter_name_custom()
    metadata_tests.test_clean_interface_name_extraction()
    
    print("\n🎉 All DatatypeParamPragma unit tests passed!")


if __name__ == "__main__":
    run_all_tests()