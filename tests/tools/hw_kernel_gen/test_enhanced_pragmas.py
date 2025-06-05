"""
Tests for enhanced TDIM and DATATYPE pragma support in RTL Parser.

This module tests the new pragma functionality added for Phase 2 of the
Interface-Wise Dataflow Modeling Framework.
"""

import pytest
from unittest.mock import Mock, patch
from brainsmith.tools.hw_kernel_gen.rtl_parser.data import (
    TDimPragma, DatatypePragma, PragmaType, PragmaError, Interface, InterfaceType
)


class TestTDimPragma:
    """Test TDIM pragma parsing and application."""
    
    def test_tdim_pragma_creation_valid(self):
        """Test creating TDIM pragma with valid inputs."""
        pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["in0", "PE*CHANNELS", "1"],
            line_number=10
        )
        
        assert pragma.parsed_data["interface_name"] == "in0"
        assert pragma.parsed_data["dimension_expressions"] == ["PE*CHANNELS", "1"]
    
    def test_tdim_pragma_creation_invalid_inputs(self):
        """Test TDIM pragma with invalid inputs."""
        # Too few inputs
        with pytest.raises(PragmaError, match="requires interface name and at least one dimension expression"):
            TDimPragma(
                type=PragmaType.TDIM,
                inputs=["in0"],
                line_number=10
            )
        
        # Empty interface name
        with pytest.raises(PragmaError, match="interface name .* is not a valid identifier"):
            TDimPragma(
                type=PragmaType.TDIM,
                inputs=["", "PE*CHANNELS"],
                line_number=10
            )
        
        # Empty dimension expression
        with pytest.raises(PragmaError, match="dimension expression .* is empty"):
            TDimPragma(
                type=PragmaType.TDIM,
                inputs=["in0", "PE*CHANNELS", ""],
                line_number=10
            )
    
    def test_tdim_expression_evaluation_valid(self):
        """Test TDIM expression evaluation with valid parameters."""
        pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["in0", "PE*CHANNELS", "BATCH_SIZE+1"],
            line_number=10
        )
        
        parameters = {"PE": 4, "CHANNELS": 16, "BATCH_SIZE": 8}
        
        # Test valid expressions
        assert pragma._evaluate_expression("PE*CHANNELS", parameters) == 64
        assert pragma._evaluate_expression("BATCH_SIZE+1", parameters) == 9
        assert pragma._evaluate_expression("16", parameters) == 16
        assert pragma._evaluate_expression("max(PE, CHANNELS)", parameters) == 16
    
    def test_tdim_expression_evaluation_invalid(self):
        """Test TDIM expression evaluation with invalid cases."""
        pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["in0", "PE*CHANNELS"],
            line_number=10
        )
        
        parameters = {"PE": 4, "CHANNELS": 16}
        
        # Undefined parameter
        with pytest.raises(PragmaError, match="references undefined parameter"):
            pragma._evaluate_expression("UNDEFINED_PARAM", parameters)
        
        # Invalid syntax
        with pytest.raises(PragmaError, match="has invalid syntax"):
            pragma._evaluate_expression("PE*", parameters)
        
        # Non-positive result
        with pytest.raises(PragmaError, match="must evaluate to a positive integer"):
            pragma._evaluate_expression("PE-10", parameters)
        
        # Non-numeric result
        with pytest.raises(PragmaError, match="must evaluate to a number"):
            pragma._evaluate_expression("'string'", parameters)
    
    def test_tdim_pragma_apply_valid(self):
        """Test TDIM pragma application to valid interface."""
        pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["in0", "PE*CHANNELS", "1"],
            line_number=10
        )
        
        # Create mock interface
        interface = Interface(
            name="in0",
            type=InterfaceType.AXI_STREAM,
            ports={},
            validation_result=Mock(),
            metadata={}
        )
        
        interfaces = {"in0": interface}
        parameters = {"PE": 4, "CHANNELS": 16}
        
        # Apply pragma
        pragma.apply(interfaces=interfaces, parameters=parameters)
        
        # Verify metadata was set
        assert interface.metadata["tdim_override"] == [64, 1]
        assert interface.metadata["tdim_expressions"] == ["PE*CHANNELS", "1"]
    
    def test_tdim_pragma_apply_interface_not_found(self):
        """Test TDIM pragma application when interface not found."""
        pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["nonexistent", "PE*CHANNELS"],
            line_number=10
        )
        
        interfaces = {}
        parameters = {"PE": 4, "CHANNELS": 16}
        
        with patch('brainsmith.tools.hw_kernel_gen.rtl_parser.data.logger') as mock_logger:
            pragma.apply(interfaces=interfaces, parameters=parameters)
            mock_logger.warning.assert_called_with(
                "TDIM pragma at line 10: interface 'nonexistent' not found"
            )


class TestEnhancedDatatypePragma:
    """Test enhanced DATATYPE pragma parsing and application."""
    
    def test_datatype_pragma_creation_valid(self):
        """Test creating DATATYPE pragma with valid inputs."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "INT,UINT", "1", "16"],
            line_number=10
        )
        
        expected_data = {
            "interface_name": "in0",
            "base_types": ["INT", "UINT"],
            "min_bitwidth": 1,
            "max_bitwidth": 16
        }
        
        assert pragma.parsed_data == expected_data
    
    def test_datatype_pragma_creation_invalid_inputs(self):
        """Test DATATYPE pragma with invalid inputs."""
        # Wrong number of inputs
        with pytest.raises(PragmaError, match="requires interface_name, base_types, min_bits, max_bits"):
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["in0", "INT", "8"],
                line_number=10
            )
        
        # Invalid bit widths
        with pytest.raises(PragmaError, match="must be integers"):
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["in0", "INT", "abc", "16"],
                line_number=10
            )
        
        # Min > Max
        with pytest.raises(PragmaError, match="cannot be greater than"):
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["in0", "INT", "16", "8"],
                line_number=10
            )
        
        # Invalid base type
        with pytest.raises(PragmaError, match="invalid base type"):
            DatatypePragma(
                type=PragmaType.DATATYPE,
                inputs=["in0", "INVALID", "8", "16"],
                line_number=10
            )
    
    def test_datatype_pragma_apply_valid(self):
        """Test DATATYPE pragma application to valid interface."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "FIXED", "8", "8"],
            line_number=10
        )
        
        # Create mock interface
        interface = Interface(
            name="in0",
            type=InterfaceType.AXI_STREAM,
            ports={},
            validation_result=Mock(),
            metadata={}
        )
        
        interfaces = {"in0": interface}
        
        # Apply pragma
        pragma.apply(interfaces=interfaces)
        
        # Verify metadata was set
        expected_constraints = {
            "base_types": ["FIXED"],
            "min_bitwidth": 8,
            "max_bitwidth": 8
        }
        
        assert interface.metadata["datatype_constraints"] == expected_constraints
    
    def test_datatype_pragma_apply_interface_not_found(self):
        """Test DATATYPE pragma application when interface not found."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["nonexistent", "INT", "8", "16"],
            line_number=10
        )
        
        interfaces = {}
        
        with patch('brainsmith.tools.hw_kernel_gen.rtl_parser.data.logger') as mock_logger:
            pragma.apply(interfaces=interfaces)
            mock_logger.warning.assert_called_with(
                "DATATYPE pragma from line 10 for interface 'nonexistent' did not match any existing interfaces."
            )
    
    def test_datatype_pragma_multiple_base_types(self):
        """Test DATATYPE pragma with multiple base types."""
        pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["weights", "INT,UINT,FIXED", "4", "32"],
            line_number=10
        )
        
        assert pragma.parsed_data["base_types"] == ["INT", "UINT", "FIXED"]
        assert pragma.parsed_data["min_bitwidth"] == 4
        assert pragma.parsed_data["max_bitwidth"] == 32


class TestPragmaIntegration:
    """Test integration of enhanced pragmas with RTL Parser."""
    
    def test_pragma_type_enum_contains_tdim(self):
        """Test that PragmaType enum contains TDIM."""
        assert PragmaType.TDIM.value == "tdim"
    
    def test_pragma_str_representation(self):
        """Test string representation of pragmas."""
        tdim_pragma = TDimPragma(
            type=PragmaType.TDIM,
            inputs=["in0", "PE*CHANNELS", "1"],
            line_number=10
        )
        
        datatype_pragma = DatatypePragma(
            type=PragmaType.DATATYPE,
            inputs=["in0", "INT", "8", "16"],
            line_number=20
        )
        
        assert str(tdim_pragma) == "@brainsmith tdim in0 PE*CHANNELS 1"
        assert str(datatype_pragma) == "@brainsmith datatype in0 INT 8 16"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])