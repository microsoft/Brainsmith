#!/usr/bin/env python3
"""Tests for standardized error handling."""

import pytest
from brainsmith.tools.hw_kernel_gen.errors import (
    BrainsmithError, RTLParsingError, InterfaceDetectionError,
    PragmaProcessingError, CodeGenerationError, ValidationError,
    ErrorSeverity, handle_error_with_recovery
)

class TestBrainsmithError:
    """Test base BrainsmithError functionality."""
    
    def test_basic_error_creation(self):
        """Test basic error creation."""
        error = BrainsmithError("Test error message")
        assert error.message == "Test error message"
        assert error.context == {}
        assert error.severity == ErrorSeverity.ERROR
        assert error.suggestions == []
        assert error.timestamp is not None
    
    def test_error_with_context(self):
        """Test error creation with context."""
        context = {"file": "test.sv", "line": 42}
        suggestions = ["Check syntax", "Review documentation"]
        
        error = BrainsmithError(
            "Test error with context",
            context=context,
            severity=ErrorSeverity.WARNING,
            suggestions=suggestions
        )
        
        assert error.context == context
        assert error.severity == ErrorSeverity.WARNING
        assert error.suggestions == suggestions
    
    def test_error_serialization(self):
        """Test error dictionary serialization."""
        error = BrainsmithError(
            "Serialization test",
            context={"key": "value"},
            suggestions=["suggestion1"]
        )
        
        error_dict = error.to_dict()
        assert error_dict['type'] == 'BrainsmithError'
        assert error_dict['message'] == 'Serialization test'
        assert error_dict['context'] == {"key": "value"}
        assert error_dict['suggestions'] == ["suggestion1"]

class TestSpecializedErrors:
    """Test specialized error classes."""
    
    def test_rtl_parsing_error(self):
        """Test RTL parsing error."""
        error = RTLParsingError(
            "Parse failed",
            file_path="test.sv",
            line_number=10
        )
        
        assert "test.sv" in error.context['file_path']
        assert error.context['line_number'] == 10
        assert len(error.suggestions) > 0
    
    def test_interface_detection_error(self):
        """Test interface detection error."""
        error = InterfaceDetectionError(
            "Interface not found",
            interface_name="s_axis_input"
        )
        
        assert error.context['interface_name'] == "s_axis_input"
        assert len(error.suggestions) > 0
    
    def test_code_generation_error(self):
        """Test code generation error."""
        error = CodeGenerationError(
            "Generation failed",
            generator_type="HWCustomOpGenerator",
            template_name="hw_custom_op_slim.py.j2"
        )
        
        assert error.context['generator_type'] == "HWCustomOpGenerator"
        assert error.context['template_name'] == "hw_custom_op_slim.py.j2"

class TestErrorRecovery:
    """Test error recovery mechanisms."""
    
    def test_successful_recovery(self):
        """Test successful error recovery."""
        def recovery_strategy(error):
            return "recovered_value"
        
        test_error = Exception("Test error")
        result = handle_error_with_recovery(test_error, [recovery_strategy])
        assert result == "recovered_value"
    
    def test_failed_recovery(self):
        """Test failed error recovery."""
        def failing_strategy(error):
            raise Exception("Recovery failed")
        
        test_error = Exception("Test error")
        with pytest.raises(Exception, match="Test error"):
            handle_error_with_recovery(test_error, [failing_strategy])
    
    def test_no_recovery_strategies(self):
        """Test error handling with no recovery strategies."""
        test_error = Exception("Test error")
        with pytest.raises(Exception, match="Test error"):
            handle_error_with_recovery(test_error, [])

if __name__ == "__main__":
    pytest.main([__file__])