"""
Unit tests for Phase 1 exception classes.
"""

import pytest

from brainsmith.core.phase1.exceptions import (
    BrainsmithError,
    BlueprintParseError,
    ValidationError,
    ConfigurationError,
    PluginNotFoundError
)


class TestBrainsmithError:
    """Test base exception class."""
    
    def test_basic_exception(self):
        """Test basic exception creation."""
        error = BrainsmithError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)
    
    def test_inheritance(self):
        """Test that other exceptions inherit from BrainsmithError."""
        parse_error = BlueprintParseError("parse error")
        assert isinstance(parse_error, BrainsmithError)
        
        validation_error = ValidationError("validation error")
        assert isinstance(validation_error, BrainsmithError)
        
        config_error = ConfigurationError("config error")
        assert isinstance(config_error, BrainsmithError)


class TestBlueprintParseError:
    """Test blueprint parsing exception."""
    
    def test_simple_message(self):
        """Test error with just a message."""
        error = BlueprintParseError("Invalid YAML syntax")
        assert str(error) == "Invalid YAML syntax"
    
    def test_with_line_number(self):
        """Test error with line number."""
        error = BlueprintParseError("Missing colon", line=42)
        assert str(error) == "Error at line 42: Missing colon"
    
    def test_with_line_and_column(self):
        """Test error with line and column."""
        error = BlueprintParseError("Unexpected character", line=10, column=15)
        assert str(error) == "Error at line 10, column 15: Unexpected character"
    
    def test_attributes_accessible(self):
        """Test that line and column attributes are accessible."""
        error = BlueprintParseError("Test", line=5, column=10)
        assert error.line == 5
        assert error.column == 10


class TestValidationError:
    """Test validation exception with errors and warnings."""
    
    def test_simple_message(self):
        """Test error with just a message."""
        error = ValidationError("Validation failed")
        assert str(error) == "Validation failed"
        assert error.errors == []
        assert error.warnings == []
    
    def test_with_errors_list(self):
        """Test error with list of errors."""
        errors = [
            "Kernel 'X' not found",
            "Invalid backend 'Y'",
            "Transform 'Z' has wrong stage"
        ]
        error = ValidationError("Design space validation failed", errors=errors)
        
        error_msg = str(error)
        assert "Design space validation failed" in error_msg
        assert "\n\nErrors:\n" in error_msg
        assert "  - Kernel 'X' not found" in error_msg
        assert "  - Invalid backend 'Y'" in error_msg
        assert "  - Transform 'Z' has wrong stage" in error_msg
    
    def test_with_warnings_list(self):
        """Test error with list of warnings."""
        warnings = [
            "High number of combinations",
            "Missing common build step"
        ]
        error = ValidationError("Validation completed with warnings", warnings=warnings)
        
        error_msg = str(error)
        assert "Validation completed with warnings" in error_msg
        assert "\n\nWarnings:\n" in error_msg
        assert "  - High number of combinations" in error_msg
        assert "  - Missing common build step" in error_msg
    
    def test_with_errors_and_warnings(self):
        """Test error with both errors and warnings."""
        errors = ["Critical error 1", "Critical error 2"]
        warnings = ["Warning 1", "Warning 2"]
        
        error = ValidationError(
            "Multiple issues found",
            errors=errors,
            warnings=warnings
        )
        
        error_msg = str(error)
        assert "Multiple issues found" in error_msg
        assert "\n\nErrors:\n" in error_msg
        assert "  - Critical error 1" in error_msg
        assert "\n\nWarnings:\n" in error_msg
        assert "  - Warning 1" in error_msg
    
    def test_attributes_accessible(self):
        """Test that errors and warnings lists are accessible."""
        errors = ["Error 1"]
        warnings = ["Warning 1"]
        error = ValidationError("Test", errors=errors, warnings=warnings)
        
        assert error.errors == errors
        assert error.warnings == warnings


class TestConfigurationError:
    """Test configuration exception."""
    
    def test_simple_error(self):
        """Test basic configuration error."""
        error = ConfigurationError("Invalid configuration value")
        assert str(error) == "Invalid configuration value"
        assert isinstance(error, BrainsmithError)
    
    def test_common_messages(self):
        """Test common configuration error messages."""
        # Model not found
        error = ConfigurationError("Model file not found: /path/to/model.onnx")
        assert "Model file not found" in str(error)
        
        # Invalid path
        error = ConfigurationError("Working directory path is invalid")
        assert "Working directory" in str(error)


class TestPluginNotFoundError:
    """Test plugin not found exception."""
    
    def test_inheritance(self):
        """Test that PluginNotFoundError inherits from BlueprintParseError."""
        error = PluginNotFoundError("transform", "BadTransform")
        assert isinstance(error, BlueprintParseError)
        assert isinstance(error, BrainsmithError)
    
    def test_transform_not_found(self):
        """Test transform not found error."""
        error = PluginNotFoundError(
            "transform",
            "NonExistentTransform",
            ["Transform1", "Transform2", "Transform3"]
        )
        
        error_msg = str(error)
        assert error_msg == "Transform 'NonExistentTransform' not found. Available: ['Transform1', 'Transform2', 'Transform3']"
    
    def test_kernel_not_found(self):
        """Test kernel not found error."""
        error = PluginNotFoundError(
            "kernel",
            "BadKernel",
            ["Kernel1", "Kernel2"]
        )
        
        error_msg = str(error)
        assert error_msg == "Kernel 'BadKernel' not found. Available: ['Kernel1', 'Kernel2']"
    
    def test_backend_not_found(self):
        """Test backend not found error."""
        error = PluginNotFoundError(
            "backend",
            "InvalidBackend",
            ["rtl", "hls", "dsp"]
        )
        
        error_msg = str(error)
        assert "Backend 'InvalidBackend' not found" in error_msg
        assert "Available: ['rtl', 'hls', 'dsp']" in error_msg
    
    def test_truncation_of_long_lists(self):
        """Test that long lists are truncated with ellipsis."""
        many_options = [f"Option{i}" for i in range(10)]
        error = PluginNotFoundError("transform", "Bad", many_options)
        
        error_msg = str(error)
        # Should show first 5
        assert "Option0" in error_msg
        assert "Option1" in error_msg
        assert "Option2" in error_msg
        assert "Option3" in error_msg
        assert "Option4" in error_msg
        # Should not show 6th and beyond
        assert "Option5" not in error_msg
        # Should have ellipsis
        assert "..." in error_msg
    
    def test_no_suggestions(self):
        """Test error without suggestions."""
        error = PluginNotFoundError("kernel", "Unknown", None)
        error_msg = str(error)
        assert error_msg == "Kernel 'Unknown' not found"
    
    def test_empty_suggestions(self):
        """Test error with empty suggestion list."""
        error = PluginNotFoundError("transform", "Bad", [])
        error_msg = str(error)
        assert error_msg == "Transform 'Bad' not found. Available: []"
    
    def test_capitalization(self):
        """Test that plugin type is capitalized in message."""
        error = PluginNotFoundError("transform", "X", None)
        assert str(error).startswith("Transform")
        
        error = PluginNotFoundError("kernel", "Y", None)
        assert str(error).startswith("Kernel")
        
        error = PluginNotFoundError("backend", "Z", None)
        assert str(error).startswith("Backend")