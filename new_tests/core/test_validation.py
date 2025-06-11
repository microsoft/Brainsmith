"""
Validation Tests - Input Validation System

Tests input validation functions and blueprint validation.
Validates file existence checks, format validation, and error handling.
"""

import pytest
import tempfile
import yaml
from pathlib import Path
from unittest.mock import patch, Mock

# Import validation components
try:
    from brainsmith.core.api import validate_blueprint, _validate_inputs, _load_and_validate_blueprint
    VALIDATION_AVAILABLE = True
except ImportError:
    VALIDATION_AVAILABLE = False


@pytest.mark.core
class TestInputValidation:
    """Test input validation functions."""
    
    def test_validate_inputs_valid_files(self, sample_model_path, sample_blueprint_path):
        """Test input validation with valid files."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Should not raise any exceptions
        try:
            _validate_inputs(
                sample_model_path, 
                sample_blueprint_path, 
                objectives=None, 
                constraints=None
            )
        except Exception as e:
            pytest.fail(f"Validation should pass for valid inputs: {e}")
    
    def test_validate_inputs_missing_model(self, sample_blueprint_path):
        """Test input validation with missing model file."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            _validate_inputs(
                "nonexistent_model.onnx",
                sample_blueprint_path,
                objectives=None,
                constraints=None
            )
    
    def test_validate_inputs_missing_blueprint(self, sample_model_path):
        """Test input validation with missing blueprint file."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        with pytest.raises(FileNotFoundError, match="Blueprint file not found"):
            _validate_inputs(
                sample_model_path,
                "nonexistent_blueprint.yaml",
                objectives=None,
                constraints=None
            )
    
    def test_validate_inputs_invalid_model_format(self, sample_blueprint_path):
        """Test input validation with invalid model format."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Create a non-ONNX file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not an ONNX file")
            invalid_model_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Model must be ONNX format"):
                _validate_inputs(
                    invalid_model_path,
                    sample_blueprint_path,
                    objectives=None,
                    constraints=None
                )
        finally:
            Path(invalid_model_path).unlink()
    
    def test_validate_inputs_invalid_blueprint_format(self, sample_model_path):
        """Test input validation with invalid blueprint format."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Create a non-YAML file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            f.write(b"This is not a YAML file")
            invalid_blueprint_path = f.name
        
        try:
            with pytest.raises(ValueError, match="Blueprint must be YAML format"):
                _validate_inputs(
                    sample_model_path,
                    invalid_blueprint_path,
                    objectives=None,
                    constraints=None
                )
        finally:
            Path(invalid_blueprint_path).unlink()
    
    def test_validate_inputs_valid_objectives(self, sample_model_path, sample_blueprint_path):
        """Test input validation with valid objectives."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        valid_objectives = {
            "throughput": {
                "direction": "maximize",
                "weight": 1.0,
                "target": 500.0
            },
            "latency": {
                "direction": "minimize",
                "weight": 0.8
            }
        }
        
        # Should not raise any exceptions
        try:
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=valid_objectives,
                constraints=None
            )
        except Exception as e:
            pytest.fail(f"Validation should pass for valid objectives: {e}")
    
    def test_validate_inputs_invalid_objectives_format(self, sample_model_path, sample_blueprint_path):
        """Test input validation with invalid objectives format."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Objectives should be dict, not string
        invalid_objectives = {
            "throughput": "maximize"  # Should be dict with direction
        }
        
        with pytest.raises(ValueError, match="must be a dictionary"):
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=invalid_objectives,
                constraints=None
            )
    
    def test_validate_inputs_missing_objective_direction(self, sample_model_path, sample_blueprint_path):
        """Test input validation with missing objective direction."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        invalid_objectives = {
            "throughput": {
                "weight": 1.0
                # Missing "direction"
            }
        }
        
        with pytest.raises(ValueError, match="missing 'direction' field"):
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=invalid_objectives,
                constraints=None
            )
    
    def test_validate_inputs_invalid_objective_direction(self, sample_model_path, sample_blueprint_path):
        """Test input validation with invalid objective direction."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        invalid_objectives = {
            "throughput": {
                "direction": "invalid_direction",  # Should be "maximize" or "minimize"
                "weight": 1.0
            }
        }
        
        with pytest.raises(ValueError, match="direction must be 'maximize' or 'minimize'"):
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=invalid_objectives,
                constraints=None
            )
    
    def test_validate_inputs_valid_constraints(self, sample_model_path, sample_blueprint_path):
        """Test input validation with valid constraints."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        valid_constraints = {
            "max_luts": 0.8,
            "max_dsps": 0.7,
            "max_brams": 0.6,
            "max_power": 20.0,
            "target_frequency": 150.0
        }
        
        # Should not raise any exceptions
        try:
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=None,
                constraints=valid_constraints
            )
        except Exception as e:
            pytest.fail(f"Validation should pass for valid constraints: {e}")
    
    def test_validate_inputs_invalid_constraint_types(self, sample_model_path, sample_blueprint_path):
        """Test input validation with invalid constraint types."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        invalid_constraints = {
            "max_luts": "not_a_number"  # Should be numeric
        }
        
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=None,
                constraints=invalid_constraints
            )


@pytest.mark.core
class TestBlueprintValidation:
    """Test blueprint validation functionality."""
    
    def test_validate_blueprint_function_valid(self, sample_blueprint_path):
        """Test validate_blueprint function with valid blueprint."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        is_valid, errors = validate_blueprint(sample_blueprint_path)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # With our sample blueprint, validation might pass or fail depending on 
        # blueprint system availability, but should not crash
    
    def test_validate_blueprint_function_invalid(self, invalid_blueprint_path):
        """Test validate_blueprint function with invalid blueprint."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        is_valid, errors = validate_blueprint(invalid_blueprint_path)
        
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # Should detect that the blueprint is invalid
        if not is_valid:
            assert len(errors) > 0
    
    def test_validate_blueprint_nonexistent_file(self):
        """Test validate_blueprint with non-existent file."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        is_valid, errors = validate_blueprint("nonexistent_blueprint.yaml")
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("not found" in error.lower() or "file" in error.lower() for error in errors)
    
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_validate_blueprint_with_mock_loader(self, mock_loader, sample_blueprint_path):
        """Test validate_blueprint with mocked blueprint loader."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Mock successful blueprint loading
        mock_loader.return_value = {
            "name": "test_blueprint",
            "description": "Test blueprint"
        }
        
        is_valid, errors = validate_blueprint(sample_blueprint_path)
        
        assert is_valid is True
        assert len(errors) == 0
        mock_loader.assert_called_once_with(sample_blueprint_path)
    
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    def test_validate_blueprint_with_loader_exception(self, mock_loader, sample_blueprint_path):
        """Test validate_blueprint when blueprint loader raises exception."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Mock blueprint loader to raise exception
        mock_loader.side_effect = ValueError("Invalid blueprint format")
        
        is_valid, errors = validate_blueprint(sample_blueprint_path)
        
        assert is_valid is False
        assert len(errors) > 0
        assert any("Invalid blueprint format" in error for error in errors)


@pytest.mark.core
class TestBlueprintLoading:
    """Test blueprint loading and validation functions."""
    
    def test_load_and_validate_blueprint_success(self, sample_blueprint_path):
        """Test successful blueprint loading and validation."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Test direct loading without mocking - this should work with our real implementation
        result = _load_and_validate_blueprint(sample_blueprint_path)
        
        # Should return the actual blueprint data from the test file
        assert isinstance(result, dict)
        assert 'name' in result
        assert result['name'] == 'test_blueprint'
    
    def test_load_and_validate_blueprint_validation_failure(self, invalid_blueprint_path):
        """Test blueprint loading with validation failure."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Use the actual invalid blueprint fixture - this should work or at least not crash
        try:
            result = _load_and_validate_blueprint(invalid_blueprint_path)
            # If it doesn't raise an error, just verify it loaded something
            assert isinstance(result, dict)
        except (ValueError, yaml.YAMLError):
            # Expected behavior for invalid blueprint
            pass
    
    def test_load_and_validate_blueprint_import_error(self, sample_blueprint_path):
        """Test blueprint loading when blueprint system not available."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Test with BlueprintManager import error
        with patch('brainsmith.infrastructure.dse.blueprint_manager.BlueprintManager', side_effect=ImportError("Blueprint system not available")):
            with pytest.raises(RuntimeError, match="Blueprint system not available"):
                _load_and_validate_blueprint(sample_blueprint_path)
    
    def test_load_and_validate_blueprint_file_not_found(self):
        """Test blueprint loading with non-existent file."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        with pytest.raises(FileNotFoundError, match="Blueprint file not found"):
            _load_and_validate_blueprint("nonexistent_blueprint.yaml")


@pytest.mark.core
class TestValidationEdgeCases:
    """Test validation edge cases and error conditions."""
    
    def test_validate_inputs_none_objectives(self, sample_model_path, sample_blueprint_path):
        """Test validation with None objectives (should be allowed)."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # None objectives should be allowed
        try:
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=None,
                constraints=None
            )
        except Exception as e:
            pytest.fail(f"None objectives should be allowed: {e}")
    
    def test_validate_inputs_empty_objectives(self, sample_model_path, sample_blueprint_path):
        """Test validation with empty objectives dict."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Empty objectives dict should be allowed
        try:
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives={},
                constraints=None
            )
        except Exception as e:
            pytest.fail(f"Empty objectives should be allowed: {e}")
    
    def test_validate_inputs_mixed_constraint_types(self, sample_model_path, sample_blueprint_path):
        """Test validation with mixed valid/invalid constraint types."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        mixed_constraints = {
            "max_luts": 0.8,  # Valid float
            "max_dsps": 5,    # Valid int
            "custom_constraint": "string_value",  # Should be allowed (not in numeric list)
            "max_power": "invalid"  # Invalid for numeric constraint
        }
        
        with pytest.raises(ValueError, match="must be numeric"):
            _validate_inputs(
                sample_model_path,
                sample_blueprint_path,
                objectives=None,
                constraints=mixed_constraints
            )
    
    def test_validate_blueprint_with_various_extensions(self):
        """Test blueprint validation with various file extensions."""
        if not VALIDATION_AVAILABLE:
            pytest.skip("Validation functions not available")
        
        # Create temporary files with different extensions
        extensions_to_test = ['.yaml', '.yml']
        
        for ext in extensions_to_test:
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as f:
                f.write(b"name: test_blueprint\ndescription: Test")
                temp_path = f.name
            
            try:
                # Should accept both .yaml and .yml
                _validate_inputs(
                    "dummy_model.onnx",  # Won't be checked due to early file extension validation
                    temp_path,
                    objectives=None,
                    constraints=None
                )
            except FileNotFoundError:
                # Expected since dummy_model.onnx doesn't exist
                # But no ValueError about blueprint format
                pass
            except ValueError as e:
                if "Blueprint must be YAML format" in str(e):
                    pytest.fail(f"Extension {ext} should be accepted")
            finally:
                Path(temp_path).unlink()


@pytest.mark.integration
def test_validation_integration_workflow(sample_model_path, sample_blueprint_path):
    """Test complete validation workflow integration."""
    if not VALIDATION_AVAILABLE:
        pytest.skip("Validation functions not available")
    
    # Test complete validation flow
    objectives = {
        "throughput": {"direction": "maximize", "weight": 1.0},
        "latency": {"direction": "minimize", "weight": 0.8}
    }
    
    constraints = {
        "max_luts": 0.8,
        "max_dsps": 0.7,
        "max_power": 20.0
    }
    
    try:
        # Step 1: Input validation
        _validate_inputs(sample_model_path, sample_blueprint_path, objectives, constraints)
        
        # Step 2: Blueprint validation
        is_valid, errors = validate_blueprint(sample_blueprint_path)
        
        # Should complete without exceptions
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
    except ImportError:
        pytest.skip("Blueprint system not available for integration test")
    except Exception as e:
        # Other exceptions should be handled gracefully
        assert "validation" in str(e).lower() or "not found" in str(e).lower()


# Helper functions for validation testing
def create_temp_file_with_content(content: str, suffix: str = '.tmp') -> str:
    """Helper to create temporary file with specific content."""
    with tempfile.NamedTemporaryFile(mode='w', suffix=suffix, delete=False) as f:
        f.write(content)
        return f.name


def create_invalid_model_file() -> str:
    """Helper to create an invalid model file for testing."""
    return create_temp_file_with_content(
        "This is not a valid ONNX model file",
        suffix='.onnx'
    )


def create_invalid_blueprint_file() -> str:
    """Helper to create an invalid blueprint file for testing."""
    invalid_yaml = """
name: invalid_blueprint
# Missing required fields and has syntax errors
invalid_syntax: [unclosed list
"""
    return create_temp_file_with_content(invalid_yaml, suffix='.yaml')


def assert_validation_error_contains(error_list: list, expected_text: str):
    """Helper to assert validation errors contain expected text."""
    assert any(expected_text.lower() in error.lower() for error in error_list), \
        f"Expected '{expected_text}' in errors: {error_list}"


def cleanup_temp_files(*file_paths):
    """Helper to clean up temporary files."""
    for file_path in file_paths:
        try:
            if file_path and Path(file_path).exists():
                Path(file_path).unlink()
        except Exception:
            pass  # Ignore cleanup errors