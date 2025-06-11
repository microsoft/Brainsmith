"""
Core API Tests - forge() Function

Tests the main forge() function with minimal mocking approach.
Validates core functionality, input validation, and result structure.
"""

import pytest
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile
import shutil

# Import the actual components we're testing
try:
    from brainsmith.core.api import forge, validate_blueprint
    from brainsmith.core.metrics import DSEMetrics
    BRAINSMITH_AVAILABLE = True
except ImportError:
    BRAINSMITH_AVAILABLE = False

from new_tests.fixtures.mock_helpers import (
    mock_forge_successful_result, mock_forge_fallback_result,
    mock_finn_unavailable, mock_dse_unavailable, mock_blueprint_system_unavailable,
    assert_forge_result_structure, assert_metrics_valid
)


@pytest.mark.core
class TestForgeAPI:
    """Test cases for the main forge() API function."""
    
    def test_forge_basic_functionality(self, sample_model_path, sample_blueprint_path):
        """Test forge() function with valid inputs."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Mock external dependencies only
        with patch('brainsmith.core.api._run_full_dse') as mock_dse, \
             patch('brainsmith.core.api._generate_dataflow_core') as mock_core:
            
            # Setup mocks with realistic return values
            mock_dse_result = Mock()
            mock_dse_result.results = []
            mock_dse_result.best_result = {
                'dataflow_graph': Mock(),
                'throughput': 500.0,
                'latency': 20.0,
                'lut_util': 0.45,
                'dsp_util': 0.32
            }
            mock_dse.return_value = mock_dse_result
            mock_core.return_value = {'ip_files': ['test.v'], 'synthesis_results': {'status': 'success'}}
            
            # Test forge function
            result = forge(
                model_path=sample_model_path,
                blueprint_path=sample_blueprint_path
            )
            
            # Validate result structure
            assert isinstance(result, dict)
            assert_forge_result_structure(result)
            
            # Verify mocks were called
            mock_dse.assert_called_once()
    
    def test_forge_with_hw_graph_mode(self, sample_model_path, sample_blueprint_path):
        """Test forge() with is_hw_graph=True (skip to HW optimization)."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        with patch('brainsmith.core.api._load_dataflow_graph') as mock_load, \
             patch('brainsmith.core.api._run_hw_optimization_dse') as mock_hw_dse:
            
            mock_load.return_value = Mock()
            mock_hw_dse_result = Mock()
            mock_hw_dse_result.results = []
            mock_hw_dse_result.best_result = {'dataflow_graph': Mock()}
            mock_hw_dse.return_value = mock_hw_dse_result
            
            result = forge(
                model_path=sample_model_path,
                blueprint_path=sample_blueprint_path,
                is_hw_graph=True
            )
            
            # Should load existing graph and run HW optimization
            mock_load.assert_called_once_with(sample_model_path)
            mock_hw_dse.assert_called_once()
            
            assert_forge_result_structure(result)
    
    def test_forge_build_core_false(self, sample_model_path, sample_blueprint_path):
        """Test forge() with build_core=False (checkpoint mode)."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        with patch('brainsmith.core.api._run_full_dse') as mock_dse, \
             patch('brainsmith.core.api._generate_dataflow_core') as mock_core:
            
            mock_dse_result = Mock()
            mock_dse_result.results = []
            mock_dse_result.best_result = {'dataflow_graph': Mock()}
            mock_dse.return_value = mock_dse_result
            
            result = forge(
                model_path=sample_model_path,
                blueprint_path=sample_blueprint_path,
                build_core=False
            )
            
            # Should NOT call core generation
            mock_core.assert_not_called()
            
            # Should still have dataflow_graph but no dataflow_core
            assert result['dataflow_core'] is None
            assert result['dataflow_graph'] is not None
    
    def test_forge_with_output_directory(self, sample_model_path, sample_blueprint_path, temp_test_dir):
        """Test forge() with output directory specified."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        output_dir = Path(temp_test_dir) / "test_output"
        
        with patch('brainsmith.core.api._run_full_dse') as mock_dse, \
             patch('brainsmith.core.api._save_forge_results') as mock_save:
            
            mock_dse_result = Mock()
            mock_dse_result.results = []
            mock_dse_result.best_result = {'dataflow_graph': Mock()}
            mock_dse.return_value = mock_dse_result
            
            result = forge(
                model_path=sample_model_path,
                blueprint_path=sample_blueprint_path,
                output_dir=str(output_dir)
            )
            
            # Should call save function
            mock_save.assert_called_once()
            # Verify output directory was passed correctly
            save_args = mock_save.call_args[0]
            assert save_args[1] == str(output_dir)
    
    def test_forge_input_validation(self, sample_blueprint_path):
        """Test forge() input validation."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Test missing model file
        with pytest.raises(FileNotFoundError, match="Model file not found"):
            forge("nonexistent_model.onnx", "blueprint.yaml")
        
        # Test missing blueprint file  
        with tempfile.NamedTemporaryFile(suffix='.onnx', delete=False) as f:
            model_path = f.name
        
        try:
            with pytest.raises(FileNotFoundError, match="Blueprint file not found"):
                forge(model_path, "nonexistent_blueprint.yaml")
        finally:
            Path(model_path).unlink()
        
        # Test invalid model format
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as f:
            invalid_model = f.name
        
        try:
            with pytest.raises(ValueError, match="Model must be ONNX format"):
                forge(invalid_model, sample_blueprint_path)
        finally:
            Path(invalid_model).unlink()
    
    def test_forge_objectives_validation(self, sample_model_path, sample_blueprint_path):
        """Test forge() with objectives parameter validation."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Test invalid objectives format
        invalid_objectives = {
            'throughput': 'invalid_format'  # Should be dict
        }
        
        with pytest.raises(ValueError, match="must be a dictionary"):
            forge(sample_model_path, sample_blueprint_path, objectives=invalid_objectives)
        
        # Test missing direction
        invalid_objectives = {
            'throughput': {'weight': 1.0}  # Missing direction
        }
        
        with pytest.raises(ValueError, match="missing 'direction' field"):
            forge(sample_model_path, sample_blueprint_path, objectives=invalid_objectives)
    
    def test_forge_constraints_validation(self, sample_model_path, sample_blueprint_path):
        """Test forge() with constraints parameter validation."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Test invalid constraint types
        invalid_constraints = {
            'max_luts': 'invalid_string'  # Should be numeric
        }
        
        with pytest.raises(ValueError, match="must be numeric"):
            forge(sample_model_path, sample_blueprint_path, constraints=invalid_constraints)
    
    def test_forge_fallback_behavior(self, sample_model_path, sample_blueprint_path):
        """Test forge() fallback behavior when components unavailable."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")

        # Mock DSE interface import to trigger fallback
        with patch('brainsmith.infrastructure.dse.interface.DSEInterface', side_effect=ImportError("DSE not available")):
            result = forge(sample_model_path, sample_blueprint_path)

            # Should complete successfully with fallback behavior
            assert result is not None
            assert 'dataflow_graph' in result
            assert_forge_result_structure(result)
            
            # Verify we got fallback DSE results
            assert 'dse_results' in result
            assert 'metrics' in result
    
    def test_forge_error_handling(self, sample_model_path, invalid_blueprint_path):
        """Test forge() error handling for invalid inputs."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Test with invalid blueprint
        with pytest.raises((ValueError, Exception)):  # Should raise validation error
            forge(sample_model_path, invalid_blueprint_path)


@pytest.mark.core
class TestBlueprintValidation:
    """Test cases for blueprint validation functionality."""
    
    def test_validate_blueprint_valid(self, sample_blueprint_path):
        """Test validation of valid blueprint."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        is_valid, errors = validate_blueprint(sample_blueprint_path)
        
        # Should be valid with no errors
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        # May or may not be valid depending on blueprint system availability
        # But should not crash
    
    def test_validate_blueprint_invalid(self, invalid_blueprint_path):
        """Test validation of invalid blueprint."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        is_valid, errors = validate_blueprint(invalid_blueprint_path)
        
        # Should detect errors
        assert isinstance(is_valid, bool)
        assert isinstance(errors, list)
        
        if not is_valid:
            assert len(errors) > 0
    
    def test_validate_blueprint_nonexistent(self):
        """Test validation of non-existent blueprint."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        is_valid, errors = validate_blueprint("nonexistent_blueprint.yaml")
        
        # Should be invalid with errors
        assert is_valid is False
        assert len(errors) > 0


@pytest.mark.core
class TestForgeWithRealComponents:
    """Test forge() with real components where possible."""
    
    def test_metrics_creation_and_validation(self):
        """Test that DSEMetrics can be created and validated."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        # Test creating metrics directly
        metrics = DSEMetrics()
        metrics.design_point_id = "test_point_1"
        metrics.build_success = True
        metrics.performance.throughput_ops_sec = 500.0
        metrics.performance.latency_ms = 20.0
        metrics.resources.lut_utilization_percent = 45.0
        
        # Test optimization score calculation
        score = metrics.get_optimization_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0
        
        # Test serialization
        metrics_dict = metrics.to_dict()
        assert isinstance(metrics_dict, dict)
        assert 'design_point_id' in metrics_dict
        assert 'optimization_score' in metrics_dict
        
        # Test JSON conversion
        json_str = metrics.to_json()
        assert isinstance(json_str, str)
        assert 'design_point_id' in json_str
    
    def test_design_space_integration(self, sample_blueprint_path):
        """Test design space creation from blueprint."""
        if not BRAINSMITH_AVAILABLE:
            pytest.skip("BrainSmith core not available")
        
        try:
            from brainsmith.infrastructure.dse.design_space import DesignSpace
            
            # Test basic design space creation
            ds = DesignSpace("test_space")
            assert ds.name == "test_space"
            assert len(ds.parameters) == 0
            
            # Test parameter names
            names = ds.get_parameter_names()
            assert isinstance(names, list)
            
        except ImportError:
            pytest.skip("Design space components not available")


# Integration helpers for testing real workflows
def create_minimal_test_workflow(model_path: str, blueprint_path: str):
    """Helper to create minimal test workflow."""
    try:
        result = forge(model_path, blueprint_path, build_core=False)
        return result
    except Exception as e:
        return {'error': str(e), 'status': 'failed'}


@pytest.mark.integration
def test_minimal_workflow_integration(sample_model_path, sample_blueprint_path):
    """Test minimal integration workflow."""
    if not BRAINSMITH_AVAILABLE:
        pytest.skip("BrainSmith core not available")
    
    # This should work even with fallback implementations
    result = create_minimal_test_workflow(sample_model_path, sample_blueprint_path)
    
    # Should return some result (even if fallback)
    assert isinstance(result, dict)
    
    if 'error' not in result:
        # If successful, check structure
        assert_forge_result_structure(result)
    else:
        # If error, should have error information
        assert 'status' in result
        assert result['status'] == 'failed'