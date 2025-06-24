"""
Unit tests for backend factory.
"""

import pytest

from brainsmith.core.backends.factory import create_backend, get_backend_info
from brainsmith.core.backends.workflow_detector import WorkflowType
from brainsmith.core.backends.base import EvaluationBackend


class TestBackendFactory:
    """Test backend factory functionality."""
    
    def test_create_backend_six_entrypoint(self):
        """Test creation of 6-entrypoint backend."""
        blueprint = {
            'name': 'test_6ep',
            'nodes': {'canonical_ops': ['MatMul']},
            'transforms': {'model_topology': ['cleanup']}
        }
        
        backend = create_backend(blueprint)
        
        # Check it's the right type (will be SixEntrypointBackend when implemented)
        assert isinstance(backend, EvaluationBackend)
        assert backend.blueprint_config == blueprint
        
    def test_create_backend_legacy(self):
        """Test creation of legacy backend."""
        blueprint = {
            'name': 'test_legacy',
            'finn_config': {
                'build_steps': ['step1', 'step2']
            }
        }
        
        backend = create_backend(blueprint)
        
        # Check it's the right type (will be LegacyFINNBackend when implemented)
        assert isinstance(backend, EvaluationBackend)
        assert backend.blueprint_config == blueprint
        
    def test_create_backend_invalid_workflow(self):
        """Test error on invalid workflow detection."""
        blueprint = {
            'name': 'test_invalid'
        }
        
        with pytest.raises(ValueError) as exc_info:
            create_backend(blueprint)
            
        assert "Unable to detect workflow type" in str(exc_info.value)
        
    def test_create_backend_validation_failure(self):
        """Test error on workflow validation failure."""
        blueprint = {
            'finn_config': {
                'build_steps': []  # Empty steps should fail validation
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            create_backend(blueprint)
            
        assert "cannot be empty" in str(exc_info.value)
    


class TestBackendInfo:
    """Test backend information helper."""
    
    def test_get_backend_info_six_entrypoint(self):
        """Test info for 6-entrypoint workflow."""
        blueprint = {
            'nodes': {'canonical_ops': ['MatMul']},
            'transforms': {'model_topology': ['cleanup']}
        }
        
        info = get_backend_info(blueprint)
        
        assert info['workflow_type'] == 'six_entrypoint'
        assert info['backend_class'] == 'SixEntrypointBackend'
        assert info['valid'] is True
        assert info['error'] is None
        
    def test_get_backend_info_legacy(self):
        """Test info for legacy workflow."""
        blueprint = {
            'finn_config': {
                'build_steps': ['step1']
            }
        }
        
        info = get_backend_info(blueprint)
        
        assert info['workflow_type'] == 'legacy'
        assert info['backend_class'] == 'LegacyFINNBackend'
        assert info['valid'] is True
        assert info['error'] is None
        
    def test_get_backend_info_invalid(self):
        """Test info for invalid blueprint."""
        blueprint = {}
        
        info = get_backend_info(blueprint)
        
        assert info['workflow_type'] is None
        assert info['backend_class'] is None
        assert info['valid'] is False
        assert info['error'] is not None
        assert "Unable to detect workflow type" in info['error']