"""
Unit tests for workflow detection.
"""

import pytest
from brainsmith.core.backends.workflow_detector import (
    WorkflowType, detect_workflow, validate_workflow_config
)


class TestWorkflowDetection:
    """Test workflow type detection from blueprints."""
    
    def test_detect_legacy_workflow(self):
        """Test detection of legacy workflow with build_steps."""
        blueprint = {
            'name': 'test_legacy',
            'finn_config': {
                'build_steps': [
                    'step_qonnx_to_finn',
                    'step_streamline',
                    'step_convert_to_hw'
                ]
            }
        }
        
        workflow_type = detect_workflow(blueprint)
        assert workflow_type == WorkflowType.LEGACY
        
    def test_detect_six_entrypoint_workflow(self):
        """Test detection of 6-entrypoint workflow."""
        blueprint = {
            'name': 'test_6ep',
            'nodes': {
                'canonical_ops': ['MatMul', 'LayerNorm'],
                'hw_kernels': ['RTL_Dense', 'HLS_LayerNorm']
            },
            'transforms': {
                'model_topology': ['cleanup', 'streamline'],
                'hw_kernel_transforms': ['buffer_insertion']
            }
        }
        
        workflow_type = detect_workflow(blueprint)
        assert workflow_type == WorkflowType.SIX_ENTRYPOINT
        
    def test_detect_workflow_missing_transforms(self):
        """Test error when only nodes present."""
        blueprint = {
            'name': 'test_invalid',
            'nodes': {
                'canonical_ops': ['MatMul']
            }
            # Missing transforms
        }
        
        with pytest.raises(ValueError) as exc_info:
            detect_workflow(blueprint)
        
        assert "found nodes but not transforms" in str(exc_info.value)
        
    def test_detect_workflow_missing_nodes(self):
        """Test error when only transforms present."""
        blueprint = {
            'name': 'test_invalid',
            'transforms': {
                'model_topology': ['cleanup']
            }
            # Missing nodes
        }
        
        with pytest.raises(ValueError) as exc_info:
            detect_workflow(blueprint)
            
        assert "found transforms but not nodes" in str(exc_info.value)
        
    def test_detect_workflow_no_indicators(self):
        """Test error when no workflow indicators present."""
        blueprint = {
            'name': 'test_empty',
            'objectives': [{'name': 'throughput'}]
        }
        
        with pytest.raises(ValueError) as exc_info:
            detect_workflow(blueprint)
            
        assert "Unable to detect workflow type" in str(exc_info.value)


class TestWorkflowValidation:
    """Test workflow configuration validation."""
    
    def test_validate_legacy_workflow(self):
        """Test validation of legacy workflow configuration."""
        blueprint = {
            'finn_config': {
                'build_steps': ['step1', 'step2']
            }
        }
        
        # Should not raise
        validate_workflow_config(blueprint, WorkflowType.LEGACY)
        
    def test_validate_legacy_empty_steps(self):
        """Test error on empty build_steps."""
        blueprint = {
            'finn_config': {
                'build_steps': []
            }
        }
        
        with pytest.raises(ValueError):
            validate_workflow_config(blueprint, WorkflowType.LEGACY)
        
    def test_validate_legacy_missing_steps(self):
        """Test error on missing build_steps."""
        blueprint = {
            'finn_config': {}
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_config(blueprint, WorkflowType.LEGACY)
            
        assert "requires 'build_steps'" in str(exc_info.value)
        
    def test_validate_six_entrypoint_workflow(self):
        """Test validation of 6-entrypoint workflow configuration."""
        blueprint = {
            'nodes': {
                'canonical_ops': ['MatMul']
            },
            'transforms': {
                'model_topology': ['cleanup']
            }
        }
        
        # Should not raise
        validate_workflow_config(blueprint, WorkflowType.SIX_ENTRYPOINT)
        
    def test_validate_six_entrypoint_missing_nodes(self):
        """Test error on missing nodes section."""
        blueprint = {
            'transforms': {
                'model_topology': ['cleanup']
            }
        }
        
        with pytest.raises(ValueError) as exc_info:
            validate_workflow_config(blueprint, WorkflowType.SIX_ENTRYPOINT)
            
        assert "requires 'nodes' section" in str(exc_info.value)
        
    def test_validate_six_entrypoint_empty_nodes(self):
        """Test error on empty nodes section."""
        blueprint = {
            'nodes': {},
            'transforms': {
                'model_topology': ['cleanup']
            }
        }
        
        with pytest.raises(ValueError):
            validate_workflow_config(blueprint, WorkflowType.SIX_ENTRYPOINT)