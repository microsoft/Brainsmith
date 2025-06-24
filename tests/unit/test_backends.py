"""
Unit tests for backend module.
"""

import pytest
import tempfile
from pathlib import Path
from typing import Dict, Any

from brainsmith.core.backends import (
    WorkflowType,
    detect_workflow,
    create_backend,
    EvaluationRequest,
    EvaluationResult,
    EvaluationBackend
)
from brainsmith.core.backends.six_entrypoint import SixEntrypointBackend
from brainsmith.core.backends.legacy_finn import LegacyFINNBackend


class TestWorkflowDetection:
    """Test workflow detection functionality."""
    
    def test_detect_legacy_workflow(self):
        """Test detection of legacy workflow."""
        blueprint = {
            'finn_config': {
                'build_steps': ['step1', 'step2']
            }
        }
        assert detect_workflow(blueprint) == WorkflowType.LEGACY
    
    def test_detect_six_entrypoint_workflow(self):
        """Test detection of 6-entrypoint workflow."""
        blueprint = {
            'design_space': {
                'components': {
                    'component1': {
                        'type': 'transform',
                        'nodes': ['node1'],
                        'transforms': ['transform1']
                    }
                }
            }
        }
        assert detect_workflow(blueprint) == WorkflowType.SIX_ENTRYPOINT
    
    def test_detect_workflow_with_empty_blueprint(self):
        """Test detection with empty blueprint."""
        with pytest.raises(ValueError):
            detect_workflow({})
    
    def test_detect_workflow_with_invalid_structure(self):
        """Test detection with invalid structure."""
        blueprint = {
            'design_space': {
                'components': {
                    'component1': {
                        'type': 'transform',
                        'nodes': [],  # Empty nodes
                        'transforms': []  # Empty transforms
                    }
                }
            }
        }
        with pytest.raises(ValueError):
            detect_workflow(blueprint)


class TestBackendFactory:
    """Test backend factory functionality."""
    
    def test_create_six_entrypoint_backend(self):
        """Test creation of 6-entrypoint backend."""
        blueprint = {
            'design_space': {
                'components': {
                    'component1': {
                        'type': 'transform',
                        'nodes': ['node1'],
                        'transforms': ['transform1']
                    }
                }
            }
        }
        backend = create_backend(blueprint)
        assert isinstance(backend, SixEntrypointBackend)
    
    def test_create_legacy_backend(self):
        """Test creation of legacy backend."""
        blueprint = {
            'finn_config': {
                'build_steps': ['step1', 'step2']
            }
        }
        backend = create_backend(blueprint)
        assert isinstance(backend, LegacyFINNBackend)
    
    def test_create_backend_with_invalid_workflow(self):
        """Test backend creation with invalid workflow."""
        with pytest.raises(ValueError):
            create_backend({})


class TestEvaluationRequest:
    """Test EvaluationRequest dataclass."""
    
    def test_create_evaluation_request(self):
        """Test creating an evaluation request."""
        request = EvaluationRequest(
            model_path="/path/to/model.onnx",
            combination={'component1': 'value1'},
            work_dir="/tmp/work"
        )
        assert request.model_path == "/path/to/model.onnx"
        assert request.combination == {'component1': 'value1'}
        assert request.work_dir == "/tmp/work"
        assert request.timeout is None
    
    def test_create_evaluation_request_with_timeout(self):
        """Test creating an evaluation request with timeout."""
        request = EvaluationRequest(
            model_path="/path/to/model.onnx",
            combination={'component1': 'value1'},
            work_dir="/tmp/work",
            timeout=300
        )
        assert request.timeout == 300


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_create_successful_result(self):
        """Test creating a successful result."""
        result = EvaluationResult(
            success=True,
            metrics={'throughput': 100.0}
        )
        assert result.success is True
        assert result.metrics == {'throughput': 100.0}
        assert result.error is None
        assert result.warnings == []
    
    def test_create_failed_result(self):
        """Test creating a failed result."""
        result = EvaluationResult(
            success=False,
            error="Build failed"
        )
        assert result.success is False
        assert result.error == "Build failed"
        assert result.metrics == {}
        assert result.warnings == []


class TestSixEntrypointBackend:
    """Test SixEntrypointBackend functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.blueprint = {
            'design_space': {
                'components': {
                    'transforms': {
                        'type': 'transform',
                        'nodes': ['FoldConstants'],
                        'transforms': ['FoldConstants']
                    }
                }
            }
        }
        self.backend = SixEntrypointBackend(self.blueprint)
    
    def test_initialization(self):
        """Test backend initialization."""
        assert isinstance(self.backend, EvaluationBackend)
        assert self.backend.blueprint_config == self.blueprint
    
    def test_extract_entrypoint_config(self):
        """Test entrypoint configuration extraction."""
        combination = {
            'entrypoint_1': ['FoldConstants'],
            'entrypoint_2': ['InferShapes'],
            'entrypoint_3': ['kernel1'],
            'entrypoint_4': [],
            'entrypoint_5': ['MinimizeBitWidth'],
            'entrypoint_6': ['Streamline']
        }
        
        config = self.backend._extract_entrypoint_config(combination)
        assert config.entrypoint_1 == ['FoldConstants']
        assert config.entrypoint_2 == ['InferShapes']
        assert config.entrypoint_3 == ['kernel1']
        assert config.entrypoint_4 == []
        assert config.entrypoint_5 == ['MinimizeBitWidth']
        assert config.entrypoint_6 == ['Streamline']
    
    def test_extract_entrypoint_config_legacy_names(self):
        """Test entrypoint config extraction with legacy attribute names."""
        combination = {
            'canonical_ops': ['op1', 'op2'],
            'model_topology': ['topo1'],
            'hw_kernels': {'kernel1': 'spec1'},
            'hw_kernel_transforms': ['transform1'],
            'hw_graph_transforms': ['graph1']
        }
        
        config = self.backend._extract_entrypoint_config(combination)
        assert config.entrypoint_1 == ['op1', 'op2']
        assert config.entrypoint_2 == ['topo1']
        assert config.entrypoint_3 == ['kernel1']
        assert config.entrypoint_5 == ['transform1']
        assert config.entrypoint_6 == ['graph1']
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid configuration
        config = {
            'entrypoint_1': ['FoldConstants', 'InferShapes'],
            'entrypoint_2': ['Streamline']
        }
        errors = self.backend.validate_configuration(config)
        assert errors == []
        
        # Invalid transform names
        config = {
            'entrypoint_1': ['InvalidTransform']
        }
        errors = self.backend.validate_configuration(config)
        assert len(errors) == 1
        assert 'Unknown transform' in errors[0]
        
        # Non-list entrypoint
        config = {
            'entrypoint_1': 'not_a_list'
        }
        errors = self.backend.validate_configuration(config)
        assert len(errors) == 1
        assert 'must be a list' in errors[0]


class TestLegacyFINNBackend:
    """Test LegacyFINNBackend functionality."""
    
    def setup_method(self):
        """Set up test environment."""
        self.blueprint = {
            'legacy_preproc': ['onnx_preprocessing_step', 'cleanup_step'],
            'legacy_postproc': ['shell_metadata_handover_step'],
            'finn_config': {
                'output_dir': './output',
                'synth_clk_period_ns': 5.0
            }
        }
        self.backend = LegacyFINNBackend(self.blueprint)
    
    def test_initialization(self):
        """Test backend initialization."""
        assert isinstance(self.backend, EvaluationBackend)
        assert self.backend.blueprint_config == self.blueprint
        assert hasattr(self.backend, 'legacy_converter')
        assert hasattr(self.backend, 'metrics_extractor')
    
    def test_combination_to_entrypoint_config_dict(self):
        """Test conversion of dict combination to entrypoint config."""
        combination = {
            'canonical_ops': ['op1'],
            'model_topology': ['topo1'],
            'hw_kernels': ['kernel1'],
            'hw_kernel_specializations': ['spec1'],
            'hw_kernel_transforms': ['transform1'],
            'hw_graph_transforms': ['graph1']
        }
        
        config = self.backend._combination_to_entrypoint_config(combination)
        assert config['entrypoint_1'] == ['op1']
        assert config['entrypoint_2'] == ['topo1']
        assert config['entrypoint_3'] == ['kernel1']
        assert config['entrypoint_4'] == ['spec1']
        assert config['entrypoint_5'] == ['transform1']
        assert config['entrypoint_6'] == ['graph1']
    
    def test_validate_configuration(self):
        """Test configuration validation."""
        # Valid configuration (has legacy_preproc)
        errors = self.backend.validate_configuration({})
        assert errors == []
        
        # Invalid build_steps type
        errors = self.backend.validate_configuration({'build_steps': 'not_a_list'})
        assert len(errors) == 1
        assert "'build_steps' must be a list" in errors[0]
        
        # Missing legacy sections
        backend_no_legacy = LegacyFINNBackend({})
        errors = backend_no_legacy.validate_configuration({})
        assert len(errors) == 1
        assert 'legacy_preproc' in errors[0]
    
    def test_serialize_dataflow_config(self):
        """Test DataflowBuildConfig serialization."""
        # Create mock dataflow config
        class MockStep:
            def __init__(self, name):
                self.__name__ = name
        
        class MockDataflowConfig:
            def __init__(self):
                self.output_dir = './output'
                self.synth_clk_period_ns = 5.0
                self.target_fps = 100
                self.steps = [MockStep('step1'), MockStep('step2')]
        
        config = MockDataflowConfig()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            config_file = Path(tmpdir) / "config.json"
            self.backend._serialize_dataflow_config(config, config_file)
            
            assert config_file.exists()
            import json
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            assert data['output_dir'] == './output'
            assert data['synth_clk_period_ns'] == 5.0
            assert data['target_fps'] == 100
            assert data['steps'] == ['step1', 'step2']