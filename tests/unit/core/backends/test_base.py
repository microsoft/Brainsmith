"""
Unit tests for base backend classes.
"""

import pytest
from pathlib import Path
import tempfile

from brainsmith.core.backends.base import (
    EvaluationRequest, EvaluationResult, EvaluationBackend
)


class TestEvaluationRequest:
    """Test EvaluationRequest dataclass."""
    
    def test_valid_request(self):
        """Test creation of valid request."""
        request = EvaluationRequest(
            model_path="/path/to/model.onnx",
            combination={'components': ['test']},
            work_dir="/tmp/work",
            timeout=3600
        )
        
        assert request.model_path == "/path/to/model.onnx"
        assert request.combination == {'components': ['test']}
        assert request.work_dir == "/tmp/work"
        assert request.timeout == 3600
        
    def test_request_path_conversion(self):
        """Test that paths are converted to strings."""
        request = EvaluationRequest(
            model_path=Path("/path/to/model.onnx"),
            combination={},
            work_dir=Path("/tmp/work")
        )
        
        assert isinstance(request.model_path, str)
        assert isinstance(request.work_dir, str)
        
    def test_request_missing_model_path(self):
        """Test error on missing model_path."""
        with pytest.raises(ValueError) as exc_info:
            EvaluationRequest(
                model_path="",
                combination={},
                work_dir="/tmp"
            )
        
        assert "model_path is required" in str(exc_info.value)
        
    def test_request_missing_work_dir(self):
        """Test error on missing work_dir."""
        with pytest.raises(ValueError) as exc_info:
            EvaluationRequest(
                model_path="/model.onnx",
                combination={},
                work_dir=""
            )
        
        assert "work_dir is required" in str(exc_info.value)


class TestEvaluationResult:
    """Test EvaluationResult dataclass."""
    
    def test_successful_result(self):
        """Test creation of successful result."""
        result = EvaluationResult(
            success=True,
            metrics={'throughput': 100.0, 'latency': 5.0},
            reports={'synthesis': '/path/to/synthesis.rpt'},
            warnings=['Warning 1']
        )
        
        assert result.success is True
        assert result.metrics['throughput'] == 100.0
        assert result.reports['synthesis'] == '/path/to/synthesis.rpt'
        assert len(result.warnings) == 1
        assert result.error is None
        
    def test_error_result(self):
        """Test creation of error result."""
        result = EvaluationResult(
            success=False,
            error="FINN build failed"
        )
        
        assert result.success is False
        assert result.error == "FINN build failed"
        assert result.metrics == {}
        assert result.reports == {}
        
    def test_result_to_dict(self):
        """Test conversion to dictionary."""
        result = EvaluationResult(
            success=True,
            metrics={'throughput': 100.0},
            reports={'test': '/path/to/report'},
            warnings=['Warning'],
            error=None
        )
        
        result_dict = result.to_dict()
        
        assert result_dict['success'] is True
        assert result_dict['metrics'] == {'throughput': 100.0}
        assert result_dict['reports'] == {'test': '/path/to/report'}
        assert result_dict['warnings'] == ['Warning']
        assert result_dict['error'] is None
        
    def test_result_from_error(self):
        """Test creation of error result using class method."""
        result = EvaluationResult.from_error("Test error")
        
        assert result.success is False
        assert result.error == "Test error"
        assert result.metrics == {}
        assert result.reports == {}


class TestEvaluationBackend:
    """Test EvaluationBackend abstract base class."""
    
    class ConcreteBackend(EvaluationBackend):
        """Concrete implementation for testing."""
        
        def evaluate(self, request):
            return EvaluationResult(success=True)
    
    def test_backend_initialization(self):
        """Test backend initialization with config."""
        config = {'test': 'config'}
        backend = self.ConcreteBackend(config)
        
        assert backend.blueprint_config == config
        
    def test_validate_request_valid(self):
        """Test validation of valid request."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a temporary model file
            model_path = Path(temp_dir) / "model.onnx"
            model_path.touch()
            
            request = EvaluationRequest(
                model_path=str(model_path),
                combination={},
                work_dir=temp_dir,
                timeout=100
            )
            
            backend = self.ConcreteBackend({})
            error = backend.validate_request(request)
            
            assert error is None
            
    def test_validate_request_missing_model(self):
        """Test validation error for missing model."""
        with tempfile.TemporaryDirectory() as temp_dir:
            request = EvaluationRequest(
                model_path="/nonexistent/model.onnx",
                combination={},
                work_dir=temp_dir
            )
            
            backend = self.ConcreteBackend({})
            error = backend.validate_request(request)
            
            assert error is not None
            assert "Model file not found" in error
            
    def test_validate_request_invalid_work_dir(self):
        """Test validation error for invalid work directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            model_path.touch()
            
            request = EvaluationRequest(
                model_path=str(model_path),
                combination={},
                work_dir="/nonexistent/dir"
            )
            
            backend = self.ConcreteBackend({})
            error = backend.validate_request(request)
            
            assert error is not None
            assert "Work directory does not exist" in error
            
    def test_validate_request_negative_timeout(self):
        """Test validation error for negative timeout."""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "model.onnx"
            model_path.touch()
            
            request = EvaluationRequest(
                model_path=str(model_path),
                combination={},
                work_dir=temp_dir,
                timeout=-100
            )
            
            backend = self.ConcreteBackend({})
            error = backend.validate_request(request)
            
            assert error is not None
            assert "Invalid timeout" in error