"""
Unit tests for BuildRunner orchestrator.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
from datetime import datetime

from brainsmith.core_v3.phase1.data_structures import (
    GlobalConfig,
    OutputStage,
    ProcessingStep,
)
from brainsmith.core_v3.phase2.data_structures import BuildConfig
from brainsmith.core_v3.phase3.data_structures import BuildStatus, BuildMetrics, BuildResult
from brainsmith.core_v3.phase3.build_runner import BuildRunner
from brainsmith.core_v3.phase3.interfaces import BuildRunnerInterface


class MockBackend(BuildRunnerInterface):
    """Mock backend for testing BuildRunner."""
    
    def __init__(self, should_succeed=True):
        self.should_succeed = should_succeed
        self.run_called = False
        self.run_args = None
        
    def get_backend_name(self) -> str:
        return "Mock Backend"
        
    def get_supported_output_stages(self):
        return [OutputStage.RTL]
        
    def run(self, config: BuildConfig, model_path: str) -> BuildResult:
        self.run_called = True
        self.run_args = (config, model_path)
        
        result = BuildResult(config_id=config.id)
        
        if self.should_succeed:
            result.metrics = BuildMetrics(
                throughput=1000.0,
                latency=10.0,
                lut_utilization=0.5
            )
            result.complete(BuildStatus.SUCCESS)
        else:
            result.complete(BuildStatus.FAILED, "Mock failure")
            
        return result


class TestBuildRunner:
    """Test BuildRunner orchestrator class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self):
        """Create a mock BuildConfig."""
        return BuildConfig(
            id="test_config_001",
            design_space_id="test_ds_001",
            kernels=[("MatrixVectorUnit", ["8", "4"])],
            transforms={"default": ["AbsorbSignBias"]},
            preprocessing=[
                ProcessingStep(name="graph_optimization", type="preprocessing", enabled=True),
                ProcessingStep(name="quantize_model", type="preprocessing", enabled=True),
            ],
            postprocessing=[
                ProcessingStep(name="performance_analysis", type="postprocessing", enabled=True),
            ],
            build_steps=["step_streamline"],
            config_flags={},
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory="/tmp/dse_work",
                cache_results=True,
                save_artifacts=True,
                log_level="INFO",
            ),
            timestamp=datetime.now(),
            combination_index=1,
            total_combinations=10,
            output_dir="/tmp/test_output",
        )
    
    def test_build_runner_creation(self):
        """Test creating BuildRunner with backend."""
        backend = MockBackend()
        runner = BuildRunner(backend)
        
        assert runner.backend is backend
        assert runner.preprocessing_pipeline is not None
        assert runner.postprocessing_pipeline is not None
    
    @patch('brainsmith.core_v3.phase3.build_runner.PreprocessingPipeline')
    @patch('brainsmith.core_v3.phase3.build_runner.PostprocessingPipeline')
    def test_run_success(self, mock_postproc_class, mock_preproc_class, mock_config):
        """Test successful build execution through BuildRunner."""
        # Create mocks
        backend = MockBackend(should_succeed=True)
        
        # Mock preprocessing pipeline
        mock_preproc = Mock()
        mock_preproc.execute.return_value = "/path/to/preprocessed_model.onnx"
        mock_preproc_class.return_value = mock_preproc
        
        # Mock postprocessing pipeline
        mock_postproc = Mock()
        mock_postproc_class.return_value = mock_postproc
        
        # Create and run BuildRunner
        runner = BuildRunner(backend)
        result = runner.run(mock_config, "/path/to/original_model.onnx")
        
        # Verify preprocessing was called
        mock_preproc.execute.assert_called_once_with(mock_config, "/path/to/original_model.onnx")
        
        # Verify backend was called with preprocessed model
        assert backend.run_called
        assert backend.run_args[0] == mock_config
        assert backend.run_args[1] == "/path/to/preprocessed_model.onnx"
        
        # Verify postprocessing was called (since build succeeded)
        mock_postproc.analyze.assert_called_once_with(mock_config, result)
        
        # Verify result
        assert result.status == BuildStatus.SUCCESS
        assert result.metrics is not None
        assert result.metrics.throughput == 1000.0
    
    @patch('brainsmith.core_v3.phase3.build_runner.PreprocessingPipeline')
    @patch('brainsmith.core_v3.phase3.build_runner.PostprocessingPipeline')
    def test_run_failure(self, mock_postproc_class, mock_preproc_class, mock_config):
        """Test failed build execution through BuildRunner."""
        # Create mocks
        backend = MockBackend(should_succeed=False)
        
        # Mock preprocessing pipeline
        mock_preproc = Mock()
        mock_preproc.execute.return_value = "/path/to/preprocessed_model.onnx"
        mock_preproc_class.return_value = mock_preproc
        
        # Mock postprocessing pipeline
        mock_postproc = Mock()
        mock_postproc_class.return_value = mock_postproc
        
        # Create and run BuildRunner
        runner = BuildRunner(backend)
        result = runner.run(mock_config, "/path/to/original_model.onnx")
        
        # Verify preprocessing was called
        mock_preproc.execute.assert_called_once_with(mock_config, "/path/to/original_model.onnx")
        
        # Verify backend was called
        assert backend.run_called
        
        # Verify postprocessing was NOT called (since build failed)
        mock_postproc.analyze.assert_not_called()
        
        # Verify result
        assert result.status == BuildStatus.FAILED
        assert result.error_message == "Mock failure"
    
    @patch('brainsmith.core_v3.phase3.build_runner.PreprocessingPipeline')
    @patch('brainsmith.core_v3.phase3.build_runner.PostprocessingPipeline')
    def test_preprocessing_failure(self, mock_postproc_class, mock_preproc_class, mock_config):
        """Test handling of preprocessing failure."""
        # Create mocks
        backend = MockBackend(should_succeed=True)
        
        # Mock preprocessing pipeline that fails
        mock_preproc = Mock()
        mock_preproc.execute.side_effect = Exception("Preprocessing failed")
        mock_preproc_class.return_value = mock_preproc
        
        # Mock postprocessing pipeline
        mock_postproc = Mock()
        mock_postproc_class.return_value = mock_postproc
        
        # Create and run BuildRunner
        runner = BuildRunner(backend)
        result = runner.run(mock_config, "/path/to/original_model.onnx")
        
        # Verify backend was NOT called (preprocessing failed)
        assert not backend.run_called
        
        # Verify postprocessing was NOT called
        mock_postproc.analyze.assert_not_called()
        
        # Verify result
        assert result.status == BuildStatus.FAILED
        assert "Preprocessing failed" in result.error_message
    
    @patch('brainsmith.core_v3.phase3.build_runner.PreprocessingPipeline')
    @patch('brainsmith.core_v3.phase3.build_runner.PostprocessingPipeline')
    def test_postprocessing_failure(self, mock_postproc_class, mock_preproc_class, mock_config):
        """Test handling of postprocessing failure (should not affect result)."""
        # Create mocks
        backend = MockBackend(should_succeed=True)
        
        # Mock preprocessing pipeline
        mock_preproc = Mock()
        mock_preproc.execute.return_value = "/path/to/preprocessed_model.onnx"
        mock_preproc_class.return_value = mock_preproc
        
        # Mock postprocessing pipeline that fails
        mock_postproc = Mock()
        mock_postproc.analyze.side_effect = Exception("Postprocessing failed")
        mock_postproc_class.return_value = mock_postproc
        
        # Create and run BuildRunner
        runner = BuildRunner(backend)
        result = runner.run(mock_config, "/path/to/original_model.onnx")
        
        # Verify all steps were called
        assert backend.run_called
        mock_postproc.analyze.assert_called_once()
        
        # Verify result is still successful (postprocessing failure doesn't affect build result)
        assert result.status == BuildStatus.SUCCESS
        assert result.metrics is not None
    
    def test_backend_delegation(self):
        """Test that BuildRunner properly delegates to backend methods."""
        backend = MockBackend()
        runner = BuildRunner(backend)
        
        # Test get_backend_name delegation
        assert runner.get_backend_name() == "Mock Backend"
        
        # Test get_supported_output_stages delegation
        assert runner.get_supported_output_stages() == [OutputStage.RTL]