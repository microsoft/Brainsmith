"""
Unit tests for Future FINN-Brainsmith Backend.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import json
from datetime import datetime

from brainsmith.core.phase1.data_structures import (
    GlobalConfig,
    OutputStage,
    ProcessingStep,
)
from brainsmith.core.phase2.data_structures import BuildConfig
from brainsmith.core.phase3.data_structures import BuildStatus, BuildMetrics
from brainsmith.core.phase3.future_brainsmith_backend import FutureBrainsmithBackend


class TestFutureBrainsmithBackend:
    """Test FutureBrainsmithBackend class."""
    
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
            kernels=[
                ("MatrixVectorUnit", ["8", "4"]),
                ("ConvolutionUnit", ["16", "8", "3"]),
            ],
            transforms={"default": ["AbsorbSignBias", "MoveLinearPastFork", "InferDataLayouts"]},
            preprocessing=[
                ProcessingStep(name="graph_optimization", type="preprocessing", enabled=True),
                ProcessingStep(name="quantize_model", type="preprocessing", enabled=True, parameters={"bits": 8}),
            ],
            postprocessing=[
                ProcessingStep(name="performance_analysis", type="postprocessing", enabled=True),
                ProcessingStep(name="resource_analysis", type="postprocessing", enabled=True),
            ],
            build_steps=[
                "step_qonnx_to_finn",
                "step_streamline",
                "step_convert_to_hw",
            ],
            config_flags={
                "clock_period_ns": 5.0,
                "target_fps": 1000,
                "optimization_level": 3,
            },
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory="/tmp/dse_work",
                cache_results=True,
                save_artifacts=True,
                log_level="INFO",
            ),
            timestamp=datetime.now(),
            combination_index=5,
            total_combinations=20,
            output_dir="/tmp/test_output",
        )
    
    def test_backend_creation(self):
        """Test creating FutureBrainsmithBackend."""
        backend = FutureBrainsmithBackend()
        
        assert backend.mock_success_rate == 0.9
        assert backend.mock_build_time_range == (10.0, 60.0)
    
    def test_backend_creation_custom_params(self):
        """Test creating backend with custom parameters."""
        backend = FutureBrainsmithBackend(
            mock_success_rate=0.7,
            mock_build_time_range=(5.0, 30.0)
        )
        
        assert backend.mock_success_rate == 0.7
        assert backend.mock_build_time_range == (5.0, 30.0)
    
    def test_get_backend_name(self):
        """Test get_backend_name method."""
        backend = FutureBrainsmithBackend()
        assert backend.get_backend_name() == "FINN-Brainsmith Direct (Stub)"
    
    def test_get_supported_output_stages(self):
        """Test get_supported_output_stages method."""
        backend = FutureBrainsmithBackend()
        stages = backend.get_supported_output_stages()
        
        assert len(stages) == 3
        assert OutputStage.DATAFLOW_GRAPH in stages
        assert OutputStage.RTL in stages
        assert OutputStage.STITCHED_IP in stages
    
    def test_prepare_finn_brainsmith_config(self, mock_config):
        """Test _prepare_finn_brainsmith_config method."""
        backend = FutureBrainsmithBackend()
        
        config = backend._prepare_finn_brainsmith_config(mock_config)
        
        # Check structure
        assert "kernels" in config
        assert "transform_stages" in config
        assert "output_stage" in config
        assert "design_space_id" in config
        
        # Check kernels marshaling
        assert len(config["kernels"]) == 2
        assert config["kernels"][0]["name"] == "MatrixVectorUnit"
        assert config["kernels"][0]["parameters"] == ["8", "4"]
        assert config["kernels"][0]["metadata"]["index"] == 0
        assert config["kernels"][0]["metadata"]["type"] == "hw_kernel"
        
        assert config["kernels"][1]["name"] == "ConvolutionUnit"
        assert config["kernels"][1]["parameters"] == ["16", "8", "3"]
        
        # Check transforms marshaling
        assert config["transform_stages"] == {"default": ["AbsorbSignBias", "MoveLinearPastFork", "InferDataLayouts"]}
        
        # Check global config marshaling
        assert config["output_stage"] == "RTL"
        assert config["target_clock_ns"] == 5.0  # From config_flags
        assert "design_space_id" in config
        assert "combination_index" in config
        
        # Check metadata
        assert config["design_space_id"] == "test_ds_001"
        assert config["combination_index"] == 5
        assert config["total_combinations"] == 20
        assert "timestamp" in config
        assert "api_version" in config
    
    @patch('brainsmith.core.phase3.future_brainsmith_backend.Path.mkdir')
    def test_run_method_success(self, mock_mkdir, temp_dir, mock_config):
        """Test successful run execution."""
        backend = FutureBrainsmithBackend(mock_success_rate=1.0)  # Always succeed
        
        # Use actual temp directory for file operations
        mock_config.global_config.working_directory = temp_dir
        mock_config.output_dir = str(Path(temp_dir) / "output")
        
        # Run the backend with preprocessed model path
        result = backend.run(mock_config, "/path/to/preprocessed_model.onnx")
        
        # Verify result
        assert result.config_id == "test_config_001"
        assert result.status == BuildStatus.SUCCESS
        assert result.metrics is not None
        assert result.metrics.throughput > 0
        assert result.metrics.lut_utilization > 0
        assert len(result.artifacts) > 0
        assert result.error_message is None
        
        # Check that config was saved in the output dir
        config_file = Path(mock_config.output_dir) / "finn_brainsmith_config.json"
        assert config_file.exists()
        
        # Load and verify saved config
        with open(config_file) as f:
            saved_config = json.load(f)
        assert saved_config["design_space_id"] == "test_ds_001"
    
    @patch('brainsmith.core.phase3.future_brainsmith_backend.Path.mkdir')
    def test_run_method_failure(self, mock_mkdir, temp_dir, mock_config):
        """Test run execution with simulated failure."""
        backend = FutureBrainsmithBackend(mock_success_rate=0.0)  # Always fail
        
        # Use actual temp directory
        mock_config.global_config.working_directory = temp_dir
        mock_config.output_dir = str(Path(temp_dir) / "output")
        
        # Run the backend with preprocessed model path
        result = backend.run(mock_config, "/path/to/preprocessed_model.onnx")
        
        # Verify failure result
        assert result.config_id == "test_config_001"
        assert result.status == BuildStatus.FAILED
        assert result.metrics is None
        assert "mock failure" in result.error_message
        assert result.duration_seconds >= 0  # May be less than 10 due to sleep cap
    
    def test_generate_mock_metrics_complexity(self, mock_config):
        """Test that mock metrics correlate with complexity."""
        backend = FutureBrainsmithBackend()
        
        # Simple config (1 kernel, 1 transform)
        simple_config = BuildConfig(
            id="simple",
            design_space_id="test",
            kernels=[("MatrixVectorUnit", ["4", "2"])],
            transforms={"default": ["AbsorbSignBias"]},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
            config_flags={},
            global_config=mock_config.global_config,
            timestamp=datetime.now(),
            combination_index=0,
            total_combinations=1,
            output_dir="/tmp/simple_output",
        )
        
        simple_metrics = backend._generate_mock_metrics(simple_config)
        
        # Complex config (original mock_config with 2 kernels, 3 transforms)
        complex_metrics = backend._generate_mock_metrics(mock_config)
        
        # Complex config should have worse metrics (more resource usage)
        # Due to randomness, we can't guarantee exact ordering, but complexity factor should be higher
        assert complex_metrics.raw_metrics["complexity_factor"] > simple_metrics.raw_metrics["complexity_factor"]
        assert complex_metrics.raw_metrics["kernel_count"] > simple_metrics.raw_metrics["kernel_count"]
        assert complex_metrics.raw_metrics["transform_count"] > simple_metrics.raw_metrics["transform_count"]
    
    def test_generate_mock_metrics_output_stage(self, mock_config):
        """Test that mock metrics vary by output stage."""
        backend = FutureBrainsmithBackend()
        
        # RTL stage
        mock_config.global_config.output_stage = OutputStage.RTL
        rtl_metrics = backend._generate_mock_metrics(mock_config)
        
        # STITCHED_IP stage (should have synthesis data)
        mock_config.global_config.output_stage = OutputStage.STITCHED_IP
        ip_metrics = backend._generate_mock_metrics(mock_config)
        
        # STITCHED_IP should have actual timing data
        assert ip_metrics.clock_frequency is not None
        assert ip_metrics.clock_frequency > 0
        
        # Both should have resource estimates
        assert rtl_metrics.lut_utilization is not None
        assert ip_metrics.lut_utilization is not None
    
    def test_generate_mock_artifacts(self, temp_dir, mock_config):
        """Test _generate_mock_artifacts method."""
        backend = FutureBrainsmithBackend()
        
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True)
        
        artifacts = backend._generate_mock_artifacts(str(output_dir))
        
        # Check returned artifacts
        assert "design_summary" in artifacts
        assert "kernel_mapping" in artifacts
        assert "transform_log" in artifacts
        assert "performance_report" in artifacts
        assert "resource_report" in artifacts
        
        # Check files were created
        assert Path(artifacts["design_summary"]).exists()
        assert Path(artifacts["kernel_mapping"]).exists()
        assert Path(artifacts["transform_log"]).exists()
        
        # Verify design summary content
        with open(artifacts["design_summary"]) as f:
            summary_data = json.load(f)
        assert summary_data["generated_by"] == "FINN-Brainsmith Direct (Stub)"
        assert summary_data["mock_data"] is True
        assert "design_stats" in summary_data
    
    def test_execute_finn_brainsmith_build_timing(self, temp_dir, mock_config):
        """Test that build execution respects timing parameters."""
        # Fast build time
        fast_backend = FutureBrainsmithBackend(
            mock_success_rate=1.0,
            mock_build_time_range=(0.1, 0.2)
        )
        
        output_dir = Path(temp_dir) / "fast"
        output_dir.mkdir(parents=True)
        
        import time
        start = time.time()
        fast_backend._execute_finn_brainsmith_build(
            model_path="/path/to/model.onnx",
            config={
                "output_dir": str(output_dir),
                "kernels": [],
                "transform_stages": {}
            }
        )
        fast_duration = time.time() - start
        
        # Slow build time
        slow_backend = FutureBrainsmithBackend(
            mock_success_rate=1.0,
            mock_build_time_range=(0.5, 0.6)
        )
        
        output_dir = Path(temp_dir) / "slow"
        output_dir.mkdir(parents=True)
        
        start = time.time()
        slow_backend._execute_finn_brainsmith_build(
            model_path="/path/to/model.onnx",
            config={
                "output_dir": str(output_dir),
                "kernels": [],
                "transform_stages": {}
            }
        )
        slow_duration = time.time() - start
        
        # Verify timing
        assert 0.1 <= fast_duration <= 0.3  # Some overhead
        assert 0.5 <= slow_duration <= 0.8  # Some overhead
        assert slow_duration > fast_duration