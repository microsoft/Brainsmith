"""
Unit tests for Legacy FINN Backend.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import tempfile
import shutil
import os
from datetime import datetime

from brainsmith.core_v3.phase1.data_structures import (
    GlobalConfig,
    OutputStage,
    ProcessingStep,
)
from brainsmith.core_v3.phase2.data_structures import BuildConfig
from brainsmith.core_v3.phase3.data_structures import BuildStatus, BuildMetrics
from brainsmith.core_v3.phase3.legacy_finn_backend import LegacyFINNBackend
from brainsmith.core_v3.phase3.metrics_collector import MetricsCollector


class TestLegacyFINNBackend:
    """Test LegacyFINNBackend class."""
    
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
            transforms={"default": ["AbsorbSignBias", "MoveLinearPastFork"]},
            preprocessing=[
                ProcessingStep(name="graph_optimization", type="preprocessing", enabled=True),
                ProcessingStep(name="quantize_model", type="preprocessing", enabled=True, parameters={"bits": 8}),
            ],
            postprocessing=[
                ProcessingStep(name="performance_analysis", type="postprocessing", enabled=True),
            ],
            build_steps=[
                "step_qonnx_to_finn",
                "step_streamline",
                "step_convert_to_hw",
                "step_create_dataflow_partition",
                "step_specialize_layers",
                "step_apply_folding_config",
                "step_generate_estimate_reports",
                "step_hw_codegen",
                "step_hw_ipgen",
                "step_make_zynq_proj",
                "step_synthesize_bitfile",
            ],
            config_flags={
                "clock_period_ns": 5.0,
                "board": "Pynq-Z1",
                "shell_flow_type": "vivado_zynq",
            },
            global_config=GlobalConfig(
                output_stage=OutputStage.STITCHED_IP,
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
    
    def test_backend_creation(self, temp_dir):
        """Test creating LegacyFINNBackend."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        assert backend.finn_build_dir == temp_dir
        assert backend.temp_cleanup is True
        assert backend.preserve_intermediate is False
    
    def test_backend_creation_no_cleanup(self, temp_dir):
        """Test creating backend without temp cleanup."""
        backend = LegacyFINNBackend(
            finn_build_dir=temp_dir,
            temp_cleanup=False
        )
        
        assert backend.temp_cleanup is False
    
    def test_get_backend_name(self, temp_dir):
        """Test get_backend_name method."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        assert backend.get_backend_name() == "FINN Legacy Builder"
    
    def test_get_supported_output_stages(self, temp_dir):
        """Test get_supported_output_stages method."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        stages = backend.get_supported_output_stages()
        
        assert len(stages) == 2
        assert OutputStage.RTL in stages
        assert OutputStage.STITCHED_IP in stages
    
    def test_create_dataflow_config(self, temp_dir, mock_config):
        """Test _create_dataflow_config method."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Test with STITCHED_IP output stage
        dataflow_cfg = backend._create_dataflow_config(mock_config)
        
        assert dataflow_cfg["output_dir"] == "/tmp/test_output"
        assert dataflow_cfg["synth_clk_period_ns"] == 5.0
        assert dataflow_cfg["board"] == "Pynq-Z1"
        assert dataflow_cfg["shell_flow_type"] == "vivado_zynq"
        assert dataflow_cfg["generate_outputs"] == [
            "inference_cost",
            "rtlsim_perf",
            "rtlsim_reports",
            "bitfile",
            "pynq_driver",
            "deployment_package",
        ]
        assert len(dataflow_cfg["steps"]) == 11
    
    def test_create_dataflow_config_rtl_output(self, temp_dir, mock_config):
        """Test _create_dataflow_config with RTL output stage."""
        # Modify config for RTL output
        mock_config.global_config.output_stage = OutputStage.RTL
        
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        dataflow_cfg = backend._create_dataflow_config(mock_config)
        
        # Should stop at HW IP generation
        assert dataflow_cfg["generate_outputs"] == [
            "inference_cost",
            "rtlsim_perf",
            "rtlsim_reports",
        ]
        assert "step_hw_ipgen" in dataflow_cfg["steps"]
        assert "step_make_zynq_proj" not in dataflow_cfg["steps"]
        assert "step_synthesize_bitfile" not in dataflow_cfg["steps"]
    
    def test_create_dataflow_config_default_clock(self, temp_dir, mock_config):
        """Test _create_dataflow_config with default clock period."""
        # Remove clock period from config
        del mock_config.config_flags["clock_period_ns"]
        
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        dataflow_cfg = backend._create_dataflow_config(mock_config)
        
        # Should use default clock period
        assert dataflow_cfg["synth_clk_period_ns"] == 10.0
    
    @patch('brainsmith.core_v3.phase3.legacy_finn_backend.Path.mkdir')
    def test_run_method_success(self, mock_mkdir, temp_dir, mock_config):
        """Test successful run execution."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Mock FINN build execution
        with patch.object(backend, '_execute_finn_build') as mock_build:
            mock_build.return_value = 0  # Success exit code
            
            # Mock metrics extraction
            metrics_collector = MetricsCollector()
            with patch.object(metrics_collector, 'collect_from_finn_output') as mock_metrics:
                mock_metrics.return_value = BuildMetrics(
                    throughput=1000.0,
                    lut_utilization=0.75
                )
                
                with patch('brainsmith.core_v3.phase3.legacy_finn_backend.MetricsCollector') as mock_mc_class:
                    mock_mc_class.return_value = metrics_collector
                    
                    # Mock artifact collection
                    with patch.object(backend, '_collect_artifacts') as mock_artifacts:
                        mock_artifacts.return_value = {
                            "performance_report": "/path/to/perf.json"
                        }
                        
                        # Run the backend with preprocessed model path
                        result = backend.run(mock_config, "/path/to/preprocessed_model.onnx")
        
        # Verify result
        assert result.config_id == "test_config_001"
        assert result.status == BuildStatus.SUCCESS
        assert result.metrics.throughput == 1000.0
        assert result.metrics.lut_utilization == 0.75
        assert "performance_report" in result.artifacts
        assert result.error_message is None
        
        # Verify FINN build was called with correct model path
        mock_build.assert_called_once_with("/path/to/preprocessed_model.onnx", mock.ANY)
    
    @patch('brainsmith.core_v3.phase3.legacy_finn_backend.Path.mkdir')
    def test_run_method_failure(self, mock_mkdir, temp_dir, mock_config):
        """Test run execution with build failure."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Mock FINN build failure
        with patch.object(backend, '_execute_finn_build') as mock_build:
            mock_build.side_effect = Exception("Build failed: synthesis error")
            
            # Run the backend with preprocessed model path
            result = backend.run(mock_config, "/path/to/preprocessed_model.onnx")
        
        # Verify failure result
        assert result.config_id == "test_config_001"
        assert result.status == BuildStatus.FAILED
        assert result.metrics is None
        assert "Build failed: synthesis error" in result.error_message
        assert result.duration_seconds >= 0
    
    def test_execute_finn_build(self, temp_dir):
        """Test _execute_finn_build method."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Create mock dataflow config
        output_dir = str(Path(temp_dir) / "output")
        os.makedirs(output_dir, exist_ok=True)
        
        finn_config = {
            "output_dir": output_dir,
            "synth_clk_period_ns": 5.0,
            "steps": ["step_qonnx_to_finn", "step_streamline"]
        }
        
        # Execute build
        exit_code = backend._execute_finn_build(
            model_path="/path/to/model.onnx",
            finn_config=finn_config
        )
        
        assert exit_code == 0
        
        # Check that mock files were created
        assert Path(output_dir, "estimate_layer_resources_hls.json").exists()
        assert Path(output_dir, "rtlsim_performance.json").exists()
    
    def test_collect_artifacts(self, temp_dir):
        """Test _collect_artifacts method."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Create test files matching the actual implementation
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True)
        
        # Create files that match artifact_patterns
        (output_dir / "estimate_layer_resources_hls.json").write_text("{}")
        (output_dir / "rtlsim_performance.json").write_text("{}")
        (output_dir / "build_dataflow.log").write_text("build log")
        (output_dir / "time_per_step.json").write_text("{}")
        
        # Create stitched IP directory
        (output_dir / "stitched_ip").mkdir()
        (output_dir / "stitched_ip" / "finn_design.xpr").write_text("project file")
        
        # Collect artifacts
        artifacts = backend._collect_artifacts(str(output_dir))
        
        # Verify collected artifacts match implementation
        assert "estimate_reports" in artifacts
        assert artifacts["estimate_reports"].endswith("estimate_layer_resources_hls.json")
        assert "performance_data" in artifacts
        assert artifacts["performance_data"].endswith("rtlsim_performance.json")
        assert "build_log" in artifacts
        assert artifacts["build_log"].endswith("build_dataflow.log")
        assert "timing_summary" in artifacts
        assert artifacts["timing_summary"].endswith("time_per_step.json")
        assert "stitched_ip" in artifacts
        assert artifacts["stitched_ip"].endswith("finn_design.xpr")
    
    def test_collect_artifacts_missing_files(self, temp_dir):
        """Test _collect_artifacts with missing files."""
        backend = LegacyFINNBackend(finn_build_dir=temp_dir)
        
        # Create empty output directory
        output_dir = Path(temp_dir) / "output"
        output_dir.mkdir(parents=True)
        
        # Collect artifacts (should handle missing files gracefully)
        artifacts = backend._collect_artifacts(output_dir)
        
        # Should return empty dict or only existing artifacts
        assert isinstance(artifacts, dict)
        assert len(artifacts) == 0  # No artifacts found