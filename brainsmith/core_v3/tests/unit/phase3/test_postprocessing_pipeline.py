"""
Unit tests for PostprocessingPipeline.
"""

import pytest
import tempfile
import shutil
import os
import json
from pathlib import Path

from brainsmith.core_v3.phase1.data_structures import ProcessingStep, GlobalConfig, OutputStage
from brainsmith.core_v3.phase2.data_structures import BuildConfig
from brainsmith.core_v3.phase3.postprocessing import PostprocessingPipeline
from brainsmith.core_v3.phase3.data_structures import BuildResult, BuildStatus, BuildMetrics
from datetime import datetime


class TestPostprocessingPipeline:
    """Test PostprocessingPipeline class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock BuildConfig with postprocessing steps."""
        return BuildConfig(
            id="test_config_001",
            design_space_id="test_ds_001",
            kernels=[("TestKernel", ["param1"])],
            transforms={"default": ["TestTransform"]},
            preprocessing=[],
            postprocessing=[
                ProcessingStep(name="qonnx_analysis_1", type="postprocessing", enabled=True),
                ProcessingStep(name="qonnx_analysis_2", type="postprocessing", enabled=True, 
                             parameters={"analysis_param": "value1"}),
                ProcessingStep(name="qonnx_analysis_3", type="postprocessing", enabled=False),
            ],
            build_steps=["step1"],
            config_flags={},
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory=temp_dir,
                cache_results=True,
                save_artifacts=True,
                log_level="INFO",
            ),
            timestamp=datetime.now(),
            combination_index=1,
            total_combinations=10,
            output_dir=temp_dir,
        )
    
    @pytest.fixture
    def mock_result(self):
        """Create a mock BuildResult."""
        result = BuildResult(config_id="test_config_001")
        result.metrics = BuildMetrics(
            throughput=1000.0,
            latency=10.0,
            lut_utilization=0.5,
            dsp_utilization=0.3
        )
        result.artifacts = {}
        result.complete(BuildStatus.SUCCESS)
        return result
    
    def test_pipeline_creation(self):
        """Test creating PostprocessingPipeline."""
        pipeline = PostprocessingPipeline()
        assert pipeline is not None
    
    def test_analyze_no_steps(self, temp_dir, mock_config, mock_result):
        """Test analyze with no postprocessing steps."""
        # Remove all postprocessing steps
        mock_config.postprocessing = []
        
        pipeline = PostprocessingPipeline()
        
        # Should not raise any errors
        pipeline.analyze(mock_config, mock_result)
        
        # Should create postprocessing directory
        postprocess_dir = os.path.join(temp_dir, "postprocessing")
        assert os.path.exists(postprocess_dir)
    
    def test_analyze_with_enabled_steps(self, temp_dir, mock_config, mock_result):
        """Test analyze with enabled postprocessing steps."""
        pipeline = PostprocessingPipeline()
        
        pipeline.analyze(mock_config, mock_result)
        
        # Should create postprocessing directory
        postprocess_dir = os.path.join(temp_dir, "postprocessing")
        assert os.path.exists(postprocess_dir)
        
        # Should create analysis files for enabled steps only
        analysis1_file = os.path.join(postprocess_dir, "qonnx_analysis_1_analysis.json")
        analysis2_file = os.path.join(postprocess_dir, "qonnx_analysis_2_analysis.json")
        analysis3_file = os.path.join(postprocess_dir, "qonnx_analysis_3_analysis.json")
        
        assert os.path.exists(analysis1_file)
        assert os.path.exists(analysis2_file)
        assert not os.path.exists(analysis3_file)  # Step 3 is disabled
        
        # Should add analysis files to artifacts
        assert "qonnx_analysis_1_analysis" in mock_result.artifacts
        assert "qonnx_analysis_2_analysis" in mock_result.artifacts
        assert "qonnx_analysis_3_analysis" not in mock_result.artifacts
        
        # Verify analysis file content
        with open(analysis1_file, 'r') as f:
            analysis1_data = json.load(f)
        
        assert analysis1_data["step_name"] == "qonnx_analysis_1"
        assert analysis1_data["config_id"] == "test_config_001"
        assert analysis1_data["placeholder"] is True
        assert analysis1_data["parameters"] == {}
        
        # Verify analysis file with parameters
        with open(analysis2_file, 'r') as f:
            analysis2_data = json.load(f)
        
        assert analysis2_data["step_name"] == "qonnx_analysis_2"
        assert analysis2_data["parameters"] == {"analysis_param": "value1"}
    
    def test_apply_qonnx_analysis(self, temp_dir, mock_config, mock_result):
        """Test _apply_qonnx_analysis method."""
        pipeline = PostprocessingPipeline()
        
        # Create a step
        step = ProcessingStep(
            name="test_analysis",
            type="postprocessing",
            enabled=True,
            parameters={"test_param": "test_value"}
        )
        
        pipeline._apply_qonnx_analysis(step, mock_config, mock_result, temp_dir)
        
        # Should create analysis file
        analysis_file = os.path.join(temp_dir, "test_analysis_analysis.json")
        assert os.path.exists(analysis_file)
        
        # Should add to artifacts
        assert "test_analysis_analysis" in mock_result.artifacts
        assert mock_result.artifacts["test_analysis_analysis"] == analysis_file
        
        # Verify file content
        with open(analysis_file, 'r') as f:
            analysis_data = json.load(f)
        
        assert analysis_data["step_name"] == "test_analysis"
        assert analysis_data["config_id"] == "test_config_001"
        assert analysis_data["placeholder"] is True
        assert analysis_data["parameters"] == {"test_param": "test_value"}
        assert "Placeholder analysis for test_analysis" in analysis_data["message"]
    
    def test_analyze_with_disabled_steps(self, temp_dir, mock_config, mock_result):
        """Test that disabled steps are not processed."""
        # Set all steps to disabled
        for step in mock_config.postprocessing:
            step.enabled = False
        
        pipeline = PostprocessingPipeline()
        pipeline.analyze(mock_config, mock_result)
        
        # Should create postprocessing directory
        postprocess_dir = os.path.join(temp_dir, "postprocessing")
        assert os.path.exists(postprocess_dir)
        
        # Should not create any analysis files
        files = os.listdir(postprocess_dir)
        assert len(files) == 0
        
        # Should not add any artifacts
        original_artifacts_count = len(mock_result.artifacts)
        # No new artifacts should be added for disabled steps
        assert len([k for k in mock_result.artifacts.keys() if k.endswith("_analysis")]) == 0