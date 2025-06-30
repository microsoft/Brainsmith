"""
Unit tests for PreprocessingPipeline.
"""

import pytest
import tempfile
import shutil
import os
from pathlib import Path

from brainsmith.core.phase1.data_structures import ProcessingStep, GlobalConfig, OutputStage
from brainsmith.core.phase2.data_structures import BuildConfig
from brainsmith.core.phase3.preprocessing import PreprocessingPipeline
from datetime import datetime


class TestPreprocessingPipeline:
    """Test PreprocessingPipeline class."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for tests."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def mock_config(self, temp_dir):
        """Create a mock BuildConfig with preprocessing steps."""
        return BuildConfig(
            id="test_config_001",
            design_space_id="test_ds_001",
            kernels=[("TestKernel", ["param1"])],
            transforms={"default": ["TestTransform"]},
            preprocessing=[
                ProcessingStep(name="qonnx_transform_1", type="preprocessing", enabled=True),
                ProcessingStep(name="qonnx_transform_2", type="preprocessing", enabled=True, 
                             parameters={"param1": "value1"}),
                ProcessingStep(name="qonnx_transform_3", type="preprocessing", enabled=False),
            ],
            postprocessing=[],
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
    
    def test_pipeline_creation(self):
        """Test creating PreprocessingPipeline."""
        pipeline = PreprocessingPipeline()
        assert pipeline is not None
    
    def test_execute_no_steps(self, temp_dir, mock_config):
        """Test execute with no preprocessing steps."""
        # Remove all preprocessing steps
        mock_config.preprocessing = []
        
        pipeline = PreprocessingPipeline()
        model_path = "/path/to/input_model.onnx"
        
        result_path = pipeline.execute(mock_config, model_path)
        
        # Should return processed_model.onnx path
        expected_path = os.path.join(temp_dir, "processed_model.onnx")
        assert result_path == expected_path
    
    def test_execute_with_enabled_steps(self, temp_dir, mock_config):
        """Test execute with enabled preprocessing steps."""
        pipeline = PreprocessingPipeline()
        
        # Create a dummy input model file
        input_model = os.path.join(temp_dir, "input_model.onnx")
        with open(input_model, 'w') as f:
            f.write("dummy model content")
        
        result_path = pipeline.execute(mock_config, input_model)
        
        # Should return processed_model.onnx path
        expected_path = os.path.join(temp_dir, "processed_model.onnx")
        assert result_path == expected_path
        assert os.path.exists(result_path)
        
        # Should create preprocessing directory
        preprocess_dir = os.path.join(temp_dir, "preprocessing")
        assert os.path.exists(preprocess_dir)
        
        # Should create step output files for enabled steps only
        step1_output = os.path.join(preprocess_dir, "qonnx_transform_1_output.onnx")
        step2_output = os.path.join(preprocess_dir, "qonnx_transform_2_output.onnx")
        step3_output = os.path.join(preprocess_dir, "qonnx_transform_3_output.onnx")
        
        assert os.path.exists(step1_output)
        assert os.path.exists(step2_output)
        assert not os.path.exists(step3_output)  # Step 3 is disabled
    
    def test_apply_qonnx_transform(self, temp_dir):
        """Test _apply_qonnx_transform method."""
        pipeline = PreprocessingPipeline()
        
        # Create a dummy input model file
        input_model = os.path.join(temp_dir, "input.onnx")
        with open(input_model, 'w') as f:
            f.write("dummy model content")
        
        # Create a step
        step = ProcessingStep(
            name="test_transform",
            type="preprocessing",
            enabled=True,
            parameters={"test_param": "test_value"}
        )
        
        result_path = pipeline._apply_qonnx_transform(step, input_model, temp_dir)
        
        # Should create output file
        expected_path = os.path.join(temp_dir, "test_transform_output.onnx")
        assert result_path == expected_path
        assert os.path.exists(result_path)
        
        # Should copy content
        with open(result_path, 'r') as f:
            content = f.read()
        assert content == "dummy model content"
    
    def test_execute_with_model_path_fallback(self, temp_dir):
        """Test execute with model_path fallback."""
        config = BuildConfig(
            id="test_config",
            design_space_id="test_ds",
            kernels=[],
            transforms={},
            preprocessing=[],
            postprocessing=[],
            build_steps=[],
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
            total_combinations=1,
            output_dir=temp_dir,
        )
        
        pipeline = PreprocessingPipeline()
        
        # Call without model_path - should use fallback
        result_path = pipeline.execute(config, None)
        
        # Should still return processed_model.onnx path
        expected_path = os.path.join(temp_dir, "processed_model.onnx")
        assert result_path == expected_path