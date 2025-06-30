"""
Unit tests for backend step configuration functionality.
"""

import pytest
import tempfile
import os
import json
from datetime import datetime

from brainsmith.core.phase1.data_structures import GlobalConfig, OutputStage
from brainsmith.core.phase2.data_structures import BuildConfig
from brainsmith.core.phase3.legacy_finn_backend import LegacyFINNBackend
from brainsmith.core.phase3.future_brainsmith_backend import FutureBrainsmithBackend
from brainsmith.core.phase3.step_resolver import StepResolver


class TestBackendStepConfiguration:
    """Test backend step configuration functionality."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup handled by temp_dir
    
    @pytest.fixture
    def base_global_config(self):
        """Create a base global config for testing."""
        return GlobalConfig(
            output_stage=OutputStage.RTL,
            working_directory="/tmp/test",
            cache_results=True,
            save_artifacts=True,
            log_level="INFO"
        )
    
    @pytest.fixture
    def base_build_config(self, base_global_config, temp_dir):
        """Create a base build config for testing."""
        return BuildConfig(
            id="test_config_001",
            design_space_id="test_ds_001",
            kernels=[("MatMul", ["hls"])],
            transforms={"quantization": ["quantize"]},
            preprocessing=[],
            postprocessing=[],
            build_steps=["step_create_dataflow_partition", "step_specialize_layers", "step_hw_codegen"],
            config_flags={"target_device": "xczu7ev"},
            global_config=base_global_config,
            output_dir=temp_dir,
            timestamp=datetime.now(),
            combination_index=0,
            total_combinations=1
        )
    
    def test_legacy_backend_step_resolution_default(self, base_build_config):
        """Test legacy backend with default step configuration."""
        backend = LegacyFINNBackend()
        
        # Should return original build steps when no filtering
        resolved_steps = backend._resolve_build_steps(base_build_config)
        assert resolved_steps == base_build_config.build_steps
    
    def test_legacy_backend_step_resolution_with_start_stop(self, base_build_config):
        """Test legacy backend with start/stop step configuration."""
        # Configure step filtering
        base_build_config.global_config.start_step = "step_specialize_layers"
        base_build_config.global_config.stop_step = "step_hw_codegen"
        
        backend = LegacyFINNBackend()
        resolved_steps = backend._resolve_build_steps(base_build_config)
        
        expected = ["step_specialize_layers", "step_hw_codegen"]
        assert resolved_steps == expected
    
    def test_legacy_backend_step_resolution_with_indices(self, base_build_config):
        """Test legacy backend with step indices."""
        # Configure step filtering with indices
        base_build_config.global_config.start_step = 1
        base_build_config.global_config.stop_step = 2
        
        backend = LegacyFINNBackend()
        resolved_steps = backend._resolve_build_steps(base_build_config)
        
        expected = ["step_specialize_layers", "step_hw_codegen"]
        assert resolved_steps == expected
    
    def test_legacy_backend_step_resolution_with_semantic_types(self, base_build_config):
        """Test legacy backend with semantic input/output types."""
        # Configure semantic types
        base_build_config.global_config.input_type = "hwgraph"
        base_build_config.global_config.output_type = "rtl"
        
        backend = LegacyFINNBackend()
        resolved_steps = backend._resolve_build_steps(base_build_config)
        
        # The backend uses the build_steps from config when available
        # Since base_build_config has custom build_steps, it filters those, not standard steps
        assert resolved_steps == base_build_config.build_steps
        
        # Test with no build_steps to use standard steps
        base_build_config.build_steps = None
        resolved_steps = backend._resolve_build_steps(base_build_config)
        
        # Now it should use standard steps and filter based on semantic types
        resolver = StepResolver()
        standard_steps = resolver.get_standard_steps()
        
        # Find expected range in standard steps
        start_idx = standard_steps.index("step_create_dataflow_partition")
        stop_idx = standard_steps.index("step_hw_codegen")
        expected = standard_steps[start_idx:stop_idx + 1]
        
        assert resolved_steps == expected
    
    def test_legacy_backend_step_resolution_error_handling(self, base_build_config):
        """Test legacy backend error handling for invalid step configuration."""
        # Configure invalid step name
        base_build_config.global_config.start_step = "invalid_step_name"
        
        backend = LegacyFINNBackend()
        resolved_steps = backend._resolve_build_steps(base_build_config)
        
        # Should fall back to original build steps
        assert resolved_steps == base_build_config.build_steps
    
    def test_legacy_backend_dataflow_config_includes_resolved_steps(self, base_build_config):
        """Test that dataflow config includes resolved steps."""
        # Configure step filtering
        base_build_config.global_config.start_step = 1
        base_build_config.global_config.stop_step = 2
        
        backend = LegacyFINNBackend()
        finn_config = backend._create_dataflow_config(base_build_config)
        
        expected_steps = ["step_specialize_layers", "step_hw_codegen"]
        assert finn_config["steps"] == expected_steps
    
    def test_future_backend_step_configuration_preparation(self, base_build_config):
        """Test future backend step configuration preparation."""
        backend = FutureBrainsmithBackend()
        step_config = backend._prepare_step_configuration(base_build_config)
        
        # Should include all required fields
        assert "original_build_steps" in step_config
        assert "output_stage" in step_config
        assert "start_step" in step_config
        assert "stop_step" in step_config
        assert "input_type" in step_config
        assert "output_type" in step_config
        assert "resolved_start_step" in step_config
        assert "resolved_stop_step" in step_config
        assert "resolved_steps" in step_config
        assert "total_steps" in step_config
        assert "filtered_steps_count" in step_config
        assert "step_filtering_applied" in step_config
        assert "standard_steps" in step_config
        assert "supported_input_types" in step_config
        assert "supported_output_types" in step_config
        
        # No filtering applied by default
        assert step_config["step_filtering_applied"] is False
        assert step_config["resolved_steps"] == base_build_config.build_steps
    
    def test_future_backend_step_configuration_with_filtering(self, base_build_config):
        """Test future backend step configuration with filtering."""
        # Configure step filtering
        base_build_config.global_config.start_step = "step_specialize_layers"
        base_build_config.global_config.stop_step = "step_hw_codegen"
        
        backend = FutureBrainsmithBackend()
        step_config = backend._prepare_step_configuration(base_build_config)
        
        # Should show filtering applied
        assert step_config["step_filtering_applied"] is True
        assert step_config["resolved_start_step"] == "step_specialize_layers"
        assert step_config["resolved_stop_step"] == "step_hw_codegen"
        assert step_config["resolved_steps"] == ["step_specialize_layers", "step_hw_codegen"]
        assert step_config["filtered_steps_count"] == 2
        assert step_config["total_steps"] == 3
    
    def test_future_backend_step_configuration_with_semantic_types(self, base_build_config):
        """Test future backend step configuration with semantic types."""
        # Configure semantic types
        base_build_config.global_config.input_type = "hwgraph"
        base_build_config.global_config.output_type = "rtl"
        
        backend = FutureBrainsmithBackend()
        step_config = backend._prepare_step_configuration(base_build_config)
        
        # Should resolve semantic types
        assert step_config["input_type"] == "hwgraph"
        assert step_config["output_type"] == "rtl"
        assert step_config["resolved_start_step"] == "step_create_dataflow_partition"
        assert step_config["resolved_stop_step"] == "step_hw_codegen"
        assert step_config["step_filtering_applied"] is True
    
    def test_future_backend_step_configuration_error_handling(self, base_build_config):
        """Test future backend error handling for step configuration."""
        # Configure invalid step
        base_build_config.global_config.start_step = "invalid_step"
        
        backend = FutureBrainsmithBackend()
        step_config = backend._prepare_step_configuration(base_build_config)
        
        # Should include error information
        assert "error" in step_config
        assert "fallback_steps" in step_config
        assert step_config["step_filtering_applied"] is False
        assert step_config["fallback_steps"] == base_build_config.build_steps
    
    def test_future_backend_full_config_includes_step_configuration(self, base_build_config):
        """Test that full config includes step configuration."""
        # Configure step filtering
        base_build_config.global_config.input_type = "hwgraph"
        base_build_config.global_config.output_type = "rtl"
        
        backend = FutureBrainsmithBackend()
        full_config = backend._prepare_finn_brainsmith_config(base_build_config)
        
        # Should include step configuration section
        assert "step_configuration" in full_config
        step_config = full_config["step_configuration"]
        
        assert step_config["input_type"] == "hwgraph"
        assert step_config["output_type"] == "rtl"
        assert step_config["step_filtering_applied"] is True
    
    def test_future_backend_config_serialization(self, base_build_config, temp_dir):
        """Test that future backend config can be serialized to JSON."""
        # Configure step filtering
        base_build_config.global_config.start_step = 0
        base_build_config.global_config.stop_step = 1
        
        backend = FutureBrainsmithBackend()
        
        # Run build to generate config file
        result = backend.run(base_build_config, "/path/to/model.onnx")
        
        # Check that config file was created and contains step configuration
        config_file = os.path.join(temp_dir, "finn_brainsmith_config.json")
        assert os.path.exists(config_file)
        
        with open(config_file, 'r') as f:
            saved_config = json.load(f)
        
        assert "step_configuration" in saved_config
        step_config = saved_config["step_configuration"]
        assert "resolved_steps" in step_config
        assert "step_filtering_applied" in step_config
        assert step_config["step_filtering_applied"] is True
    
    def test_build_config_serialization_includes_step_fields(self, base_build_config):
        """Test that BuildConfig serialization includes step fields."""
        # Configure step fields
        base_build_config.global_config.start_step = "test_step"
        base_build_config.global_config.stop_step = 5
        base_build_config.global_config.input_type = "hwgraph"
        base_build_config.global_config.output_type = "rtl"
        
        serialized = base_build_config.to_dict()
        
        global_config = serialized["global_config"]
        assert global_config["start_step"] == "test_step"
        assert global_config["stop_step"] == 5
        assert global_config["input_type"] == "hwgraph"
        assert global_config["output_type"] == "rtl"