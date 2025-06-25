"""Unit tests for validator with global configuration integration."""

import os
import pytest
from unittest.mock import patch

from brainsmith.core_v3.phase1.validator import DesignSpaceValidator
from brainsmith.core_v3.phase1.data_structures import (
    DesignSpace, HWCompilerSpace, ProcessingSpace, SearchConfig, GlobalConfig,
    SearchStrategy, OutputStage
)
from brainsmith.core_v3.config import BrainsmithConfig, reset_config


class TestValidatorWithGlobalConfig:
    """Test validator integration with global configuration system."""
    
    def setup_method(self):
        """Reset config before each test."""
        reset_config()
        # Store original env vars
        self.orig_max_combinations = os.environ.get("BRAINSMITH_MAX_COMBINATIONS")
        # Clear env vars
        os.environ.pop("BRAINSMITH_MAX_COMBINATIONS", None)
    
    def teardown_method(self):
        """Restore original state after each test."""
        reset_config()
        # Restore original env vars
        if self.orig_max_combinations is not None:
            os.environ["BRAINSMITH_MAX_COMBINATIONS"] = self.orig_max_combinations
        else:
            os.environ.pop("BRAINSMITH_MAX_COMBINATIONS", None)
    
    @pytest.fixture
    def validator(self):
        return DesignSpaceValidator()
    
    @pytest.fixture
    def basic_design_space(self, tmp_path):
        """Create a basic design space for testing."""
        # Create a temporary model file
        model_file = tmp_path / "test_model.onnx"
        model_file.write_text("fake onnx model")
        
        return DesignSpace(
            model_path=str(model_file),
            hw_compiler_space=HWCompilerSpace(
                kernels=["MatMul"],
                transforms=["quantize"],
                build_steps=["ConvertToHW", "PrepareIP"],  # Add required build steps
                config_flags={}
            ),
            processing_space=ProcessingSpace(
                preprocessing=[],
                postprocessing=[]
            ),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[],
                max_evaluations=None,
                timeout_minutes=None,
                parallel_builds=1
            ),
            global_config=GlobalConfig()
        )
    
    def test_combinations_within_default_limit(self, validator, basic_design_space):
        """Test validation passes when combinations are within default limit."""
        # Basic design space has 1 combination (well within 100k default)
        result = validator.validate(basic_design_space)
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_combinations_exceed_global_config_limit(self, validator, basic_design_space):
        """Test validation fails when combinations exceed global config limit."""
        # Set a very low global limit
        with patch('brainsmith.core_v3.phase1.validator.get_config') as mock_get_config:
            mock_get_config.return_value = BrainsmithConfig(
                max_combinations=0,  # Set to 0 to ensure failure
                timeout_minutes=60
            )
            
            result = validator.validate(basic_design_space)
            assert not result.is_valid
            # Check that one of the errors is about exceeding the limit
            error_messages = " ".join(result.errors)
            assert "exceeding maximum of 0" in error_messages
            assert "You can increase this limit" in error_messages
    
    def test_blueprint_override_global_limit(self, validator, basic_design_space):
        """Test blueprint max_combinations overrides global config."""
        # Set global limit very low
        with patch('brainsmith.core_v3.phase1.validator.get_config') as mock_get_config:
            mock_get_config.return_value = BrainsmithConfig(
                max_combinations=0,  # Very low global limit
                timeout_minutes=60
            )
            
            # But blueprint allows more
            basic_design_space.global_config.max_combinations = 1000
            
            result = validator.validate(basic_design_space)
            assert result.is_valid  # Should pass due to blueprint override
            assert len(result.errors) == 0
    
    def test_environment_variable_override(self, validator, basic_design_space):
        """Test environment variable overrides work."""
        # Set environment variable to very low value
        os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "0"
        
        # Reset config to pick up environment change
        reset_config()
        
        result = validator.validate(basic_design_space)
        assert not result.is_valid
        assert len(result.errors) == 1
        assert "exceeding maximum of 0" in result.errors[0]
    
    def test_large_design_space_combinations(self, validator):
        """Test validation with a design space that has many combinations."""
        # Create a design space with multiple options to generate many combinations
        large_design_space = DesignSpace(
            model_path="test_model.onnx",
            hw_compiler_space=HWCompilerSpace(
                kernels=[
                    ["MatMul", ["rtl", "hls"]],  # 2 combinations
                    ["Conv", ["rtl", "hls"]],    # 2 combinations
                ],
                transforms=[
                    ["quantize", "fold"],        # 2 combinations
                    ["stream_v1", "stream_v2"],  # 2 combinations
                ],
                build_steps=["synth"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(
                preprocessing=[],
                postprocessing=[]
            ),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[],
                max_evaluations=None,
                timeout_minutes=None,
                parallel_builds=1
            ),
            global_config=GlobalConfig()
        )
        
        # This should generate: 2 * 2 * 2 * 2 = 16 combinations
        total_combinations = large_design_space.get_total_combinations()
        assert total_combinations == 16
        
        # Set global limit to less than total
        with patch('brainsmith.core_v3.phase1.validator.get_config') as mock_get_config:
            mock_get_config.return_value = BrainsmithConfig(
                max_combinations=10,  # Less than 16
                timeout_minutes=60
            )
            
            result = validator.validate(large_design_space)
            assert not result.is_valid
            assert len(result.errors) == 1
            assert "16 combinations" in result.errors[0]
            assert "exceeding maximum of 10" in result.errors[0]
    
    def test_priority_order_blueprint_over_environment(self, validator, basic_design_space):
        """Test that blueprint config takes priority over environment variables."""
        # Set environment to very restrictive
        os.environ["BRAINSMITH_MAX_COMBINATIONS"] = "0"
        
        # Blueprint allows more
        basic_design_space.global_config.max_combinations = 1000
        
        # Reset config to pick up environment change
        reset_config()
        
        result = validator.validate(basic_design_space)
        # Should pass because blueprint override takes precedence
        assert result.is_valid
        assert len(result.errors) == 0
    
    def test_zero_combinations_error(self, validator, tmp_path):
        """Test that zero combinations is always an error regardless of limits."""
        # Create a temporary model file
        model_file = tmp_path / "test_model.onnx"
        model_file.write_text("fake onnx model")
        
        # Create a design space with no valid combinations
        empty_design_space = DesignSpace(
            model_path=str(model_file),
            hw_compiler_space=HWCompilerSpace(
                kernels=[],  # No kernels
                transforms=[],
                build_steps=["ConvertToHW", "PrepareIP"],
                config_flags={}
            ),
            processing_space=ProcessingSpace(
                preprocessing=[],
                postprocessing=[]
            ),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[],
                max_evaluations=None,
                timeout_minutes=None,
                parallel_builds=1
            ),
            global_config=GlobalConfig()
        )
        
        result = validator.validate(empty_design_space)
        assert not result.is_valid
        assert len(result.errors) >= 1
        assert any("no valid combinations" in error for error in result.errors)
    
    def test_helpful_error_message(self, validator, basic_design_space):
        """Test that error message provides helpful guidance."""
        with patch('brainsmith.core_v3.phase1.validator.get_config') as mock_get_config:
            mock_get_config.return_value = BrainsmithConfig(
                max_combinations=0,
                timeout_minutes=60
            )
            
            result = validator.validate(basic_design_space)
            assert not result.is_valid
            error_messages = " ".join(result.errors)
            
            # Check that error message contains helpful guidance
            assert "You can increase this limit" in error_messages
            assert "max_combinations in the blueprint" in error_messages
            assert "~/.brainsmith/config.yaml" in error_messages
            assert "BRAINSMITH_MAX_COMBINATIONS" in error_messages