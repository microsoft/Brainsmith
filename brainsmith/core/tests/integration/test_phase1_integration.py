"""
Integration tests for Phase 1: Design Space Constructor.
"""

import os
import pytest
from pathlib import Path

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.phase1.exceptions import (
    ConfigurationError,
    BlueprintParseError,
    ValidationError,
)

# No fake plugins - use real QONNX/FINN plugins only


class TestForgeIntegration:
    """Integration tests for the Forge API."""
    
    @pytest.fixture
    def fixtures_dir(self):
        """Get path to fixtures directory."""
        return Path(__file__).parent.parent / "fixtures"
    
    @pytest.fixture
    def model_path(self, fixtures_dir):
        """Get path to test ONNX model."""
        return str(fixtures_dir / "simple_model.onnx")
    
    @pytest.fixture
    def simple_blueprint(self, fixtures_dir):
        """Get path to simple blueprint."""
        return str(fixtures_dir / "simple_blueprint.yaml")
    
    @pytest.fixture
    def complex_blueprint(self, fixtures_dir):
        """Get path to complex blueprint."""
        return str(fixtures_dir / "complex_blueprint.yaml")
    
    def test_forge_simple_blueprint(self, model_path, simple_blueprint):
        """Test forging a simple design space."""
        design_space = forge(model_path, simple_blueprint)
        
        # Verify basic structure
        assert design_space.model_path.endswith("simple_model.onnx")
        assert design_space.get_total_combinations() > 0
        
        # Check kernels
        assert len(design_space.hw_compiler_space.kernels) == 2
        # First kernel is auto-discovered, so it's a tuple (name, backends)
        assert design_space.hw_compiler_space.kernels[0][0] == "LayerNorm"
        assert isinstance(design_space.hw_compiler_space.kernels[0][1], list)
        # Second kernel has explicit backends
        assert isinstance(design_space.hw_compiler_space.kernels[1], tuple)
        assert design_space.hw_compiler_space.kernels[1][0] == "Crop"
        
        # Check transforms
        assert len(design_space.hw_compiler_space.transforms) == 2
        
        # Check search config
        assert design_space.search_config.strategy.value == "exhaustive"
        assert len(design_space.search_config.constraints) == 1
        
        # Check global config
        assert design_space.global_config.output_stage.value == "rtl"
        assert design_space.global_config.working_directory == "./test_builds"
    
    def test_forge_complex_blueprint(self, model_path, complex_blueprint):
        """Test forging a complex design space with all features."""
        design_space = forge(model_path, complex_blueprint, verbose=True)
        
        # Check complex kernel configurations
        kernels = design_space.hw_compiler_space.kernels
        assert len(kernels) == 5
        
        # Check phase-based transforms
        transforms = design_space.hw_compiler_space.transforms
        assert isinstance(transforms, dict)
        assert "cleanup" in transforms
        assert "topology_opt" in transforms
        
        # Check preprocessing/postprocessing
        assert len(design_space.processing_space.preprocessing) == 2
        assert len(design_space.processing_space.postprocessing) == 2
        
        # Check multiple constraints
        assert len(design_space.search_config.constraints) == 3
        
        # Check advanced search config
        assert design_space.search_config.max_evaluations == 100
        assert design_space.search_config.timeout_minutes == 720
        assert design_space.search_config.parallel_builds == 4
    
    def test_forge_missing_model(self, simple_blueprint):
        """Test error when model file doesn't exist."""
        with pytest.raises(ConfigurationError) as exc:
            forge("nonexistent_model.onnx", simple_blueprint)
        assert "Model file not found" in str(exc.value)
    
    def test_forge_missing_blueprint(self, model_path):
        """Test error when blueprint file doesn't exist."""
        with pytest.raises(BlueprintParseError) as exc:
            forge(model_path, "nonexistent_blueprint.yaml")
        assert "Blueprint file not found" in str(exc.value)
    
    def test_forge_invalid_blueprint(self, model_path, tmp_path):
        """Test error with invalid blueprint."""
        # Create invalid blueprint
        invalid_blueprint = tmp_path / "invalid.yaml"
        invalid_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels: "should_be_list"
""")
        
        with pytest.raises(BlueprintParseError) as exc:
            forge(model_path, str(invalid_blueprint))
        assert "kernels must be a list" in str(exc.value)
    
    def test_forge_validation_error(self, model_path, tmp_path):
        """Test validation errors."""
        # Create blueprint that will fail validation
        bad_blueprint = tmp_path / "bad.yaml"
        bad_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels: []
  transforms: []
  build_steps: []  # Empty build steps should error
search:
  strategy: "exhaustive"
  parallel_builds: 0  # Invalid value
global:
  output_stage: "rtl"
  working_directory: ""  # Empty directory should error
""")
        
        with pytest.raises(ValidationError) as exc:
            forge(model_path, str(bad_blueprint))
        
        error_msg = str(exc.value)
        assert "Build steps are required" in error_msg
        assert "Parallel builds must be at least 1" in error_msg
        assert "Working directory is required" in error_msg
    
    def test_forge_with_warnings(self, model_path, tmp_path, caplog):
        """Test that warnings are properly displayed."""
        import logging
        caplog.set_level(logging.WARNING)
        
        # Create blueprint that will generate warnings
        warning_blueprint = tmp_path / "warnings.yaml"
        warning_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - MatMul
  transforms:
    - QuantizeGraph
  build_steps:
    - SomeCustomStep  # Missing common steps will warn
search:
  strategy: "exhaustive"
  parallel_builds: 64  # High value will warn
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        design_space = forge(model_path, str(warning_blueprint))
        
        # Check that warnings were logged
        assert any("warning" in record.message.lower() for record in caplog.records)
    
    def test_forge_api_verbose_mode(self, model_path, simple_blueprint, caplog):
        """Test verbose mode logging."""
        import logging
        caplog.set_level(logging.DEBUG)
        
        api = ForgeAPI(verbose=True)
        design_space = api.forge(model_path, simple_blueprint)
        
        # Check for INFO or DEBUG level logging
        assert any(record.levelname in ["INFO", "DEBUG"] for record in caplog.records)
    
    def test_total_combinations_calculation(self, model_path, tmp_path):
        """Test accurate combination counting."""
        # Create blueprint with known combination count
        count_blueprint = tmp_path / "count.yaml"
        count_blueprint.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - LayerNorm
    - ["Crop", "Shuffle"]  # 2 options (real QONNX kernels)
  transforms:
    - FoldConstants
    - ["InferShapes", "InferDataTypes", "RemoveUnusedTensors"]  # 3 options (real QONNX transforms)
  build_steps:
    - ConvertToHW
processing:
  preprocessing:
    - name: "norm"
      options:
        - {enabled: true}
        - {enabled: false}  # 2 options
search:
  strategy: "exhaustive"
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        design_space = forge(model_path, str(count_blueprint))
        
        # Total should be: 1 * 2 (kernels) * 1 * 3 (transforms) * 2 (preprocessing) = 12
        assert design_space.get_total_combinations() == 12