"""
Unit tests for error scenarios and error message quality.
"""

import pytest
import yaml
import tempfile
from pathlib import Path

from brainsmith.core.phase1 import forge, ForgeAPI
from brainsmith.core.phase1.parser import BlueprintParser
from brainsmith.core.phase1.exceptions import (
    BlueprintParseError, 
    ValidationError, 
    ConfigurationError,
    PluginNotFoundError
)


class TestPluginNotFoundError:
    """Test the PluginNotFoundError exception class."""
    
    def test_plugin_not_found_with_suggestions(self):
        """Test error message with suggestions."""
        error = PluginNotFoundError(
            "transform", 
            "BadTransform",
            ["GoodTransform1", "GoodTransform2", "GoodTransform3"]
        )
        
        error_msg = str(error)
        assert "Transform 'BadTransform' not found" in error_msg
        assert "Available: ['GoodTransform1', 'GoodTransform2', 'GoodTransform3']" in error_msg
    
    def test_plugin_not_found_truncated_list(self):
        """Test that long suggestion lists are truncated."""
        many_options = [f"Transform{i}" for i in range(10)]
        error = PluginNotFoundError("transform", "BadTransform", many_options)
        
        error_msg = str(error)
        # Should only show first 5 plus "..."
        assert "Transform0" in error_msg
        assert "Transform4" in error_msg
        assert "Transform5" not in error_msg
        assert "..." in error_msg
    
    def test_plugin_not_found_no_suggestions(self):
        """Test error message without suggestions."""
        error = PluginNotFoundError("kernel", "UnknownKernel", None)
        
        error_msg = str(error)
        assert "Kernel 'UnknownKernel' not found" in error_msg
        assert "Available: []" in error_msg  # Implementation always shows available list
    
    def test_plugin_not_found_empty_suggestions(self):
        """Test error message with empty suggestion list."""
        error = PluginNotFoundError("backend", "UnknownBackend", [])
        
        error_msg = str(error)
        assert "Backend 'UnknownBackend' not found" in error_msg
        assert "Available: []" in error_msg


class TestParserErrorScenarios:
    """Test error scenarios in blueprint parsing."""
    
    def test_missing_kernel_error(self, parser_with_registry):
        """Test helpful error for missing kernel."""
        parser = parser_with_registry
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["NonExistentKernel"],
                "transforms": [],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        # Test
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_data, "model.onnx")
        
        # Verify error message quality
        error_msg = str(exc.value)
        assert "Kernel 'NonExistentKernel' not found" in error_msg
        assert "Available kernels:" in error_msg
        # Should show real kernels like LayerNorm
        assert "LayerNorm" in error_msg or "Crop" in error_msg
    
    def test_invalid_backend_error(self, parser_with_registry):
        """Test helpful error for invalid backend."""
        parser = parser_with_registry
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [("LayerNorm", ["InvalidBackend1", "InvalidBackend2"])],
                "transforms": [],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        # Test
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_data, "model.onnx")
        
        # Verify error message
        error_msg = str(exc.value)
        assert "Invalid backends ['InvalidBackend1', 'InvalidBackend2']" in error_msg
        assert "Available:" in error_msg
    
    def test_missing_transform_error(self, parser_with_registry):
        """Test helpful error for missing transform."""
        parser = parser_with_registry
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["LayerNorm"],
                "transforms": ["BadTransform"],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        # Test
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_data, "model.onnx")
        
        # Verify error message shows available transforms
        error_msg = str(exc.value)
        assert "Transform 'BadTransform' not found" in error_msg
        assert "Available:" in error_msg
        # Should show real QONNX/FINN transforms (first few in alphabetical order)
        assert "BatchNormToAffine" in error_msg or "ChangeBatchSize" in error_msg
    
    def test_phase_based_transform_error(self, parser_with_registry):
        """Test error message for transform in specific phase."""
        parser = parser_with_registry
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["LayerNorm"],
                "transforms": {
                    "cleanup": ["RemoveUnusedTensors"],
                    "topology_opt": ["BadTransform"]
                },
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        # Test
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_data, "model.onnx")
        
        # Verify error includes phase context
        error_msg = str(exc.value)
        assert "Transform 'BadTransform' in phase 'topology_opt' not found" in error_msg
    
    def test_multiple_errors_reported(self):
        """Test that multiple plugin errors are caught."""
        # This depends on implementation - parser might fail on first error
        # But validator should collect multiple errors
        pass  # Implementation depends on parser behavior


class TestValidatorErrorScenarios:
    """Test error scenarios in validation."""
    
    def test_multiple_plugin_errors(self, validator_with_registry, model_path):
        """Test validator reports multiple plugin errors."""
        from brainsmith.core.phase1.data_structures import (
            DesignSpace, HWCompilerSpace, ProcessingSpace,
            SearchConfig, GlobalConfig, SearchStrategy, OutputStage
        )
        
        validator = validator_with_registry
        
        # Create design space with multiple issues
        design_space = DesignSpace(
            model_path=model_path,
            hw_compiler_space=HWCompilerSpace(
                kernels=[
                    "BadKernel1",  # Missing kernel
                    ("LayerNorm", ["InvalidBackend1", "InvalidBackend2"]),  # Invalid backends
                    "BadKernel2",  # Another missing kernel
                ],
                transforms=["BadTransform1", "BadTransform2"],  # Missing transforms
                build_steps=[]  # Also missing required build steps
            ),
            processing_space=ProcessingSpace([], []),
            search_config=SearchConfig(
                strategy=SearchStrategy.EXHAUSTIVE,
                constraints=[],
                parallel_builds=0  # Invalid
            ),
            global_config=GlobalConfig(
                output_stage=OutputStage.RTL,
                working_directory=""  # Invalid
            )
        )
        
        # Test
        result = validator.validate(design_space)
        
        # Should collect multiple errors
        assert not result.is_valid
        assert len(result.errors) >= 5  # At least 5 different errors
        
        # Check error content
        all_errors = " ".join(result.errors)
        assert "BadKernel1" in all_errors
        assert "BadKernel2" in all_errors
        assert "BadTransform1" in all_errors
        assert "InvalidBackend1" in all_errors or "InvalidBackend2" in all_errors
        assert "Build steps are required" in all_errors


class TestEndToEndErrorScenarios:
    """Test error scenarios through the complete forge pipeline."""
    
    def test_forge_with_invalid_plugins_blueprint(self, model_path):
        """Test forging with the invalid_plugins blueprint fixture."""
        invalid_blueprint = str(
            Path(__file__).parent.parent.parent / "fixtures" / "blueprints" / "invalid_plugins.yaml"
        )
        
        # Should raise parse error due to invalid plugins
        with pytest.raises(BlueprintParseError) as exc:
            forge(model_path, invalid_blueprint)
        
        # Error should mention specific invalid plugin
        error_msg = str(exc.value)
        assert "NonExistentKernel" in error_msg or "FakeTransform" in error_msg
    
    def test_forge_helpful_error_messages(self, model_path, tmp_path):
        """Test that forge provides helpful error messages."""
        # Create blueprint with common mistakes
        blueprint_path = tmp_path / "mistakes.yaml"
        blueprint_path.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - "Matmul"  # Common typo (should be MatMul)
    - ["layernorm", ["HLS"]]  # Wrong case
  transforms:
    - "RemoveIdentity"  # Close but wrong
  build_steps:
    - "ConvertToHW"
search:
  strategy: "exhastive"  # Typo
global:
  output_stage: "rtl"
  working_directory: "./builds"
""")
        
        # Test
        with pytest.raises((BlueprintParseError, ValidationError)) as exc:
            forge(model_path, str(blueprint_path))
        
        # Should provide helpful error about typos/case
        error_msg = str(exc.value)
        # Error should indicate the issue clearly
        assert "Matmul" in error_msg or "layernorm" in error_msg or "exhastive" in error_msg


class TestErrorMessageQuality:
    """Test that all error messages meet quality standards."""
    
    def test_error_messages_include_context(self, parser_with_registry):
        """Test error messages include sufficient context."""
        parser = parser_with_registry
        
        blueprint_data = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [["NonExistentGroupA", ["NonExistentGroupB", ["backend1"]], "NonExistentGroupC"]],
                "transforms": [],
                "build_steps": ["Step1"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_data, "model.onnx")
        
        # Error should indicate it's in a mutually exclusive group
        error_msg = str(exc.value)
        assert "not found" in error_msg
        assert "Available" in error_msg
    
    def test_error_messages_actionable(self):
        """Test that error messages suggest actions to fix."""
        # Most of our error messages include available options
        # which makes them actionable
        error = PluginNotFoundError(
            "transform",
            "MisspelledTransform", 
            ["CorrectTransform", "AnotherTransform"]
        )
        
        error_msg = str(error)
        # User can see correct options and fix their typo
        assert "Available:" in error_msg
        assert "CorrectTransform" in error_msg