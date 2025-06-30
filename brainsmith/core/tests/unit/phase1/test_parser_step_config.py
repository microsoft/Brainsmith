"""
Unit tests for Blueprint parser step configuration fields.
"""

import pytest
from brainsmith.core.phase1.parser import BlueprintParser
from brainsmith.core.phase1.exceptions import BlueprintParseError


class TestBlueprintParserStepConfig:
    """Test Blueprint parser step configuration parsing."""
    
    @pytest.fixture
    def parser(self):
        """Create a BlueprintParser instance for testing."""
        return BlueprintParser()
    
    
    def test_parse_step_configuration_basic(self, parser):
        """Test parsing basic step configuration."""
        blueprint_data = {
            "version": "3.0",
            "name": "Step Config Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "start_step": "step_create_dataflow_partition",
                "stop_step": "step_hw_codegen"
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        global_config = design_space.global_config
        assert global_config.start_step == "step_create_dataflow_partition"
        assert global_config.stop_step == "step_hw_codegen"
        assert global_config.input_type is None
        assert global_config.output_type is None
    
    def test_parse_step_configuration_semantic_types(self, parser):
        """Test parsing semantic input/output types."""
        blueprint_data = {
            "version": "3.0",
            "name": "Semantic Types Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "input_type": "hwgraph",
                "output_type": "rtl"
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        global_config = design_space.global_config
        assert global_config.start_step is None
        assert global_config.stop_step is None
        assert global_config.input_type == "hwgraph"
        assert global_config.output_type == "rtl"
    
    def test_parse_step_configuration_indices(self, parser):
        """Test parsing step indices."""
        blueprint_data = {
            "version": "3.0",
            "name": "Step Indices Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "start_step": 5,
                "stop_step": 10
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        global_config = design_space.global_config
        assert global_config.start_step == 5
        assert global_config.stop_step == 10
        assert global_config.input_type is None
        assert global_config.output_type is None
    
    def test_parse_step_configuration_mixed(self, parser):
        """Test parsing mixed step configuration."""
        blueprint_data = {
            "version": "3.0",
            "name": "Mixed Step Config Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "start_step": "step_create_dataflow_partition",
                "stop_step": 8,
                "input_type": "finn",
                "output_type": "ip"
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        global_config = design_space.global_config
        assert global_config.start_step == "step_create_dataflow_partition"
        assert global_config.stop_step == 8
        assert global_config.input_type == "finn"
        assert global_config.output_type == "ip"
    
    def test_parse_step_configuration_defaults(self, parser):
        """Test that step configuration has proper defaults."""
        blueprint_data = {
            "version": "3.0",
            "name": "Defaults Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl"
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        global_config = design_space.global_config
        assert global_config.start_step is None
        assert global_config.stop_step is None
        assert global_config.input_type is None
        assert global_config.output_type is None
    
    def test_parse_step_configuration_invalid_start_step_type(self, parser):
        """Test parsing invalid start_step type."""
        blueprint_data = {
            "version": "3.0",
            "name": "Invalid Start Step Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "start_step": []
            }
        }
        
        with pytest.raises(BlueprintParseError, match="global.start_step must be a string or integer"):
            parser.parse(blueprint_data, "/path/to/model.onnx")
    
    def test_parse_step_configuration_invalid_stop_step_type(self, parser):
        """Test parsing invalid stop_step type."""
        blueprint_data = {
            "version": "3.0",
            "name": "Invalid Stop Step Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "stop_step": {}
            }
        }
        
        with pytest.raises(BlueprintParseError, match="global.stop_step must be a string or integer"):
            parser.parse(blueprint_data, "/path/to/model.onnx")
    
    def test_parse_step_configuration_invalid_input_type(self, parser):
        """Test parsing invalid input_type."""
        blueprint_data = {
            "version": "3.0",
            "name": "Invalid Input Type Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "input_type": 123
            }
        }
        
        with pytest.raises(BlueprintParseError, match="global.input_type must be a string"):
            parser.parse(blueprint_data, "/path/to/model.onnx")
    
    def test_parse_step_configuration_invalid_output_type(self, parser):
        """Test parsing invalid output_type."""
        blueprint_data = {
            "version": "3.0",
            "name": "Invalid Output Type Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "global": {
                "output_stage": "rtl",
                "output_type": []
            }
        }
        
        with pytest.raises(BlueprintParseError, match="global.output_type must be a string"):
            parser.parse(blueprint_data, "/path/to/model.onnx")
    
    def test_parse_step_configuration_backward_compatibility(self, parser):
        """Test that existing blueprints without step config still work."""
        blueprint_data = {
            "version": "3.0",
            "name": "Backward Compatibility Test",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"]
            },
            "processing": {
                "preprocessing": [
                    {
                        "name": "normalization",
                        "options": [
                            {"enabled": True}
                        ]
                    }
                ]
            },
            "search": {
                "strategy": "exhaustive"
            },
            "global": {
                "output_stage": "rtl",
                "working_directory": "./test_builds"
            }
        }
        
        design_space = parser.parse(blueprint_data, "/path/to/model.onnx")
        
        # Should parse successfully
        assert design_space is not None
        assert design_space.global_config.output_stage.value == "rtl"
        assert design_space.global_config.working_directory == "./test_builds"
        
        # Step configuration should have defaults
        assert design_space.global_config.start_step is None
        assert design_space.global_config.stop_step is None
        assert design_space.global_config.input_type is None
        assert design_space.global_config.output_type is None