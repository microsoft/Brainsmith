"""
Unit tests for Blueprint parser.
"""

import pytest
import yaml
from brainsmith.core_v3.phase1.parser import BlueprintParser, load_blueprint
from brainsmith.core_v3.phase1.exceptions import BlueprintParseError
from brainsmith.core_v3.phase1.data_structures import SearchStrategy, OutputStage


class TestBlueprintParser:
    """Test BlueprintParser class."""
    
    @pytest.fixture
    def parser(self):
        return BlueprintParser()
    
    @pytest.fixture
    def minimal_blueprint(self):
        return {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": ["quantization"],
                "build_steps": ["ConvertToHW"],
                "config_flags": {}
            },
            "processing": {},
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
    
    def test_parse_minimal_blueprint(self, parser, minimal_blueprint):
        design_space = parser.parse(minimal_blueprint, "model.onnx")
        
        assert design_space.model_path == "model.onnx"
        assert len(design_space.hw_compiler_space.kernels) == 1
        assert design_space.hw_compiler_space.kernels[0] == "MatMul"
        assert len(design_space.hw_compiler_space.transforms) == 1
        assert design_space.search_config.strategy == SearchStrategy.EXHAUSTIVE
        assert design_space.global_config.output_stage == OutputStage.RTL
    
    def test_version_validation(self, parser):
        invalid_version = {
            "version": "2.0",
            "hw_compiler": {"kernels": [], "transforms": [], "build_steps": []}
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(invalid_version, "model.onnx")
        assert "Unsupported blueprint version" in str(exc.value)
    
    def test_missing_version(self, parser):
        no_version = {
            "hw_compiler": {"kernels": [], "transforms": [], "build_steps": []}
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(no_version, "model.onnx")
        assert "version not specified" in str(exc.value)
    
    def test_parse_kernel_formats(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [
                    "MatMul",                              # Simple string
                    ["Softmax", ["hls", "rtl"]],          # List with backends
                    ["LayerNorm", "RMSNorm"],              # Mutually exclusive
                    [None, "Transpose"],                   # Optional
                    ["~Attention", ["flash"]],             # Optional with backends
                ],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        assert len(design_space.hw_compiler_space.kernels) == 5
    
    def test_parse_transform_formats(self, parser):
        # Test flat transforms
        blueprint_flat = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [],
                "transforms": [
                    "quantization",
                    "~folding",                          # Optional
                    ["stream_v1", "stream_v2"],          # Mutually exclusive
                    [None, "optimize"],                  # Optional choice
                ],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        design_space = parser.parse(blueprint_flat, "model.onnx")
        assert len(design_space.hw_compiler_space.transforms) == 4
    
    def test_parse_phase_based_transforms(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": [],
                "transforms": {
                    "pre_hw": ["quantization", "~folding"],
                    "post_hw": ["optimization"]
                },
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        transforms = design_space.hw_compiler_space.transforms
        assert isinstance(transforms, dict)
        assert "pre_hw" in transforms
        assert "post_hw" in transforms
    
    def test_parse_processing_steps(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "processing": {
                "preprocessing": [
                    {
                        "name": "normalization",
                        "options": [
                            {"enabled": True, "method": "standard"},
                            {"enabled": False}
                        ]
                    }
                ],
                "postprocessing": [
                    {
                        "name": "analysis",
                        "options": [
                            {"enabled": True, "detailed": True}
                        ]
                    }
                ]
            },
            "search": {"strategy": "exhaustive"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        assert len(design_space.processing_space.preprocessing) == 1
        assert len(design_space.processing_space.preprocessing[0]) == 2
        assert len(design_space.processing_space.postprocessing) == 1
    
    def test_parse_search_constraints(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {
                "strategy": "exhaustive",
                "constraints": [
                    {"metric": "lut_utilization", "operator": "<=", "value": 0.85},
                    {"metric": "throughput", "operator": ">=", "value": 1000}
                ],
                "max_evaluations": 100,
                "timeout_minutes": 60,
                "parallel_builds": 4
            },
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        assert len(design_space.search_config.constraints) == 2
        assert design_space.search_config.max_evaluations == 100
        assert design_space.search_config.timeout_minutes == 60
        assert design_space.search_config.parallel_builds == 4
    
    def test_parse_global_config(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {
                "output_stage": "stitched_ip",
                "working_directory": "/tmp/builds",
                "cache_results": False,
                "save_artifacts": True,
                "log_level": "DEBUG"
            }
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        assert design_space.global_config.output_stage == OutputStage.STITCHED_IP
        assert design_space.global_config.working_directory == "/tmp/builds"
        assert design_space.global_config.cache_results == False
        assert design_space.global_config.save_artifacts == True
        assert design_space.global_config.log_level == "DEBUG"
    
    def test_invalid_data_types(self, parser):
        # Kernels not a list
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": "MatMul",  # Should be list
                "transforms": [],
                "build_steps": []
            }
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint, "model.onnx")
        assert "kernels must be a list" in str(exc.value)
    
    def test_invalid_search_strategy(self, parser):
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "invalid_strategy"},
            "global": {"output_stage": "rtl", "working_directory": "./builds"}
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint, "model.onnx")
        assert "Unknown search.strategy" in str(exc.value)
    
    def test_parse_global_config_with_limits(self, parser):
        """Test parsing global config with new max_combinations and timeout_minutes fields."""
        blueprint = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {
                "output_stage": "rtl",
                "working_directory": "./builds",
                "max_combinations": 50000,
                "timeout_minutes": 90
            }
        }
        
        design_space = parser.parse(blueprint, "model.onnx")
        assert design_space.global_config.max_combinations == 50000
        assert design_space.global_config.timeout_minutes == 90
    
    def test_timeout_priority_resolution(self, parser):
        """Test timeout priority: search > global > library config."""
        # Test 1: search.timeout_minutes takes precedence
        blueprint_search_priority = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {
                "strategy": "exhaustive",
                "timeout_minutes": 30  # Should take precedence
            },
            "global": {
                "timeout_minutes": 90  # Should be ignored
            }
        }
        
        design_space = parser.parse(blueprint_search_priority, "model.onnx")
        assert design_space.search_config.timeout_minutes == 30
        
        # Test 2: global.timeout_minutes used when search doesn't specify
        blueprint_global_priority = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {
                "strategy": "exhaustive"
                # No timeout_minutes specified
            },
            "global": {
                "timeout_minutes": 90  # Should be used
            }
        }
        
        design_space = parser.parse(blueprint_global_priority, "model.onnx")
        assert design_space.search_config.timeout_minutes == 90
        
        # Test 3: library config used when neither search nor global specify
        blueprint_library_config = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {
                "strategy": "exhaustive"
            },
            "global": {
                "output_stage": "rtl"
                # No timeout_minutes specified
            }
        }
        
        design_space = parser.parse(blueprint_library_config, "model.onnx")
        # Should get default from library config (60 minutes)
        assert design_space.search_config.timeout_minutes == 60
    
    def test_invalid_global_config_values(self, parser):
        """Test handling of invalid global config values."""
        # Invalid max_combinations
        blueprint_invalid_max = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {
                "max_combinations": -100  # Invalid
            }
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_invalid_max, "model.onnx")
        assert "max_combinations must be a positive integer" in str(exc.value)
        
        # Invalid timeout_minutes
        blueprint_invalid_timeout = {
            "version": "3.0",
            "hw_compiler": {
                "kernels": ["MatMul"],
                "transforms": [],
                "build_steps": ["ConvertToHW"]
            },
            "search": {"strategy": "exhaustive"},
            "global": {
                "timeout_minutes": "not_a_number"  # Invalid
            }
        }
        
        with pytest.raises(BlueprintParseError) as exc:
            parser.parse(blueprint_invalid_timeout, "model.onnx")
        assert "timeout_minutes must be an integer" in str(exc.value)


class TestLoadBlueprint:
    """Test load_blueprint function."""
    
    def test_load_valid_yaml(self, tmp_path):
        # Create a temporary YAML file
        yaml_file = tmp_path / "test.yaml"
        yaml_file.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - MatMul
""")
        
        data = load_blueprint(str(yaml_file))
        assert data["version"] == "3.0"
        assert "hw_compiler" in data
    
    def test_file_not_found(self):
        with pytest.raises(BlueprintParseError) as exc:
            load_blueprint("nonexistent.yaml")
        assert "Blueprint file not found" in str(exc.value)
    
    def test_invalid_yaml_syntax(self, tmp_path):
        yaml_file = tmp_path / "invalid.yaml"
        yaml_file.write_text("""
version: "3.0"
hw_compiler:
  kernels:
    - MatMul
    invalid syntax here
""")
        
        with pytest.raises(BlueprintParseError) as exc:
            load_blueprint(str(yaml_file))
        assert "Invalid YAML syntax" in str(exc.value)