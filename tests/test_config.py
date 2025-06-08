"""
Tests for the configuration framework.
"""

import pytest
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from brainsmith.tools.hw_kernel_gen.config import (
    PipelineConfig,
    TemplateConfig,
    GenerationConfig,
    AnalysisConfig,
    ValidationConfig,
    GeneratorType,
    ValidationLevel,
    create_default_config,
    load_config
)
from brainsmith.tools.hw_kernel_gen.errors import ConfigurationError, ValidationError


class TestTemplateConfig:
    """Test TemplateConfig class."""
    
    def test_default_template_config(self):
        """Test default template configuration."""
        config = TemplateConfig()
        
        assert config.template_dirs == []
        assert config.enable_caching is True
        assert config.cache_size == 100
        assert config.custom_templates == {}
        assert config.template_overrides == {}
        assert config.trim_blocks is True
        assert config.lstrip_blocks is True
        assert config.keep_trailing_newline is True
    
    def test_template_config_with_valid_dirs(self):
        """Test template configuration with valid directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            template_dir = Path(tmpdir)
            
            config = TemplateConfig(template_dirs=[template_dir])
            
            assert len(config.template_dirs) == 1
            assert config.template_dirs[0] == template_dir
    
    def test_template_config_with_invalid_dirs(self):
        """Test template configuration with invalid directories."""
        invalid_dir = Path("/nonexistent/directory")
        
        with pytest.raises(ConfigurationError) as exc_info:
            TemplateConfig(template_dirs=[invalid_dir])
        
        assert "Template directory does not exist" in str(exc_info.value)
        assert exc_info.value.config_section == "template"
    
    def test_template_config_with_custom_templates(self):
        """Test template configuration with custom templates."""
        with tempfile.NamedTemporaryFile(suffix='.j2') as tmpfile:
            template_path = Path(tmpfile.name)
            
            config = TemplateConfig(custom_templates={"test": template_path})
            
            assert "test" in config.custom_templates
            assert config.custom_templates["test"] == template_path
    
    def test_template_config_with_invalid_custom_template(self):
        """Test template configuration with invalid custom template."""
        invalid_template = Path("/nonexistent/template.j2")
        
        with pytest.raises(ConfigurationError) as exc_info:
            TemplateConfig(custom_templates={"test": invalid_template})
        
        assert "Custom template 'test' not found" in str(exc_info.value)
        assert exc_info.value.config_section == "template"


class TestGenerationConfig:
    """Test GenerationConfig class."""
    
    def test_default_generation_config(self):
        """Test default generation configuration."""
        config = GenerationConfig()
        
        assert config.output_dir == Path("./generated")
        assert config.overwrite_existing is False
        assert config.create_directories is True
        assert config.include_debug_info is False
        assert config.include_documentation is True
        assert config.include_type_hints is True
        assert config.indent_size == 4
        assert config.use_tabs is False
        assert config.max_line_length == 88
        assert config.skip_empty_methods is True
        assert config.skip_unused_imports is True
    
    def test_generation_config_path_conversion(self):
        """Test that string paths are converted to Path objects."""
        config = GenerationConfig(output_dir="./test_output")
        
        assert isinstance(config.output_dir, Path)
        assert config.output_dir == Path("./test_output")
    
    def test_generation_config_invalid_indent_size(self):
        """Test invalid indent size validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            GenerationConfig(indent_size=0)
        
        assert "Invalid indent_size" in str(exc_info.value)
        assert exc_info.value.config_section == "generation"
        
        with pytest.raises(ConfigurationError) as exc_info:
            GenerationConfig(indent_size=10)
        
        assert "Invalid indent_size" in str(exc_info.value)


class TestAnalysisConfig:
    """Test AnalysisConfig class."""
    
    def test_default_analysis_config(self):
        """Test default analysis configuration."""
        config = AnalysisConfig()
        
        assert config.analyze_interfaces is True
        assert config.analyze_dependencies is True
        assert config.analyze_timing is False
        assert len(config.interface_patterns) > 0
        assert len(config.dependency_patterns) > 0
        assert config.max_depth == 5
        assert config.follow_includes is True
    
    def test_analysis_config_invalid_max_depth(self):
        """Test invalid max_depth validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            AnalysisConfig(max_depth=0)
        
        assert "Invalid max_depth" in str(exc_info.value)
        assert exc_info.value.config_section == "analysis"
        
        with pytest.raises(ConfigurationError) as exc_info:
            AnalysisConfig(max_depth=25)
        
        assert "Invalid max_depth" in str(exc_info.value)


class TestValidationConfig:
    """Test ValidationConfig class."""
    
    def test_default_validation_config(self):
        """Test default validation configuration."""
        config = ValidationConfig()
        
        assert config.level == ValidationLevel.MODERATE
        assert config.validate_syntax is True
        assert config.validate_semantics is True
        assert config.validate_compatibility is True
        assert config.fail_on_warnings is False
        assert config.max_errors == 10
        assert config.required_fields == []
        assert config.forbidden_patterns == []
    
    def test_validation_config_invalid_max_errors(self):
        """Test invalid max_errors validation."""
        with pytest.raises(ConfigurationError) as exc_info:
            ValidationConfig(max_errors=0)
        
        assert "Invalid max_errors" in str(exc_info.value)
        assert exc_info.value.config_section == "validation"
        
        with pytest.raises(ConfigurationError) as exc_info:
            ValidationConfig(max_errors=150)
        
        assert "Invalid max_errors" in str(exc_info.value)


class TestPipelineConfig:
    """Test PipelineConfig class."""
    
    def test_default_pipeline_config(self):
        """Test default pipeline configuration."""
        config = PipelineConfig()
        
        assert isinstance(config.template, TemplateConfig)
        assert isinstance(config.generation, GenerationConfig)
        assert isinstance(config.analysis, AnalysisConfig)
        assert isinstance(config.validation, ValidationConfig)
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
        assert config.enable_caching is True
        assert config.parallel_processing is False
        assert config.verbose is False
        assert config.debug is False
        assert config.log_level == "INFO"
        assert config.maintain_backward_compatibility is True
        assert config.legacy_mode is False
    
    def test_from_args(self):
        """Test creating configuration from arguments."""
        args = {
            'output_dir': './test_output',
            'overwrite': True,
            'debug': True,
            'verbose': True,
            'generator_type': 'rtl_backend'
        }
        
        config = PipelineConfig.from_args(args)
        
        assert config.generation.output_dir == Path('./test_output')
        assert config.generation.overwrite_existing is True
        assert config.debug is True
        assert config.verbose is True
        assert config.generator_type == GeneratorType.RTL_BACKEND
    
    def test_from_defaults_hw_custom_op(self):
        """Test creating default configuration for HW custom op."""
        config = PipelineConfig.from_defaults(GeneratorType.HW_CUSTOM_OP)
        
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
        assert config.generation.include_debug_info is True
        assert config.analysis.analyze_timing is False
        assert config.validation.level == ValidationLevel.MODERATE
    
    def test_from_defaults_rtl_backend(self):
        """Test creating default configuration for RTL backend."""
        config = PipelineConfig.from_defaults(GeneratorType.RTL_BACKEND)
        
        assert config.generator_type == GeneratorType.RTL_BACKEND
        assert config.generation.include_debug_info is False
        assert config.analysis.analyze_timing is True
        assert config.validation.level == ValidationLevel.STRICT
    
    def test_from_dict(self):
        """Test creating configuration from dictionary."""
        data = {
            'generator_type': 'hw_custom_op',
            'debug': True,
            'generation': {
                'output_dir': './test_output',
                'overwrite_existing': True
            },
            'validation': {
                'level': 'strict',
                'max_errors': 5
            }
        }
        
        config = PipelineConfig.from_dict(data)
        
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
        assert config.debug is True
        assert config.generation.output_dir == Path('./test_output')
        assert config.generation.overwrite_existing is True
        assert config.validation.level == ValidationLevel.STRICT
        assert config.validation.max_errors == 5
    
    def test_to_dict(self):
        """Test converting configuration to dictionary."""
        config = PipelineConfig()
        config.debug = True
        config.generator_type = GeneratorType.RTL_BACKEND
        
        data = config.to_dict()
        
        assert data['debug'] is True
        assert data['generator_type'] == 'rtl_backend'
        assert 'template' in data
        assert 'generation' in data
        assert 'analysis' in data
        assert 'validation' in data
    
    def test_from_file_valid(self):
        """Test loading configuration from valid JSON file."""
        config_data = {
            'generator_type': 'hw_custom_op',
            'debug': True,
            'generation': {
                'output_dir': './test_output'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = PipelineConfig.from_file(config_file)
            
            assert config.generator_type == GeneratorType.HW_CUSTOM_OP
            assert config.debug is True
            assert config.generation.output_dir == Path('./test_output')
        finally:
            Path(config_file).unlink()
    
    def test_from_file_not_found(self):
        """Test loading configuration from non-existent file."""
        with pytest.raises(ConfigurationError) as exc_info:
            PipelineConfig.from_file('/nonexistent/config.json')
        
        assert "Configuration file not found" in str(exc_info.value)
        assert exc_info.value.config_section == "file"
    
    def test_from_file_invalid_json(self):
        """Test loading configuration from invalid JSON file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            f.write('{ invalid json }')
            config_file = f.name
        
        try:
            with pytest.raises(ConfigurationError) as exc_info:
                PipelineConfig.from_file(config_file)
            
            assert "Invalid JSON" in str(exc_info.value)
            assert exc_info.value.config_section == "file"
        finally:
            Path(config_file).unlink()
    
    def test_to_file(self):
        """Test saving configuration to file."""
        config = PipelineConfig()
        config.debug = True
        
        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            config_file = f.name
        
        try:
            config.to_file(config_file)
            
            # Verify file was created and contains expected data
            assert Path(config_file).exists()
            
            with open(config_file, 'r') as f:
                data = json.load(f)
            
            assert data['debug'] is True
            assert 'generator_type' in data
        finally:
            Path(config_file).unlink()
    
    def test_validate_success(self):
        """Test successful configuration validation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config = PipelineConfig()
            config.generation.output_dir = Path(tmpdir)
            
            # Should not raise an exception
            config.validate()
    
    def test_validate_custom_templates_without_dirs(self):
        """Test validation fails for custom templates without template dirs."""
        with tempfile.NamedTemporaryFile(suffix='.j2') as tmpfile:
            config = PipelineConfig()
            config.template.custom_templates = {"test": Path(tmpfile.name)}
            config.template.template_dirs = []
            
            with pytest.raises(ValidationError) as exc_info:
                config.validate()
            
            assert "Custom templates specified but no template directories" in str(exc_info.value)
    
    def test_validate_rtl_backend_requirements(self):
        """Test validation for RTL backend specific requirements."""
        config = PipelineConfig()
        config.generator_type = GeneratorType.RTL_BACKEND
        config.analysis.analyze_interfaces = False
        
        with pytest.raises(ValidationError) as exc_info:
            config.validate()
        
        assert "RTL backend requires interface analysis" in str(exc_info.value)
    
    def test_get_effective_config(self):
        """Test getting effective configuration with validation."""
        config = PipelineConfig()
        
        effective_config = config.get_effective_config()
        
        # Should be a different instance
        assert effective_config is not config
        # But should have same values
        assert effective_config.generator_type == config.generator_type


class TestConfigurationHelpers:
    """Test configuration helper functions."""
    
    def test_create_default_config(self):
        """Test creating default configuration."""
        config = create_default_config()
        
        assert isinstance(config, PipelineConfig)
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
    
    def test_create_default_config_with_type(self):
        """Test creating default configuration with specific type."""
        config = create_default_config(GeneratorType.RTL_BACKEND)
        
        assert config.generator_type == GeneratorType.RTL_BACKEND
    
    def test_load_config_none(self):
        """Test loading configuration with None source."""
        config = load_config(None)
        
        assert isinstance(config, PipelineConfig)
        assert config.generator_type == GeneratorType.HW_CUSTOM_OP
    
    def test_load_config_dict(self):
        """Test loading configuration from dictionary."""
        data = {'debug': True, 'generator_type': 'rtl_backend'}
        config = load_config(data)
        
        assert config.debug is True
        assert config.generator_type == GeneratorType.RTL_BACKEND
    
    def test_load_config_file(self):
        """Test loading configuration from file."""
        config_data = {'debug': True}
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(config_data, f)
            config_file = f.name
        
        try:
            config = load_config(config_file)
            assert config.debug is True
        finally:
            Path(config_file).unlink()
    
    def test_load_config_invalid_type(self):
        """Test loading configuration with invalid source type."""
        with pytest.raises(ConfigurationError) as exc_info:
            load_config(123)  # Invalid type
        
        assert "Invalid configuration source type" in str(exc_info.value)
        assert exc_info.value.config_section == "loading"


class TestEnums:
    """Test enum classes."""
    
    def test_generator_type_enum(self):
        """Test GeneratorType enum."""
        assert GeneratorType.HW_CUSTOM_OP.value == "hw_custom_op"
        assert GeneratorType.RTL_BACKEND.value == "rtl_backend"
        
        # Test conversion from string
        assert GeneratorType("hw_custom_op") == GeneratorType.HW_CUSTOM_OP
        assert GeneratorType("rtl_backend") == GeneratorType.RTL_BACKEND
    
    def test_validation_level_enum(self):
        """Test ValidationLevel enum."""
        assert ValidationLevel.STRICT.value == "strict"
        assert ValidationLevel.MODERATE.value == "moderate"
        assert ValidationLevel.RELAXED.value == "relaxed"
        
        # Test conversion from string
        assert ValidationLevel("strict") == ValidationLevel.STRICT
        assert ValidationLevel("moderate") == ValidationLevel.MODERATE
        assert ValidationLevel("relaxed") == ValidationLevel.RELAXED