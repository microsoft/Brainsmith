"""
Configuration framework for the Hardware Kernel Generator.

This module provides centralized configuration management for the entire
HWKG pipeline, including validation, defaults, and factory methods.
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
import json
from enum import Enum

from .errors import ConfigurationError, ValidationError


class GeneratorType(Enum):
    """Supported generator types."""
    HW_CUSTOM_OP = "hw_custom_op"
    RTL_BACKEND = "rtl_backend"


class ValidationLevel(Enum):
    """Validation strictness levels."""
    STRICT = "strict"
    MODERATE = "moderate"
    RELAXED = "relaxed"


@dataclass
class TemplateConfig:
    """Configuration for template handling."""
    
    # Template directories
    template_dirs: List[Path] = field(default_factory=list)
    
    # Template caching
    enable_caching: bool = True
    cache_size: int = 100
    
    # Template customization
    custom_templates: Dict[str, Path] = field(default_factory=dict)
    template_overrides: Dict[str, str] = field(default_factory=dict)
    
    # Jinja2 environment settings
    trim_blocks: bool = True
    lstrip_blocks: bool = True
    keep_trailing_newline: bool = True
    
    def __post_init__(self):
        """Validate template configuration."""
        # Convert string paths to Path objects
        self.template_dirs = [Path(d) for d in self.template_dirs]
        
        # Validate template directories exist
        for template_dir in self.template_dirs:
            if not template_dir.exists():
                raise ConfigurationError(
                    f"Template directory does not exist: {template_dir}",
                    config_section="template",
                    suggestion="Ensure all template directories exist before configuration"
                )
        
        # Validate custom templates exist
        for name, path in self.custom_templates.items():
            template_path = Path(path)
            if not template_path.exists():
                raise ConfigurationError(
                    f"Custom template '{name}' not found: {template_path}",
                    config_section="template",
                    suggestion=f"Ensure custom template file exists: {template_path}"
                )


@dataclass
class GenerationConfig:
    """Configuration for code generation."""
    
    # Output settings
    output_dir: Path = field(default_factory=lambda: Path("./generated"))
    overwrite_existing: bool = False
    create_directories: bool = True
    
    # Generation options
    include_debug_info: bool = False
    include_documentation: bool = True
    include_type_hints: bool = True
    
    # Code formatting
    indent_size: int = 4
    use_tabs: bool = False
    max_line_length: int = 88
    
    # Generation filters
    skip_empty_methods: bool = True
    skip_unused_imports: bool = True
    
    def __post_init__(self):
        """Validate generation configuration."""
        self.output_dir = Path(self.output_dir)
        
        # Validate output directory can be created
        if not self.output_dir.exists() and not self.create_directories:
            raise ConfigurationError(
                f"Output directory does not exist: {self.output_dir}",
                config_section="generation",
                suggestion="Set create_directories=True or create the directory manually"
            )
        
        # Validate formatting options
        if self.indent_size < 1 or self.indent_size > 8:
            raise ConfigurationError(
                f"Invalid indent_size: {self.indent_size}",
                config_section="generation",
                suggestion="Use indent_size between 1 and 8"
            )


@dataclass
class AnalysisConfig:
    """Configuration for RTL analysis."""
    
    # Analysis options
    analyze_interfaces: bool = True
    analyze_dependencies: bool = True
    analyze_timing: bool = False
    
    # Interface analysis
    interface_patterns: List[str] = field(default_factory=lambda: [
        r"input\s+.*",
        r"output\s+.*",
        r"inout\s+.*"
    ])
    
    # Dependency analysis
    dependency_patterns: List[str] = field(default_factory=lambda: [
        r"import\s+.*",
        r"include\s+.*",
        r"`include\s+.*"
    ])
    
    # Analysis depth
    max_depth: int = 5
    follow_includes: bool = True
    
    def __post_init__(self):
        """Validate analysis configuration."""
        if self.max_depth < 1 or self.max_depth > 20:
            raise ConfigurationError(
                f"Invalid max_depth: {self.max_depth}",
                config_section="analysis",
                suggestion="Use max_depth between 1 and 20"
            )


@dataclass
class ValidationConfig:
    """Configuration for validation."""
    
    # Validation levels
    level: ValidationLevel = ValidationLevel.MODERATE
    
    # Validation options
    validate_syntax: bool = True
    validate_semantics: bool = True
    validate_compatibility: bool = True
    
    # Error handling
    fail_on_warnings: bool = False
    max_errors: int = 10
    
    # Validation rules
    required_fields: List[str] = field(default_factory=list)
    forbidden_patterns: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Validate validation configuration."""
        if self.max_errors < 1 or self.max_errors > 100:
            raise ConfigurationError(
                f"Invalid max_errors: {self.max_errors}",
                config_section="validation",
                suggestion="Use max_errors between 1 and 100"
            )


@dataclass
class PipelineConfig:
    """Main configuration for the HWKG pipeline."""
    
    # Sub-configurations
    template: TemplateConfig = field(default_factory=TemplateConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Pipeline options
    generator_type: GeneratorType = GeneratorType.HW_CUSTOM_OP
    enable_caching: bool = True
    parallel_processing: bool = False
    
    # Logging and debugging
    verbose: bool = False
    debug: bool = False
    log_level: str = "INFO"
    
    # Compatibility
    maintain_backward_compatibility: bool = True
    legacy_mode: bool = False
    
    @classmethod
    def from_args(cls, args: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from command line arguments or dictionary."""
        config = cls()
        
        # Map common argument patterns to configuration
        arg_mapping = {
            'output_dir': ('generation', 'output_dir'),
            'template_dir': ('template', 'template_dirs'),
            'overwrite': ('generation', 'overwrite_existing'),
            'debug': ('debug',),
            'verbose': ('verbose',),
            'generator_type': ('generator_type',),
        }
        
        for arg_name, config_path in arg_mapping.items():
            if arg_name in args:
                value = args[arg_name]
                
                # Special handling for specific arguments
                if arg_name == 'template_dir':
                    # Convert single template dir to list
                    if isinstance(value, (str, Path)):
                        value = [value]
                elif arg_name == 'generator_type':
                    # Convert string to enum
                    if isinstance(value, str):
                        value = GeneratorType(value)
                
                # Set the configuration value
                if len(config_path) == 1:
                    setattr(config, config_path[0], value)
                else:
                    sub_config = getattr(config, config_path[0])
                    setattr(sub_config, config_path[1], value)
                    
                    # Re-run __post_init__ for sub-config to apply path conversions
                    if hasattr(sub_config, '__post_init__'):
                        sub_config.__post_init__()
        
        return config
    
    @classmethod
    def from_defaults(cls, generator_type: GeneratorType = GeneratorType.HW_CUSTOM_OP) -> 'PipelineConfig':
        """Create configuration with sensible defaults for a generator type."""
        config = cls(generator_type=generator_type)
        
        # Generator-specific defaults
        if generator_type == GeneratorType.HW_CUSTOM_OP:
            config.generation.include_debug_info = True
            config.analysis.analyze_timing = False
            config.validation.level = ValidationLevel.MODERATE
        elif generator_type == GeneratorType.RTL_BACKEND:
            config.generation.include_debug_info = False
            config.analysis.analyze_timing = True
            config.validation.level = ValidationLevel.STRICT
        
        return config
    
    @classmethod
    def from_file(cls, config_file: Union[str, Path]) -> 'PipelineConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_file)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Configuration file not found: {config_path}",
                config_section="file",
                suggestion=f"Create configuration file or check path: {config_path}"
            )
        
        try:
            with open(config_path, 'r') as f:
                data = json.load(f)
            
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ConfigurationError(
                f"Invalid JSON in configuration file: {e}",
                config_section="file",
                suggestion="Check JSON syntax in configuration file"
            )
        except Exception as e:
            raise ConfigurationError(
                f"Error loading configuration file: {e}",
                config_section="file",
                suggestion="Ensure file is readable and contains valid configuration"
            )
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineConfig':
        """Create configuration from dictionary."""
        # Extract sub-configuration data
        template_data = data.pop('template', {})
        generation_data = data.pop('generation', {})
        analysis_data = data.pop('analysis', {})
        validation_data = data.pop('validation', {})
        
        # Create sub-configurations
        template_config = TemplateConfig(**template_data)
        generation_config = GenerationConfig(**generation_data)
        analysis_config = AnalysisConfig(**analysis_data)
        validation_config = ValidationConfig(**validation_data)
        
        # Handle enum conversions
        if 'generator_type' in data:
            if isinstance(data['generator_type'], str):
                data['generator_type'] = GeneratorType(data['generator_type'])
        
        if 'level' in validation_data:
            if isinstance(validation_data['level'], str):
                validation_config.level = ValidationLevel(validation_data['level'])
        
        # Create main configuration
        config = cls(
            template=template_config,
            generation=generation_config,
            analysis=analysis_config,
            validation=validation_config,
            **data
        )
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        data = asdict(self)
        
        # Convert enums to strings
        data['generator_type'] = self.generator_type.value
        data['validation']['level'] = self.validation.level.value
        
        # Convert Path objects to strings
        data['generation']['output_dir'] = str(self.generation.output_dir)
        data['template']['template_dirs'] = [str(d) for d in self.template.template_dirs]
        data['template']['custom_templates'] = {
            k: str(v) for k, v in self.template.custom_templates.items()
        }
        
        return data
    
    def to_file(self, config_file: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_file)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                json.dump(self.to_dict(), f, indent=2)
        except Exception as e:
            raise ConfigurationError(
                f"Error saving configuration file: {e}",
                config_section="file",
                suggestion="Ensure directory is writable"
            )
    
    def validate(self) -> None:
        """Validate the entire configuration."""
        # Validate sub-configurations (done in __post_init__)
        # Additional cross-configuration validation
        
        # Ensure template directories are set if using custom templates
        if self.template.custom_templates and not self.template.template_dirs:
            raise ValidationError(
                "Custom templates specified but no template directories configured",
                validation_type="configuration",
                suggestion="Add template directories to template.template_dirs"
            )
        
        # Ensure output directory is writable
        if self.generation.output_dir.exists():
            if not self.generation.output_dir.is_dir():
                raise ValidationError(
                    f"Output path is not a directory: {self.generation.output_dir}",
                    validation_type="configuration",
                    suggestion="Use a directory path for output_dir"
                )
        
        # Validate generator-specific requirements
        if self.generator_type == GeneratorType.RTL_BACKEND:
            if not self.analysis.analyze_interfaces:
                raise ValidationError(
                    "RTL backend requires interface analysis",
                    validation_type="configuration",
                    suggestion="Set analysis.analyze_interfaces=True for RTL backend"
                )
    
    def get_effective_config(self) -> 'PipelineConfig':
        """Get configuration with all defaults resolved and validation applied."""
        # Create a copy to avoid modifying the original
        config = PipelineConfig.from_dict(self.to_dict())
        
        # Apply validation
        config.validate()
        
        return config


def create_default_config(generator_type: GeneratorType = GeneratorType.HW_CUSTOM_OP) -> PipelineConfig:
    """Create a default configuration for common use cases."""
    return PipelineConfig.from_defaults(generator_type)


def load_config(config_source: Union[str, Path, Dict[str, Any], None] = None) -> PipelineConfig:
    """Load configuration from various sources."""
    if config_source is None:
        return create_default_config()
    elif isinstance(config_source, (str, Path)):
        return PipelineConfig.from_file(config_source)
    elif isinstance(config_source, dict):
        return PipelineConfig.from_dict(config_source)
    else:
        raise ConfigurationError(
            f"Invalid configuration source type: {type(config_source)}",
            config_section="loading",
            suggestion="Use file path, dictionary, or None for defaults"
        )