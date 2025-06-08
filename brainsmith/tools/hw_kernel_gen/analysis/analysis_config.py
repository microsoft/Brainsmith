"""
Analysis Configuration for Enhanced Interface and Pragma Analysis.

This module provides configuration classes and profiles for customizing
the analysis behavior and performance characteristics.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Any, Union
from pathlib import Path
from enum import Enum

from ..enhanced_config import PipelineConfig, AnalysisConfig
from ..errors import ConfigurationError


class AnalysisStrategy(Enum):
    """Analysis strategy modes."""
    FAST = "fast"                   # Quick analysis with basic patterns
    COMPREHENSIVE = "comprehensive" # Thorough analysis with all patterns
    CUSTOM = "custom"              # User-defined analysis configuration


class ConfidenceThreshold(Enum):
    """Confidence threshold levels for classifications."""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERY_HIGH = 0.9


@dataclass
class InterfaceAnalysisConfig:
    """Configuration for interface analysis."""
    
    # Analysis strategy
    strategy: AnalysisStrategy = AnalysisStrategy.COMPREHENSIVE
    confidence_threshold: float = ConfidenceThreshold.MEDIUM.value
    
    # Pattern configuration
    enable_axi_stream_detection: bool = True
    enable_axi_lite_detection: bool = True
    enable_control_detection: bool = True
    enable_custom_patterns: bool = True
    
    # Analysis options
    strict_naming_conventions: bool = False
    require_complete_interfaces: bool = False
    allow_partial_matches: bool = True
    
    # Performance options
    enable_pattern_caching: bool = True
    max_signal_count: int = 1000
    analysis_timeout: float = 30.0  # seconds
    
    # Dataflow integration
    enable_dataflow_conversion: bool = True
    infer_tensor_dimensions: bool = True
    validate_dataflow_constraints: bool = True
    
    # Custom patterns
    custom_interface_patterns: List[str] = field(default_factory=list)
    custom_signal_patterns: Dict[str, List[str]] = field(default_factory=dict)
    
    # Validation options
    validate_signal_directions: bool = True
    validate_signal_widths: bool = True
    validate_naming_consistency: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if not (0.0 <= self.confidence_threshold <= 1.0):
            raise ConfigurationError(
                f"Invalid confidence_threshold: {self.confidence_threshold}",
                config_section="interface_analysis",
                suggestion="Use value between 0.0 and 1.0"
            )
        
        if self.max_signal_count < 1:
            raise ConfigurationError(
                f"Invalid max_signal_count: {self.max_signal_count}",
                config_section="interface_analysis",
                suggestion="Use positive integer"
            )
        
        if self.analysis_timeout <= 0:
            raise ConfigurationError(
                f"Invalid analysis_timeout: {self.analysis_timeout}",
                config_section="interface_analysis",
                suggestion="Use positive timeout value"
            )


@dataclass
class PragmaAnalysisConfig:
    """Configuration for pragma analysis."""
    
    # Analysis strategy
    strategy: AnalysisStrategy = AnalysisStrategy.COMPREHENSIVE
    
    # Pragma types to process
    enable_brainsmith_pragmas: bool = True
    enable_hls_pragmas: bool = True
    enable_interface_pragmas: bool = True
    enable_custom_pragmas: bool = True
    
    # Parsing options
    case_sensitive_parsing: bool = False
    allow_malformed_pragmas: bool = True
    strict_parameter_validation: bool = False
    
    # Reference validation
    validate_signal_references: bool = True
    validate_parameter_references: bool = True
    allow_forward_references: bool = True
    
    # Constraint processing
    enable_constraint_generation: bool = True
    enable_dataflow_constraints: bool = True
    enable_parallelism_inference: bool = True
    
    # Performance options
    enable_pragma_caching: bool = True
    max_pragma_count: int = 500
    processing_timeout: float = 15.0  # seconds
    
    # Custom patterns
    custom_pragma_patterns: Dict[str, List[str]] = field(default_factory=dict)
    custom_parameter_patterns: Dict[str, str] = field(default_factory=dict)
    
    # Error handling
    continue_on_parse_errors: bool = True
    continue_on_validation_errors: bool = True
    
    def __post_init__(self):
        """Validate configuration."""
        if self.max_pragma_count < 1:
            raise ConfigurationError(
                f"Invalid max_pragma_count: {self.max_pragma_count}",
                config_section="pragma_analysis",
                suggestion="Use positive integer"
            )
        
        if self.processing_timeout <= 0:
            raise ConfigurationError(
                f"Invalid processing_timeout: {self.processing_timeout}",
                config_section="pragma_analysis",
                suggestion="Use positive timeout value"
            )


@dataclass
class AnalysisMetrics:
    """Metrics and performance tracking for analysis."""
    
    # Interface analysis metrics
    interface_analysis_count: int = 0
    interface_analysis_time: float = 0.0
    interface_classification_accuracy: float = 0.0
    
    # Pragma analysis metrics
    pragma_processing_count: int = 0
    pragma_processing_time: float = 0.0
    pragma_parse_success_rate: float = 0.0
    
    # Dataflow conversion metrics
    dataflow_conversion_count: int = 0
    dataflow_conversion_time: float = 0.0
    dataflow_conversion_success_rate: float = 0.0
    
    # Cache metrics
    cache_hit_rate: float = 0.0
    cache_size: int = 0
    
    # Error metrics
    error_count: int = 0
    warning_count: int = 0
    
    def update_interface_metrics(
        self,
        analysis_time: float,
        accuracy: float = None
    ) -> None:
        """Update interface analysis metrics."""
        self.interface_analysis_count += 1
        self.interface_analysis_time += analysis_time
        
        if accuracy is not None:
            # Calculate running average
            n = self.interface_analysis_count
            self.interface_classification_accuracy = (
                (self.interface_classification_accuracy * (n - 1) + accuracy) / n
            )
    
    def update_pragma_metrics(
        self,
        processing_time: float,
        success_rate: float = None
    ) -> None:
        """Update pragma processing metrics."""
        self.pragma_processing_count += 1
        self.pragma_processing_time += processing_time
        
        if success_rate is not None:
            # Calculate running average
            n = self.pragma_processing_count
            self.pragma_parse_success_rate = (
                (self.pragma_parse_success_rate * (n - 1) + success_rate) / n
            )
    
    def update_dataflow_metrics(
        self,
        conversion_time: float,
        success: bool = True
    ) -> None:
        """Update dataflow conversion metrics."""
        self.dataflow_conversion_count += 1
        self.dataflow_conversion_time += conversion_time
        
        if success:
            # Calculate running success rate
            n = self.dataflow_conversion_count
            current_successes = self.dataflow_conversion_success_rate * (n - 1) + 1.0
            self.dataflow_conversion_success_rate = current_successes / n
        else:
            # Calculate running success rate
            n = self.dataflow_conversion_count
            current_successes = self.dataflow_conversion_success_rate * (n - 1)
            self.dataflow_conversion_success_rate = current_successes / n
    
    def update_cache_metrics(self, hit_rate: float, cache_size: int) -> None:
        """Update cache performance metrics."""
        self.cache_hit_rate = hit_rate
        self.cache_size = cache_size
    
    def add_error(self) -> None:
        """Add to error count."""
        self.error_count += 1
    
    def add_warning(self) -> None:
        """Add to warning count."""
        self.warning_count += 1
    
    def get_average_interface_time(self) -> float:
        """Get average interface analysis time."""
        if self.interface_analysis_count > 0:
            return self.interface_analysis_time / self.interface_analysis_count
        return 0.0
    
    def get_average_pragma_time(self) -> float:
        """Get average pragma processing time."""
        if self.pragma_processing_count > 0:
            return self.pragma_processing_time / self.pragma_processing_count
        return 0.0
    
    def get_average_dataflow_time(self) -> float:
        """Get average dataflow conversion time."""
        if self.dataflow_conversion_count > 0:
            return self.dataflow_conversion_time / self.dataflow_conversion_count
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary representation."""
        return {
            "interface_analysis": {
                "count": self.interface_analysis_count,
                "total_time": self.interface_analysis_time,
                "average_time": self.get_average_interface_time(),
                "accuracy": self.interface_classification_accuracy
            },
            "pragma_processing": {
                "count": self.pragma_processing_count,
                "total_time": self.pragma_processing_time,
                "average_time": self.get_average_pragma_time(),
                "success_rate": self.pragma_parse_success_rate
            },
            "dataflow_conversion": {
                "count": self.dataflow_conversion_count,
                "total_time": self.dataflow_conversion_time,
                "average_time": self.get_average_dataflow_time(),
                "success_rate": self.dataflow_conversion_success_rate
            },
            "cache": {
                "hit_rate": self.cache_hit_rate,
                "size": self.cache_size
            },
            "errors": {
                "error_count": self.error_count,
                "warning_count": self.warning_count
            }
        }


@dataclass
class AnalysisProfile:
    """Pre-configured analysis profile for common use cases."""
    
    name: str
    description: str
    interface_config: InterfaceAnalysisConfig
    pragma_config: PragmaAnalysisConfig
    
    @classmethod
    def create_fast_profile(cls) -> "AnalysisProfile":
        """Create a fast analysis profile for quick results."""
        interface_config = InterfaceAnalysisConfig(
            strategy=AnalysisStrategy.FAST,
            confidence_threshold=ConfidenceThreshold.LOW.value,
            enable_custom_patterns=False,
            strict_naming_conventions=False,
            require_complete_interfaces=False,
            validate_dataflow_constraints=False,
            analysis_timeout=5.0
        )
        
        pragma_config = PragmaAnalysisConfig(
            strategy=AnalysisStrategy.FAST,
            enable_custom_pragmas=False,
            strict_parameter_validation=False,
            validate_signal_references=False,
            validate_parameter_references=False,
            enable_parallelism_inference=False,
            processing_timeout=3.0
        )
        
        return cls(
            name="fast",
            description="Fast analysis with basic pattern matching",
            interface_config=interface_config,
            pragma_config=pragma_config
        )
    
    @classmethod
    def create_comprehensive_profile(cls) -> "AnalysisProfile":
        """Create a comprehensive analysis profile for thorough analysis."""
        interface_config = InterfaceAnalysisConfig(
            strategy=AnalysisStrategy.COMPREHENSIVE,
            confidence_threshold=ConfidenceThreshold.HIGH.value,
            enable_custom_patterns=True,
            strict_naming_conventions=True,
            require_complete_interfaces=True,
            validate_dataflow_constraints=True,
            validate_signal_directions=True,
            validate_signal_widths=True,
            validate_naming_consistency=True
        )
        
        pragma_config = PragmaAnalysisConfig(
            strategy=AnalysisStrategy.COMPREHENSIVE,
            enable_custom_pragmas=True,
            strict_parameter_validation=True,
            validate_signal_references=True,
            validate_parameter_references=True,
            enable_constraint_generation=True,
            enable_dataflow_constraints=True,
            enable_parallelism_inference=True
        )
        
        return cls(
            name="comprehensive",
            description="Comprehensive analysis with full validation",
            interface_config=interface_config,
            pragma_config=pragma_config
        )
    
    @classmethod
    def create_dataflow_optimized_profile(cls) -> "AnalysisProfile":
        """Create a dataflow-optimized analysis profile."""
        interface_config = InterfaceAnalysisConfig(
            strategy=AnalysisStrategy.COMPREHENSIVE,
            confidence_threshold=ConfidenceThreshold.MEDIUM.value,
            enable_dataflow_conversion=True,
            infer_tensor_dimensions=True,
            validate_dataflow_constraints=True
        )
        
        pragma_config = PragmaAnalysisConfig(
            strategy=AnalysisStrategy.COMPREHENSIVE,
            enable_constraint_generation=True,
            enable_dataflow_constraints=True,
            enable_parallelism_inference=True,
            validate_signal_references=True
        )
        
        return cls(
            name="dataflow_optimized",
            description="Optimized for dataflow modeling and constraint generation",
            interface_config=interface_config,
            pragma_config=pragma_config
        )
    
    @classmethod
    def create_legacy_compatible_profile(cls) -> "AnalysisProfile":
        """Create a legacy-compatible analysis profile."""
        interface_config = InterfaceAnalysisConfig(
            strategy=AnalysisStrategy.FAST,
            confidence_threshold=ConfidenceThreshold.LOW.value,
            enable_dataflow_conversion=False,
            strict_naming_conventions=False,
            require_complete_interfaces=False,
            allow_partial_matches=True
        )
        
        pragma_config = PragmaAnalysisConfig(
            strategy=AnalysisStrategy.FAST,
            allow_malformed_pragmas=True,
            strict_parameter_validation=False,
            continue_on_parse_errors=True,
            continue_on_validation_errors=True,
            enable_dataflow_constraints=False
        )
        
        return cls(
            name="legacy_compatible",
            description="Compatible with legacy RTL and pragma formats",
            interface_config=interface_config,
            pragma_config=pragma_config
        )


def create_analysis_config(
    pipeline_config: PipelineConfig = None,
    profile: str = "comprehensive",
    custom_interface_config: InterfaceAnalysisConfig = None,
    custom_pragma_config: PragmaAnalysisConfig = None
) -> tuple[InterfaceAnalysisConfig, PragmaAnalysisConfig]:
    """
    Create analysis configuration from profile or custom configs.
    
    Args:
        pipeline_config: Pipeline configuration for context
        profile: Analysis profile name
        custom_interface_config: Custom interface analysis config
        custom_pragma_config: Custom pragma analysis config
        
    Returns:
        Tuple of (interface_config, pragma_config)
    """
    if custom_interface_config and custom_pragma_config:
        return custom_interface_config, custom_pragma_config
    
    # Create profile-based configuration
    if profile == "fast":
        analysis_profile = AnalysisProfile.create_fast_profile()
    elif profile == "comprehensive":
        analysis_profile = AnalysisProfile.create_comprehensive_profile()
    elif profile == "dataflow_optimized":
        analysis_profile = AnalysisProfile.create_dataflow_optimized_profile()
    elif profile == "legacy_compatible":
        analysis_profile = AnalysisProfile.create_legacy_compatible_profile()
    else:
        # Default to comprehensive
        analysis_profile = AnalysisProfile.create_comprehensive_profile()
    
    interface_config = custom_interface_config or analysis_profile.interface_config
    pragma_config = custom_pragma_config or analysis_profile.pragma_config
    
    # Apply pipeline config overrides
    if pipeline_config:
        _apply_pipeline_config_overrides(interface_config, pragma_config, pipeline_config)
    
    return interface_config, pragma_config


def _apply_pipeline_config_overrides(
    interface_config: InterfaceAnalysisConfig,
    pragma_config: PragmaAnalysisConfig,
    pipeline_config: PipelineConfig
) -> None:
    """Apply pipeline configuration overrides to analysis configs."""
    # Apply dataflow settings
    if not pipeline_config.is_dataflow_enabled():
        interface_config.enable_dataflow_conversion = False
        pragma_config.enable_dataflow_constraints = False
        pragma_config.enable_parallelism_inference = False
    
    # Apply analysis settings
    if not pipeline_config.analysis.analyze_dataflow_interfaces:
        interface_config.enable_dataflow_conversion = False
    
    if not pipeline_config.analysis.validate_pragma_compatibility:
        pragma_config.validate_signal_references = False
        pragma_config.validate_parameter_references = False
    
    # Apply validation settings
    if not pipeline_config.validation.validate_interface_constraints:
        interface_config.validate_signal_directions = False
        interface_config.validate_signal_widths = False
        interface_config.validate_naming_consistency = False


def get_available_profiles() -> List[str]:
    """Get list of available analysis profiles."""
    return ["fast", "comprehensive", "dataflow_optimized", "legacy_compatible"]


def get_profile_description(profile: str) -> str:
    """Get description for an analysis profile."""
    descriptions = {
        "fast": "Fast analysis with basic pattern matching",
        "comprehensive": "Comprehensive analysis with full validation",
        "dataflow_optimized": "Optimized for dataflow modeling and constraint generation",
        "legacy_compatible": "Compatible with legacy RTL and pragma formats"
    }
    return descriptions.get(profile, "Unknown profile")


def create_custom_profile(
    name: str,
    description: str,
    interface_config: InterfaceAnalysisConfig,
    pragma_config: PragmaAnalysisConfig
) -> AnalysisProfile:
    """Create a custom analysis profile."""
    return AnalysisProfile(
        name=name,
        description=description,
        interface_config=interface_config,
        pragma_config=pragma_config
    )