"""
Enhanced configuration for unified HWKG.

Based on hw_kernel_gen_simple configuration with additional complexity level controls
following HWKG Axiom 8: Configuration Layering.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .errors import ConfigurationError


@dataclass
class UnifiedConfig:
    """
    Configuration for unified HWKG with complexity levels.
    
    Based on hw_kernel_gen_simple Config class with enhancements for
    advanced pragma processing and multi-phase execution support.
    Follows HWKG Axiom 8: Configuration Layering precedence.
    """
    # Core configuration (from simple system)
    rtl_file: Path
    compiler_data_file: Path
    output_dir: Path
    template_dir: Optional[Path] = None
    debug: bool = False
    
    # Complexity level controls
    advanced_pragmas: bool = False
    multi_phase_execution: bool = False
    stop_after: Optional[str] = None
    
    def __post_init__(self):
        """Validate configuration following robust validation patterns."""
        if not self.rtl_file.exists():
            raise ConfigurationError(f"RTL file not found: {self.rtl_file}")
        if not self.compiler_data_file.exists():
            raise ConfigurationError(f"Compiler data file not found: {self.compiler_data_file}")
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate template directory if provided
        if self.template_dir and not self.template_dir.exists():
            raise ConfigurationError(f"Template directory not found: {self.template_dir}")
        
        # Validate multi-phase options
        if self.stop_after and not self.multi_phase_execution:
            raise ConfigurationError("--stop-after requires --multi-phase flag")
        
        # Validate stop-after values
        valid_stop_points = [
            'parse_rtl', 'parse_compiler_data', 'build_dataflow_model',
            'generate_hw_custom_op', 'generate_rtl_backend', 'generate_test_suite'
        ]
        if self.stop_after and self.stop_after not in valid_stop_points:
            raise ConfigurationError(f"Invalid stop point: {self.stop_after}. Valid options: {valid_stop_points}")
    
    @property
    def complexity_level(self) -> str:
        """
        Determine complexity level from configuration flags.
        
        Following the simple-by-default, powerful-when-needed philosophy.
        """
        if self.advanced_pragmas and self.multi_phase_execution:
            return "expert"
        elif self.advanced_pragmas:
            return "advanced"
        else:
            return "simple"
    
    @property
    def uses_sophisticated_features(self) -> bool:
        """Check if any sophisticated features are enabled."""
        return self.advanced_pragmas or self.multi_phase_execution
    
    @property
    def configuration_summary(self) -> dict:
        """
        Get comprehensive configuration summary for logging and debugging.
        
        Useful for expert mode debugging and configuration optimization.
        """
        return {
            'complexity_level': self.complexity_level,
            'feature_flags': {
                'advanced_pragmas': self.advanced_pragmas,
                'multi_phase_execution': self.multi_phase_execution,
                'debug_enabled': self.debug
            },
            'file_paths': {
                'rtl_file': str(self.rtl_file),
                'compiler_data_file': str(self.compiler_data_file),
                'output_dir': str(self.output_dir),
                'template_dir': str(self.template_dir) if self.template_dir else None
            },
            'execution_options': {
                'stop_after': self.stop_after,
                'uses_custom_templates': self.template_dir is not None,
                'sophisticated_features_enabled': self.uses_sophisticated_features
            },
            'optimization_recommendations': self._get_optimization_recommendations()
        }
    
    def _get_optimization_recommendations(self) -> list:
        """Get configuration optimization recommendations."""
        recommendations = []
        
        # Recommend advanced pragmas for complex RTL files
        if not self.advanced_pragmas and self._is_complex_rtl_file():
            recommendations.append({
                'type': 'feature_enhancement',
                'suggestion': 'Consider enabling --advanced-pragmas for better BDIM integration',
                'benefit': 'Enhanced chunking strategies and performance optimization'
            })
        
        # Recommend debug mode for first-time users
        if not self.debug and self.complexity_level == 'simple':
            recommendations.append({
                'type': 'usability',
                'suggestion': 'Consider adding --debug for detailed generation information',
                'benefit': 'Better understanding of generation process'
            })
        
        # Recommend multi-phase for debugging complex issues
        if self.advanced_pragmas and not self.multi_phase_execution:
            recommendations.append({
                'type': 'debugging',
                'suggestion': 'Consider --multi-phase for step-by-step debugging',
                'benefit': 'Isolated phase execution for troubleshooting'
            })
        
        return recommendations
    
    def _is_complex_rtl_file(self) -> bool:
        """Heuristic to determine if RTL file might benefit from advanced features."""
        try:
            rtl_content = self.rtl_file.read_text(encoding='utf-8')
            
            # Check for BDIM pragmas
            if '@brainsmith' in rtl_content and 'BDIM' in rtl_content:
                return True
            
            # Check for multiple AXI interfaces
            axi_interface_count = rtl_content.count('_axis_') + rtl_content.count('_axilite_')
            if axi_interface_count > 2:
                return True
            
            # Check for complex parameter structures
            if rtl_content.count('parameter') > 5:
                return True
            
            return False
        except:
            return False  # Safe default
    
    @property
    def performance_profile(self) -> dict:
        """
        Get performance profile recommendations based on configuration.
        
        Following Interface-Wise Dataflow Axiom optimization principles.
        """
        profile = {
            'expected_generation_time': 'fast',
            'memory_usage': 'low',
            'output_quality': 'standard',
            'optimization_level': 'basic'
        }
        
        if self.advanced_pragmas:
            profile.update({
                'expected_generation_time': 'medium',
                'memory_usage': 'medium',
                'output_quality': 'enhanced',
                'optimization_level': 'advanced'
            })
        
        if self.multi_phase_execution:
            profile.update({
                'expected_generation_time': 'slow',
                'memory_usage': 'high',
                'output_quality': 'comprehensive',
                'optimization_level': 'expert'
            })
        
        return profile
    
    def validate_for_target_use_case(self, use_case: str) -> list:
        """
        Validate configuration for specific use cases.
        
        Returns list of warnings/recommendations for the target use case.
        """
        warnings = []
        
        if use_case == 'production':
            if self.debug:
                warnings.append("Debug mode may impact performance in production")
            if self.multi_phase_execution:
                warnings.append("Multi-phase execution not recommended for production automation")
        
        elif use_case == 'development':
            if not self.debug:
                warnings.append("Debug mode recommended for development workflow")
            if not self.advanced_pragmas and self._is_complex_rtl_file():
                warnings.append("Advanced pragmas recommended for development of complex RTL")
        
        elif use_case == 'research':
            if not self.advanced_pragmas:
                warnings.append("Advanced pragmas recommended for research applications")
            if not self.multi_phase_execution:
                warnings.append("Multi-phase execution useful for research analysis")
        
        elif use_case == 'ci_cd':
            if self.multi_phase_execution:
                warnings.append("Multi-phase execution may slow CI/CD pipelines")
            if not self.debug:
                warnings.append("Debug output helpful for CI/CD troubleshooting")
        
        return warnings
    
    def get_resource_estimates(self) -> dict:
        """
        Estimate resource requirements based on configuration.
        
        Helps users plan computation resources and execution time.
        """
        estimates = {
            'cpu_usage': 'low',
            'memory_usage': 'low',
            'disk_io': 'low',
            'network_io': 'none',
            'estimated_time_seconds': 5
        }
        
        # Adjust based on complexity level
        if self.complexity_level == 'advanced':
            estimates.update({
                'cpu_usage': 'medium',
                'memory_usage': 'medium',
                'estimated_time_seconds': 15
            })
        elif self.complexity_level == 'expert':
            estimates.update({
                'cpu_usage': 'high',
                'memory_usage': 'high',
                'disk_io': 'medium',
                'estimated_time_seconds': 30
            })
        
        # Adjust for file complexity
        if self._is_complex_rtl_file():
            estimates['estimated_time_seconds'] *= 2
            if estimates['cpu_usage'] == 'low':
                estimates['cpu_usage'] = 'medium'
        
        return estimates
    
    @classmethod
    def from_args(cls, args) -> 'UnifiedConfig':
        """Create config from CLI arguments."""
        return cls(
            rtl_file=Path(args.rtl_file),
            compiler_data_file=Path(args.compiler_data),
            output_dir=Path(args.output),
            template_dir=Path(args.template_dir) if hasattr(args, 'template_dir') and args.template_dir else None,
            debug=args.debug if hasattr(args, 'debug') else False,
            advanced_pragmas=args.advanced_pragmas if hasattr(args, 'advanced_pragmas') else False,
            multi_phase_execution=args.multi_phase if hasattr(args, 'multi_phase') else False,
            stop_after=args.stop_after if hasattr(args, 'stop_after') else None
        )
    
    @classmethod
    def for_use_case(cls, rtl_file: Path, compiler_data_file: Path, output_dir: Path, 
                     use_case: str, **overrides) -> 'UnifiedConfig':
        """
        Create configuration optimized for specific use cases.
        
        Provides pre-configured settings for common workflows while allowing overrides.
        """
        # Base configuration for all use cases
        base_config = {
            'rtl_file': rtl_file,
            'compiler_data_file': compiler_data_file,
            'output_dir': output_dir,
            'template_dir': None,
            'debug': False,
            'advanced_pragmas': False,
            'multi_phase_execution': False,
            'stop_after': None
        }
        
        # Use case specific optimizations
        if use_case == 'production':
            # Optimized for automated production pipelines
            base_config.update({
                'debug': False,
                'advanced_pragmas': True,  # Better output quality
                'multi_phase_execution': False  # Faster execution
            })
        
        elif use_case == 'development':
            # Optimized for interactive development
            base_config.update({
                'debug': True,  # Detailed feedback
                'advanced_pragmas': True,  # Better analysis
                'multi_phase_execution': False  # Reasonable speed
            })
        
        elif use_case == 'research':
            # Optimized for research and analysis
            base_config.update({
                'debug': True,  # Comprehensive information
                'advanced_pragmas': True,  # Full sophistication
                'multi_phase_execution': True  # Step-by-step analysis
            })
        
        elif use_case == 'ci_cd':
            # Optimized for CI/CD pipelines
            base_config.update({
                'debug': True,  # Troubleshooting information
                'advanced_pragmas': False,  # Faster execution
                'multi_phase_execution': False  # Pipeline efficiency
            })
        
        elif use_case == 'quick_test':
            # Optimized for quick validation
            base_config.update({
                'debug': False,  # Minimal output
                'advanced_pragmas': False,  # Fast execution
                'multi_phase_execution': False  # Quick turnaround
            })
        
        elif use_case == 'debugging':
            # Optimized for troubleshooting issues
            base_config.update({
                'debug': True,  # Maximum information
                'advanced_pragmas': True,  # Full analysis
                'multi_phase_execution': True,  # Step-by-step debugging
                'stop_after': None  # Complete execution by default
            })
        
        else:
            raise ConfigurationError(f"Unknown use case: {use_case}. Valid options: production, development, research, ci_cd, quick_test, debugging")
        
        # Apply user overrides
        base_config.update(overrides)
        
        return cls(**base_config)
    
    def export_config(self, file_path: Path) -> None:
        """
        Export configuration to file for reuse.
        
        Useful for standardizing configurations across teams.
        """
        import json
        
        config_data = {
            'version': '1.0',
            'unified_hwkg_config': {
                'complexity_level': self.complexity_level,
                'feature_flags': {
                    'advanced_pragmas': self.advanced_pragmas,
                    'multi_phase_execution': self.multi_phase_execution,
                    'debug_enabled': self.debug
                },
                'execution_options': {
                    'stop_after': self.stop_after,
                    'template_dir': str(self.template_dir) if self.template_dir else None
                },
                'metadata': {
                    'exported_at': self._get_timestamp(),
                    'performance_profile': self.performance_profile,
                    'optimization_recommendations': self._get_optimization_recommendations(),
                    'resource_estimates': self.get_resource_estimates()
                }
            }
        }
        
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'w') as f:
            json.dump(config_data, f, indent=2)
    
    @classmethod
    def import_config(cls, config_file: Path, rtl_file: Path, compiler_data_file: Path, 
                     output_dir: Path) -> 'UnifiedConfig':
        """
        Import configuration from file and apply to new files.
        
        Allows reusing proven configurations across different RTL files.
        """
        import json
        
        with open(config_file, 'r') as f:
            config_data = json.load(f)
        
        if 'unified_hwkg_config' not in config_data:
            raise ConfigurationError(f"Invalid configuration file format: {config_file}")
        
        hwkg_config = config_data['unified_hwkg_config']
        feature_flags = hwkg_config.get('feature_flags', {})
        execution_options = hwkg_config.get('execution_options', {})
        
        return cls(
            rtl_file=rtl_file,
            compiler_data_file=compiler_data_file,
            output_dir=output_dir,
            template_dir=Path(execution_options['template_dir']) if execution_options.get('template_dir') else None,
            debug=feature_flags.get('debug_enabled', False),
            advanced_pragmas=feature_flags.get('advanced_pragmas', False),
            multi_phase_execution=feature_flags.get('multi_phase_execution', False),
            stop_after=execution_options.get('stop_after')
        )
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for export metadata."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_usage_analytics(self) -> dict:
        """
        Get usage analytics for configuration optimization.
        
        Helps understand which features are being used and their effectiveness.
        """
        return {
            'complexity_distribution': {
                'simple': self.complexity_level == 'simple',
                'advanced': self.complexity_level == 'advanced',
                'expert': self.complexity_level == 'expert'
            },
            'feature_adoption': {
                'advanced_pragmas_used': self.advanced_pragmas,
                'multi_phase_used': self.multi_phase_execution,
                'debug_used': self.debug,
                'custom_templates_used': self.template_dir is not None
            },
            'configuration_efficiency': {
                'optimization_opportunities': len(self._get_optimization_recommendations()),
                'estimated_performance': self.performance_profile['optimization_level'],
                'resource_efficiency': self._calculate_resource_efficiency()
            }
        }
    
    def _calculate_resource_efficiency(self) -> str:
        """Calculate resource efficiency based on configuration choices."""
        estimates = self.get_resource_estimates()
        
        # Simple heuristic for resource efficiency
        if estimates['cpu_usage'] == 'low' and estimates['memory_usage'] == 'low':
            return 'high'
        elif estimates['cpu_usage'] == 'high' or estimates['memory_usage'] == 'high':
            return 'low'
        else:
            return 'medium'