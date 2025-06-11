"""
Essential FINN Types for Simplified Interface

This module contains only the essential data types needed for the simplified
FINN interface, removing enterprise complexity while maintaining functionality.
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class FINNConfig:
    """Simplified FINN configuration."""
    
    # Core FINN parameters
    target_device: str = "U250"
    target_fps: int = 1000
    clock_period: float = 3.33
    shell_flow: str = "vivado_zynq"
    output_dir: str = "./output"
    
    # Build configuration
    mvau_wwidth_max: int = 36
    enable_synthesis: bool = True
    enable_bitstream: bool = False
    
    def to_core_dict(self) -> Dict[str, Any]:
        """Convert to core interface format."""
        return {
            'target_device': self.target_device,
            'target_fps': self.target_fps,
            'clock_period': self.clock_period,
            'shell_flow': self.shell_flow,
            'mvau_wwidth_max': self.mvau_wwidth_max,
            'enable_synthesis': self.enable_synthesis,
            'enable_bitstream': self.enable_bitstream
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'target_device': self.target_device,
            'target_fps': self.target_fps,
            'clock_period': self.clock_period,
            'shell_flow': self.shell_flow,
            'output_dir': self.output_dir,
            'mvau_wwidth_max': self.mvau_wwidth_max,
            'enable_synthesis': self.enable_synthesis,
            'enable_bitstream': self.enable_bitstream
        }


@dataclass
class FINNResult:
    """Simplified FINN build result."""
    
    # Core result information
    success: bool
    model_path: str
    output_dir: str
    
    # Build metrics
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, int] = field(default_factory=dict)
    build_time: float = 0.0
    
    # Error handling
    error_message: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    
    # Build artifacts
    rtl_files: list[str] = field(default_factory=list)
    hls_files: list[str] = field(default_factory=list)
    
    @classmethod
    def from_core_result(cls, core_result: Dict[str, Any]) -> 'FINNResult':
        """Convert from core interface result."""
        return cls(
            success=core_result.get('success', False),
            model_path=core_result.get('model_path', ''),
            output_dir=core_result.get('output_dir', ''),
            performance_metrics=core_result.get('performance_metrics', {}),
            resource_usage=core_result.get('resource_usage', {}),
            build_time=core_result.get('build_time', 0.0),
            error_message=core_result.get('error'),
            warnings=core_result.get('warnings', []),
            rtl_files=core_result.get('rtl_files', []),
            hls_files=core_result.get('hls_files', [])
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'success': self.success,
            'model_path': self.model_path,
            'output_dir': self.output_dir,
            'performance_metrics': self.performance_metrics,
            'resource_usage': self.resource_usage,
            'build_time': self.build_time,
            'error_message': self.error_message,
            'warnings': self.warnings,
            'rtl_files': self.rtl_files,
            'hls_files': self.hls_files
        }
    
    @property
    def throughput_fps(self) -> float:
        """Get throughput in FPS."""
        return self.performance_metrics.get('throughput_fps', 0.0)
    
    @property
    def latency_cycles(self) -> int:
        """Get latency in cycles."""
        return int(self.performance_metrics.get('latency_cycles', 0))
    
    @property
    def lut_count(self) -> int:
        """Get LUT count."""
        return self.resource_usage.get('lut_count', 0)
    
    @property
    def dsp_count(self) -> int:
        """Get DSP count."""
        return self.resource_usage.get('dsp_count', 0)


@dataclass
class FINNHooksConfig:
    """4-hooks preparation configuration for future FINN interface."""
    
    # Future hook enablement flags
    preprocessing_enabled: bool = True
    transformation_enabled: bool = True  
    optimization_enabled: bool = True
    generation_enabled: bool = True
    
    # Hook-specific configurations
    preprocessing_params: Dict[str, Any] = field(default_factory=dict)
    transformation_params: Dict[str, Any] = field(default_factory=dict)
    optimization_params: Dict[str, Any] = field(default_factory=dict)
    generation_params: Dict[str, Any] = field(default_factory=dict)
    
    def prepare_config(self, design_point: Dict[str, Any]) -> Dict[str, Any]:
        """
        Prepare configuration for future 4-hooks interface.
        
        This method creates a structured configuration that will be compatible
        with FINN's future 4-hooks interface when it becomes available.
        """
        return {
            'preprocessing': {
                'enabled': self.preprocessing_enabled,
                'params': {
                    **self.preprocessing_params,
                    **design_point.get('preprocessing', {})
                }
            },
            'transformation': {
                'enabled': self.transformation_enabled,
                'params': {
                    **self.transformation_params,
                    **design_point.get('transforms', {}),
                    **design_point.get('transformation', {})
                }
            },
            'optimization': {
                'enabled': self.optimization_enabled, 
                'params': {
                    **self.optimization_params,
                    **design_point.get('hw_optimization', {}),
                    **design_point.get('optimization', {})
                }
            },
            'generation': {
                'enabled': self.generation_enabled,
                'params': {
                    **self.generation_params,
                    **design_point.get('generation', {}),
                    **design_point.get('codegen', {})
                }
            }
        }
    
    def is_4hooks_ready(self) -> bool:
        """Check if ready for 4-hooks interface."""
        # Always False until FINN implements the 4-hooks interface
        return False
    
    def get_enabled_hooks(self) -> list[str]:
        """Get list of enabled hooks."""
        enabled = []
        if self.preprocessing_enabled:
            enabled.append('preprocessing')
        if self.transformation_enabled:
            enabled.append('transformation')
        if self.optimization_enabled:
            enabled.append('optimization')
        if self.generation_enabled:
            enabled.append('generation')
        return enabled
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            'preprocessing_enabled': self.preprocessing_enabled,
            'transformation_enabled': self.transformation_enabled,
            'optimization_enabled': self.optimization_enabled,
            'generation_enabled': self.generation_enabled,
            'preprocessing_params': self.preprocessing_params,
            'transformation_params': self.transformation_params,
            'optimization_params': self.optimization_params,
            'generation_params': self.generation_params
        }