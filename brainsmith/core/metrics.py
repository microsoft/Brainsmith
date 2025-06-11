"""
Essential DSE Metrics for BrainSmith Core

Focuses on critical metrics needed for design space exploration decisions.
Removes research-oriented complexity in favor of practical DSE feedback.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import json


@dataclass
class PerformanceMetrics:
    """Core performance metrics for DSE decisions."""
    
    throughput_ops_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    clock_frequency_mhz: Optional[float] = None
    target_fps: Optional[float] = None
    achieved_fps: Optional[float] = None
    
    def get_fps_efficiency(self) -> Optional[float]:
        """Calculate FPS efficiency ratio."""
        if self.achieved_fps and self.target_fps and self.target_fps > 0:
            return self.achieved_fps / self.target_fps
        return None


@dataclass
class ResourceMetrics:
    """FPGA resource utilization for DSE decisions."""
    
    # Core FPGA resources
    lut_utilization_percent: Optional[float] = None
    dsp_utilization_percent: Optional[float] = None
    bram_utilization_percent: Optional[float] = None
    estimated_power_w: Optional[float] = None
    
    def get_resource_efficiency(self) -> Optional[float]:
        """Calculate overall resource utilization score."""
        utilizations = [u for u in [
            self.lut_utilization_percent,
            self.dsp_utilization_percent, 
            self.bram_utilization_percent
        ] if u is not None]
        
        if utilizations:
            return sum(utilizations) / len(utilizations)
        return None


@dataclass
class DSEMetrics:
    """Essential metrics for design space exploration feedback."""
    
    # Core metric categories
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    
    # Build status
    build_success: bool = False
    build_time_seconds: float = 0.0
    
    # DSE context
    design_point_id: str = ""
    configuration: Dict[str, Any] = field(default_factory=dict)
    
    def get_optimization_score(self) -> float:
        """Calculate combined optimization score for DSE ranking."""
        score = 0.0
        weights = 0.0
        
        # Performance contribution (weight: 0.4)
        if self.performance.throughput_ops_sec:
            score += 0.4 * min(self.performance.throughput_ops_sec / 1000.0, 1.0)
            weights += 0.4
        
        # Resource efficiency contribution (weight: 0.4)
        resource_eff = self.resources.get_resource_efficiency()
        if resource_eff:
            score += 0.4 * (1.0 - resource_eff / 100.0)  # Lower utilization is better
            weights += 0.4
        
        # Success bonus (weight: 0.2)
        if self.build_success:
            score += 0.2
            weights += 0.2
        
        return score / weights if weights > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'design_point_id': self.design_point_id,
            'build_success': self.build_success,
            'build_time_seconds': self.build_time_seconds,
            'configuration': self.configuration,
            'performance': {
                'throughput_ops_sec': self.performance.throughput_ops_sec,
                'latency_ms': self.performance.latency_ms,
                'clock_frequency_mhz': self.performance.clock_frequency_mhz,
                'fps_efficiency': self.performance.get_fps_efficiency()
            },
            'resources': {
                'lut_utilization_percent': self.resources.lut_utilization_percent,
                'dsp_utilization_percent': self.resources.dsp_utilization_percent,
                'bram_utilization_percent': self.resources.bram_utilization_percent,
                'estimated_power_w': self.resources.estimated_power_w,
                'resource_efficiency': self.resources.get_resource_efficiency()
            },
            'optimization_score': self.get_optimization_score()
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DSEMetrics':
        """Create from dictionary."""
        metrics = cls()
        metrics.design_point_id = data.get('design_point_id', '')
        metrics.build_success = data.get('build_success', False)
        metrics.build_time_seconds = data.get('build_time_seconds', 0.0)
        metrics.configuration = data.get('configuration', {})
        
        # Reconstruct performance metrics
        perf_data = data.get('performance', {})
        metrics.performance = PerformanceMetrics(
            throughput_ops_sec=perf_data.get('throughput_ops_sec'),
            latency_ms=perf_data.get('latency_ms'),
            clock_frequency_mhz=perf_data.get('clock_frequency_mhz')
        )
        
        # Reconstruct resource metrics
        resource_data = data.get('resources', {})
        metrics.resources = ResourceMetrics(
            lut_utilization_percent=resource_data.get('lut_utilization_percent'),
            dsp_utilization_percent=resource_data.get('dsp_utilization_percent'),
            bram_utilization_percent=resource_data.get('bram_utilization_percent'),
            estimated_power_w=resource_data.get('estimated_power_w')
        )
        
        return metrics


def create_metrics(design_point_id: str = "", configuration: Dict[str, Any] = None) -> DSEMetrics:
    """Create new DSEMetrics instance with basic setup."""
    return DSEMetrics(
        design_point_id=design_point_id,
        configuration=configuration or {}
    )


def compare_metrics(metrics_list: list[DSEMetrics]) -> DSEMetrics:
    """Find best metrics based on optimization score."""
    if not metrics_list:
        return create_metrics()
    
    # Filter successful builds
    successful = [m for m in metrics_list if m.build_success]
    if not successful:
        return metrics_list[0]  # Return first if none successful
    
    # Return best by optimization score
    return max(successful, key=lambda m: m.get_optimization_score())