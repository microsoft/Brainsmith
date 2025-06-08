"""
Comprehensive metrics collection system for Brainsmith platform.

This module provides structured metric collection for design space exploration
and research enablement.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List
from datetime import datetime
import numpy as np
import json


@dataclass
class PerformanceMetrics:
    """Performance characteristics of the compiled accelerator."""
    
    throughput_ops_sec: Optional[float] = None
    throughput_inferences_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    efficiency_ops_per_joule: Optional[float] = None
    clock_frequency_mhz: Optional[float] = None
    target_fps: Optional[float] = None
    achieved_fps: Optional[float] = None
    fps_efficiency: Optional[float] = None  # achieved_fps / target_fps
    
    def __post_init__(self):
        """Calculate derived metrics."""
        if self.achieved_fps is not None and self.target_fps is not None and self.target_fps > 0:
            self.fps_efficiency = self.achieved_fps / self.target_fps
            
        # Convert between throughput representations if possible
        if self.throughput_inferences_sec is not None and self.throughput_ops_sec is None:
            # Assume this will be filled in by model-specific calculations
            pass


@dataclass
class ResourceMetrics:
    """FPGA resource utilization metrics."""
    
    # LUT resources
    lut_count: Optional[int] = None
    lut_available: Optional[int] = None
    lut_utilization_percent: Optional[float] = None
    
    # DSP resources
    dsp_count: Optional[int] = None
    dsp_available: Optional[int] = None
    dsp_utilization_percent: Optional[float] = None
    
    # BRAM resources
    bram_18k_count: Optional[int] = None
    bram_18k_available: Optional[int] = None
    bram_utilization_percent: Optional[float] = None
    
    # URAM resources (for larger FPGAs)
    uram_count: Optional[int] = None
    uram_available: Optional[int] = None
    uram_utilization_percent: Optional[float] = None
    
    # Power estimates
    estimated_power_w: Optional[float] = None
    static_power_w: Optional[float] = None
    dynamic_power_w: Optional[float] = None
    
    # Derived metrics
    resource_efficiency: Optional[float] = None  # ops/resource
    power_efficiency: Optional[float] = None     # ops/watt
    
    def __post_init__(self):
        """Calculate utilization percentages and derived metrics."""
        # Calculate utilization percentages if counts and available are present
        if self.lut_count is not None and self.lut_available is not None and self.lut_available > 0:
            self.lut_utilization_percent = (self.lut_count / self.lut_available) * 100
            
        if self.dsp_count is not None and self.dsp_available is not None and self.dsp_available > 0:
            self.dsp_utilization_percent = (self.dsp_count / self.dsp_available) * 100
            
        if self.bram_18k_count is not None and self.bram_18k_available is not None and self.bram_18k_available > 0:
            self.bram_utilization_percent = (self.bram_18k_count / self.bram_18k_available) * 100
    
    def get_total_resource_score(self) -> Optional[float]:
        """Calculate normalized total resource usage score."""
        utilizations = []
        if self.lut_utilization_percent is not None:
            utilizations.append(self.lut_utilization_percent / 100.0)
        if self.dsp_utilization_percent is not None:
            utilizations.append(self.dsp_utilization_percent / 100.0)
        if self.bram_utilization_percent is not None:
            utilizations.append(self.bram_utilization_percent / 100.0)
            
        if utilizations:
            return sum(utilizations) / len(utilizations)
        return None


@dataclass
class QualityMetrics:
    """Model quality and accuracy metrics."""
    
    model_accuracy: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    
    # Verification metrics
    verification_passed: Optional[bool] = None
    verification_tolerance: Optional[float] = None
    max_error: Optional[float] = None
    mean_error: Optional[float] = None
    
    # Bit width analysis
    activation_bit_widths: Optional[Dict[str, int]] = None
    weight_bit_widths: Optional[Dict[str, int]] = None
    
    def get_quality_score(self) -> Optional[float]:
        """Calculate overall quality score."""
        if self.model_accuracy is not None:
            return self.model_accuracy
        elif self.f1_score is not None:
            return self.f1_score
        return None


@dataclass
class BuildMetrics:
    """Build process metrics and timing information."""
    
    build_time_total: float = 0.0
    build_time_preprocessing: Optional[float] = None
    build_time_finn: Optional[float] = None
    build_time_postprocessing: Optional[float] = None
    
    build_success: bool = False
    finn_steps_completed: int = 0
    finn_steps_total: Optional[int] = None
    
    # Error tracking
    error_count: int = 0
    warning_count: int = 0
    
    # Build environment
    finn_version: Optional[str] = None
    brainsmith_version: Optional[str] = None
    build_timestamp: datetime = field(default_factory=datetime.now)
    
    # Step timing (for detailed analysis)
    step_timings: Dict[str, float] = field(default_factory=dict)
    
    def add_step_timing(self, step_name: str, duration: float):
        """Add timing for a specific build step."""
        self.step_timings[step_name] = duration
    
    def get_build_efficiency(self) -> Optional[float]:
        """Calculate build time efficiency (steps per minute)."""
        if self.build_time_total > 0 and self.finn_steps_completed > 0:
            return (self.finn_steps_completed / self.build_time_total) * 60.0
        return None


@dataclass
class BrainsmithMetrics:
    """Comprehensive metrics collection for DSE research."""
    
    # Core metric categories
    performance: PerformanceMetrics = field(default_factory=PerformanceMetrics)
    resources: ResourceMetrics = field(default_factory=ResourceMetrics)
    quality: QualityMetrics = field(default_factory=QualityMetrics)
    build_info: BuildMetrics = field(default_factory=BuildMetrics)
    
    # Build identification
    build_id: str = ""
    configuration_hash: str = ""
    
    # Design space context (will be populated by DesignPoint)
    design_point_data: Dict[str, Any] = field(default_factory=dict)
    constraint_data: Dict[str, Any] = field(default_factory=dict)
    
    # Raw data for research
    raw_reports: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: List[Dict[str, Any]] = field(default_factory=list)
    
    # Custom metrics (extensible)
    custom_metrics: Dict[str, float] = field(default_factory=dict)
    
    def add_custom_metric(self, name: str, value: float, description: str = ""):
        """Add a custom metric for research purposes."""
        self.custom_metrics[name] = value
        if description:
            # Store description in metadata if needed
            pass
    
    def add_raw_report(self, report_name: str, report_data: Any):
        """Add raw report data for research analysis."""
        self.raw_reports[report_name] = report_data
    
    def add_intermediate_result(self, step_name: str, result_data: Dict[str, Any]):
        """Add intermediate build result for analysis."""
        self.intermediate_results.append({
            'step_name': step_name,
            'timestamp': datetime.now().isoformat(),
            'data': result_data
        })
    
    def to_research_dataset(self) -> Dict[str, Any]:
        """Export comprehensive data for external DSE research."""
        return {
            # Identifiers
            'build_id': self.build_id,
            'configuration_hash': self.configuration_hash,
            'timestamp': self.build_info.build_timestamp.isoformat(),
            
            # Performance features
            'throughput_ops_sec': self.performance.throughput_ops_sec,
            'latency_ms': self.performance.latency_ms,
            'clock_frequency_mhz': self.performance.clock_frequency_mhz,
            'fps_efficiency': self.performance.fps_efficiency,
            
            # Resource features
            'lut_count': self.resources.lut_count,
            'lut_utilization_percent': self.resources.lut_utilization_percent,
            'dsp_count': self.resources.dsp_count,
            'dsp_utilization_percent': self.resources.dsp_utilization_percent,
            'bram_count': self.resources.bram_18k_count,
            'estimated_power_w': self.resources.estimated_power_w,
            'total_resource_score': self.resources.get_total_resource_score(),
            
            # Quality features
            'model_accuracy': self.quality.model_accuracy,
            'verification_passed': self.quality.verification_passed,
            'quality_score': self.quality.get_quality_score(),
            
            # Build features
            'build_time_total': self.build_info.build_time_total,
            'build_success': self.build_info.build_success,
            'build_efficiency': self.build_info.get_build_efficiency(),
            
            # Design space context
            'design_point': self.design_point_data,
            'constraints': self.constraint_data,
            
            # Custom metrics
            'custom_metrics': self.custom_metrics,
        }
    
    def get_optimization_features(self) -> np.ndarray:
        """Extract feature vector for ML-based DSE algorithms."""
        features = []
        
        # Performance features
        features.extend([
            self.performance.throughput_ops_sec or 0.0,
            self.performance.latency_ms or 0.0,
            self.performance.clock_frequency_mhz or 0.0,
            self.performance.fps_efficiency or 0.0,
        ])
        
        # Resource features
        features.extend([
            self.resources.lut_utilization_percent or 0.0,
            self.resources.dsp_utilization_percent or 0.0,
            self.resources.bram_utilization_percent or 0.0,
            self.resources.estimated_power_w or 0.0,
        ])
        
        # Quality features
        features.extend([
            self.quality.get_quality_score() or 0.0,
            1.0 if self.quality.verification_passed else 0.0,
        ])
        
        # Build features
        features.extend([
            self.build_info.build_time_total,
            1.0 if self.build_info.build_success else 0.0,
        ])
        
        return np.array(features, dtype=np.float32)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'build_id': self.build_id,
            'configuration_hash': self.configuration_hash,
            'performance': self.performance.__dict__,
            'resources': self.resources.__dict__,
            'quality': self.quality.__dict__,
            'build_info': {
                **self.build_info.__dict__,
                'build_timestamp': self.build_info.build_timestamp.isoformat()
            },
            'design_point_data': self.design_point_data,
            'constraint_data': self.constraint_data,
            'custom_metrics': self.custom_metrics,
            'raw_reports': self.raw_reports,
            'intermediate_results': self.intermediate_results,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BrainsmithMetrics':
        """Create from dictionary."""
        metrics = cls()
        metrics.build_id = data.get('build_id', '')
        metrics.configuration_hash = data.get('configuration_hash', '')
        
        # Reconstruct nested objects
        if 'performance' in data:
            metrics.performance = PerformanceMetrics(**data['performance'])
        if 'resources' in data:
            metrics.resources = ResourceMetrics(**data['resources'])
        if 'quality' in data:
            metrics.quality = QualityMetrics(**data['quality'])
        if 'build_info' in data:
            build_data = data['build_info'].copy()
            if 'build_timestamp' in build_data:
                build_data['build_timestamp'] = datetime.fromisoformat(build_data['build_timestamp'])
            metrics.build_info = BuildMetrics(**build_data)
            
        # Other fields
        metrics.design_point_data = data.get('design_point_data', {})
        metrics.constraint_data = data.get('constraint_data', {})
        metrics.custom_metrics = data.get('custom_metrics', {})
        metrics.raw_reports = data.get('raw_reports', {})
        metrics.intermediate_results = data.get('intermediate_results', [])
        
        return metrics


class MetricsCollector:
    """Helper class for collecting metrics during builds."""
    
    def __init__(self):
        self.metrics = BrainsmithMetrics()
        self._start_time: Optional[float] = None
        self._step_start_times: Dict[str, float] = {}
    
    def start_build(self, build_id: str = ""):
        """Start timing the build process."""
        self._start_time = time.time()
        self.metrics.build_id = build_id or f"build_{int(time.time())}"
        self.metrics.build_info.build_timestamp = datetime.now()
    
    def end_build(self, success: bool = True):
        """End timing the build process."""
        if self._start_time is not None:
            self.metrics.build_info.build_time_total = time.time() - self._start_time
        self.metrics.build_info.build_success = success
    
    def start_step(self, step_name: str):
        """Start timing a build step."""
        self._step_start_times[step_name] = time.time()
    
    def end_step(self, step_name: str):
        """End timing a build step."""
        if step_name in self._step_start_times:
            duration = time.time() - self._step_start_times[step_name]
            self.metrics.build_info.add_step_timing(step_name, duration)
            del self._step_start_times[step_name]
            self.metrics.build_info.finn_steps_completed += 1
    
    def add_error(self):
        """Increment error count."""
        self.metrics.build_info.error_count += 1
    
    def add_warning(self):
        """Increment warning count."""
        self.metrics.build_info.warning_count += 1
    
    def set_finn_steps_total(self, total: int):
        """Set total number of FINN steps."""
        self.metrics.build_info.finn_steps_total = total
    
    def get_metrics(self) -> BrainsmithMetrics:
        """Get the collected metrics."""
        return self.metrics