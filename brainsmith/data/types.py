"""
Simple Data Types - North Star Aligned

Essential data structures for FPGA build results and analysis.
Consolidated from metrics and analysis modules, eliminating enterprise complexity.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
import time
import json


@dataclass
class PerformanceData:
    """
    Simple container for FPGA performance metrics.
    
    Integrates with:
    - brainsmith.core.api.forge() for build performance
    - brainsmith.dse for parameter sweep metrics
    - brainsmith.finn for accelerator performance
    """
    throughput_ops_sec: Optional[float] = None
    latency_ms: Optional[float] = None
    clock_freq_mhz: Optional[float] = None
    cycles_per_inference: Optional[int] = None
    max_batch_size: Optional[int] = None
    inference_time_ms: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis tools."""
        return {
            'throughput_ops_sec': self.throughput_ops_sec,
            'latency_ms': self.latency_ms,
            'clock_freq_mhz': self.clock_freq_mhz,
            'cycles_per_inference': self.cycles_per_inference,
            'max_batch_size': self.max_batch_size,
            'inference_time_ms': self.inference_time_ms
        }
    
    def get_efficiency_ratio(self) -> Optional[float]:
        """Calculate ops/second per MHz for efficiency comparison."""
        if self.throughput_ops_sec and self.clock_freq_mhz:
            return self.throughput_ops_sec / self.clock_freq_mhz
        return None


@dataclass
class ResourceData:
    """
    Simple container for FPGA resource utilization metrics.
    
    Covers standard FPGA resources for practical optimization.
    """
    lut_utilization_percent: Optional[float] = None
    dsp_utilization_percent: Optional[float] = None
    bram_utilization_percent: Optional[float] = None
    uram_utilization_percent: Optional[float] = None
    ff_utilization_percent: Optional[float] = None
    
    # Absolute counts for detailed analysis
    lut_count: Optional[int] = None
    dsp_count: Optional[int] = None
    bram_count: Optional[int] = None
    uram_count: Optional[int] = None
    ff_count: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'lut_utilization_percent': self.lut_utilization_percent,
            'dsp_utilization_percent': self.dsp_utilization_percent,
            'bram_utilization_percent': self.bram_utilization_percent,
            'uram_utilization_percent': self.uram_utilization_percent,
            'ff_utilization_percent': self.ff_utilization_percent,
            'lut_count': self.lut_count,
            'dsp_count': self.dsp_count,
            'bram_count': self.bram_count,
            'uram_count': self.uram_count,
            'ff_count': self.ff_count
        }
    
    def get_total_utilization(self) -> Optional[float]:
        """Get average utilization across all resource types."""
        utils = [
            self.lut_utilization_percent,
            self.dsp_utilization_percent, 
            self.bram_utilization_percent,
            self.uram_utilization_percent,
            self.ff_utilization_percent
        ]
        valid_utils = [u for u in utils if u is not None]
        if valid_utils:
            return sum(valid_utils) / len(valid_utils)
        return None


@dataclass  
class QualityData:
    """
    Simple container for model quality and accuracy metrics.
    
    Focuses on essential FPGA deployment quality measures.
    """
    accuracy_percent: Optional[float] = None
    precision: Optional[float] = None
    recall: Optional[float] = None
    f1_score: Optional[float] = None
    inference_error_rate: Optional[float] = None
    numerical_precision_bits: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'accuracy_percent': self.accuracy_percent,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'inference_error_rate': self.inference_error_rate,
            'numerical_precision_bits': self.numerical_precision_bits
        }
    
    def get_quality_score(self) -> Optional[float]:
        """Calculate composite quality score."""
        if self.accuracy_percent:
            base_score = self.accuracy_percent / 100.0
            
            # Adjust for other quality factors
            if self.f1_score:
                base_score = (base_score + self.f1_score) / 2.0
            
            if self.inference_error_rate:
                base_score *= (1.0 - self.inference_error_rate)
            
            return min(1.0, max(0.0, base_score))
        return None


@dataclass
class BuildData:
    """
    Simple container for build process metrics.
    
    Tracks build success, timing, and basic information.
    """
    build_success: bool = True
    build_time_seconds: float = 0.0
    synthesis_time_seconds: Optional[float] = None
    place_route_time_seconds: Optional[float] = None
    compilation_warnings: int = 0
    compilation_errors: int = 0
    target_device: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'build_success': self.build_success,
            'build_time_seconds': self.build_time_seconds,
            'synthesis_time_seconds': self.synthesis_time_seconds,
            'place_route_time_seconds': self.place_route_time_seconds,
            'compilation_warnings': self.compilation_warnings,
            'compilation_errors': self.compilation_errors,
            'target_device': self.target_device
        }


@dataclass
class BuildMetrics:
    """
    Complete metrics container for FPGA builds and evaluations.
    
    Unified data structure that integrates with:
    - brainsmith.core.api.forge() results
    - brainsmith.dse parameter sweep results
    - brainsmith.hooks for event logging
    - External analysis tools via to_dict()
    """
    performance: PerformanceData = field(default_factory=PerformanceData)
    resources: ResourceData = field(default_factory=ResourceData)
    quality: QualityData = field(default_factory=QualityData)
    build: BuildData = field(default_factory=BuildData)
    
    # Metadata
    timestamp: float = field(default_factory=time.time)
    model_path: Optional[str] = None
    blueprint_path: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for analysis tools and external export."""
        return {
            'performance': self.performance.to_dict(),
            'resources': self.resources.to_dict(),
            'quality': self.quality.to_dict(),
            'build': self.build.to_dict(),
            'timestamp': self.timestamp,
            'model_path': self.model_path,
            'blueprint_path': self.blueprint_path,
            'parameters': self.parameters,
            'metadata': self.metadata
        }
    
    def to_json(self) -> str:
        """Convert to JSON string for export."""
        return json.dumps(self.to_dict(), indent=2, default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BuildMetrics':
        """Create BuildMetrics from dictionary."""
        return cls(
            performance=PerformanceData(**data.get('performance', {})),
            resources=ResourceData(**data.get('resources', {})),
            quality=QualityData(**data.get('quality', {})),
            build=BuildData(**data.get('build', {})),
            timestamp=data.get('timestamp', time.time()),
            model_path=data.get('model_path'),
            blueprint_path=data.get('blueprint_path'),
            parameters=data.get('parameters', {}),
            metadata=data.get('metadata', {})
        )
    
    def get_efficiency_score(self) -> Optional[float]:
        """Calculate overall efficiency score combining performance and resources."""
        perf_ratio = self.performance.get_efficiency_ratio()
        resource_util = self.resources.get_total_utilization()
        
        if perf_ratio and resource_util:
            # Higher performance per MHz is better, moderate resource utilization is optimal
            normalized_util = min(1.0, resource_util / 70.0)  # Target ~70% utilization
            return perf_ratio * normalized_util
        
        return None
    
    def is_successful(self) -> bool:
        """Check if this represents a successful build."""
        return self.build.build_success and self.build.compilation_errors == 0


@dataclass
class DataSummary:
    """
    Statistical summary of multiple metrics for comparison and analysis.
    
    Simplified replacement for MetricsSummary with essential statistics only.
    """
    metric_count: int = 0
    successful_builds: int = 0
    failed_builds: int = 0
    
    # Performance statistics
    avg_throughput: Optional[float] = None
    max_throughput: Optional[float] = None
    min_throughput: Optional[float] = None
    avg_latency: Optional[float] = None
    min_latency: Optional[float] = None
    
    # Resource statistics  
    avg_lut_utilization: Optional[float] = None
    max_lut_utilization: Optional[float] = None
    avg_dsp_utilization: Optional[float] = None
    max_dsp_utilization: Optional[float] = None
    
    # Quality statistics
    avg_accuracy: Optional[float] = None
    min_accuracy: Optional[float] = None
    
    # Build statistics
    avg_build_time: Optional[float] = None
    total_build_time: Optional[float] = None
    
    @property
    def success_rate(self) -> float:
        """Calculate build success rate."""
        if self.metric_count > 0:
            return self.successful_builds / self.metric_count
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'metric_count': self.metric_count,
            'successful_builds': self.successful_builds,
            'failed_builds': self.failed_builds,
            'success_rate': self.success_rate,
            'avg_throughput': self.avg_throughput,
            'max_throughput': self.max_throughput,
            'min_throughput': self.min_throughput,
            'avg_latency': self.avg_latency,
            'min_latency': self.min_latency,
            'avg_lut_utilization': self.avg_lut_utilization,
            'max_lut_utilization': self.max_lut_utilization,
            'avg_dsp_utilization': self.avg_dsp_utilization,
            'max_dsp_utilization': self.max_dsp_utilization,
            'avg_accuracy': self.avg_accuracy,
            'min_accuracy': self.min_accuracy,
            'avg_build_time': self.avg_build_time,
            'total_build_time': self.total_build_time
        }


@dataclass
class ComparisonResult:
    """
    Result of comparing two sets of metrics.
    
    Simple data structure for metric comparison analysis.
    """
    metrics_a_better: Dict[str, str] = field(default_factory=dict)
    metrics_b_better: Dict[str, str] = field(default_factory=dict)
    improvement_ratios: Dict[str, float] = field(default_factory=dict)
    summary: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'metrics_a_better': self.metrics_a_better,
            'metrics_b_better': self.metrics_b_better,
            'improvement_ratios': self.improvement_ratios,
            'summary': self.summary
        }


@dataclass
class SelectionCriteria:
    """
    Simple criteria for FPGA design selection.
    
    North Star aligned: practical constraints vs academic MCDA complexity.
    """
    max_lut_utilization: Optional[float] = None
    max_dsp_utilization: Optional[float] = None
    max_bram_utilization: Optional[float] = None
    min_throughput: Optional[float] = None
    max_latency: Optional[float] = None
    min_accuracy: Optional[float] = None
    max_build_time: Optional[float] = None
    efficiency_weights: Dict[str, float] = field(default_factory=lambda: {
        'throughput': 0.4,
        'resource_efficiency': 0.3,
        'accuracy': 0.2,
        'build_time': 0.1
    })
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'max_lut_utilization': self.max_lut_utilization,
            'max_dsp_utilization': self.max_dsp_utilization,
            'max_bram_utilization': self.max_bram_utilization,
            'min_throughput': self.min_throughput,
            'max_latency': self.max_latency,
            'min_accuracy': self.min_accuracy,
            'max_build_time': self.max_build_time,
            'efficiency_weights': self.efficiency_weights
        }


@dataclass
class TradeoffAnalysis:
    """
    Simple trade-off analysis between two designs.
    
    Practical FPGA comparison vs complex MCDA analysis.
    """
    efficiency_ratio: float
    throughput_ratio: Optional[float] = None
    resource_ratio: Optional[float] = None
    latency_ratio: Optional[float] = None
    accuracy_ratio: Optional[float] = None
    better_design: str = "unknown"  # "design_a", "design_b", "tied"
    confidence: float = 1.0
    recommendations: List[str] = field(default_factory=list)
    trade_offs: Dict[str, str] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for external analysis."""
        return {
            'efficiency_ratio': self.efficiency_ratio,
            'throughput_ratio': self.throughput_ratio,
            'resource_ratio': self.resource_ratio,
            'latency_ratio': self.latency_ratio,
            'accuracy_ratio': self.accuracy_ratio,
            'better_design': self.better_design,
            'confidence': self.confidence,
            'recommendations': self.recommendations,
            'trade_offs': self.trade_offs
        }


# Type aliases for clarity and backwards compatibility
MetricsList = List[BuildMetrics]
DataList = List[BuildMetrics]
MetricsData = Union[BuildMetrics, MetricsList, DataSummary]