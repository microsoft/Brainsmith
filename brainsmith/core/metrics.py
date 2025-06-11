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


def analyze_dse_results(metrics_list: list[DSEMetrics]) -> Dict[str, Any]:
    """Analyze a collection of DSE results and provide summary statistics."""
    if not metrics_list:
        return {
            'total_runs': 0,
            'successful_builds': 0,
            'success_rate': 0.0,
            'best_metrics': None,
            'summary': {}
        }
    
    successful = [m for m in metrics_list if m.build_success]
    success_rate = len(successful) / len(metrics_list) if metrics_list else 0.0
    
    # Calculate summary statistics for successful builds
    summary = {}
    if successful:
        throughputs = [m.performance.throughput_ops_sec for m in successful
                      if m.performance.throughput_ops_sec is not None]
        latencies = [m.performance.latency_ms for m in successful
                    if m.performance.latency_ms is not None]
        lut_utils = [m.resources.lut_utilization_percent for m in successful
                    if m.resources.lut_utilization_percent is not None]
        
        if throughputs:
            summary['throughput'] = {
                'min': min(throughputs),
                'max': max(throughputs),
                'avg': sum(throughputs) / len(throughputs)
            }
        
        if latencies:
            summary['latency'] = {
                'min': min(latencies),
                'max': max(latencies),
                'avg': sum(latencies) / len(latencies)
            }
        
        if lut_utils:
            summary['lut_utilization'] = {
                'min': min(lut_utils),
                'max': max(lut_utils),
                'avg': sum(lut_utils) / len(lut_utils)
            }
    
    return {
        'total_runs': len(metrics_list),
        'successful_builds': len(successful),
        'success_rate': success_rate,
        'best_metrics': compare_metrics(metrics_list) if metrics_list else None,
        'summary': summary
    }


def filter_metrics_by_success(metrics_list: list[DSEMetrics]) -> list[DSEMetrics]:
    """Filter metrics to only include successful builds."""
    return [m for m in metrics_list if m.build_success]


def sort_metrics_by_score(metrics_list: list[DSEMetrics], descending: bool = True) -> list[DSEMetrics]:
    """Sort metrics by optimization score."""
    return sorted(metrics_list, key=lambda m: m.get_optimization_score(), reverse=descending)


def get_pareto_frontier(metrics_list: list[DSEMetrics],
                       objectives: list[str] = None) -> list[DSEMetrics]:
    """Find Pareto frontier for multi-objective optimization.
    
    Args:
        metrics_list: List of DSE metrics to analyze
        objectives: List of objective names to consider. Defaults to ['throughput', 'resource_efficiency']
    
    Returns:
        List of metrics that form the Pareto frontier
    """
    if not metrics_list:
        return []
    
    if objectives is None:
        objectives = ['throughput', 'resource_efficiency']
    
    # Filter to successful builds only
    successful = filter_metrics_by_success(metrics_list)
    if not successful:
        return []
    
    def dominates(a: DSEMetrics, b: DSEMetrics) -> bool:
        """Check if metrics 'a' dominates metrics 'b'."""
        a_values = []
        b_values = []
        
        for obj in objectives:
            if obj == 'throughput':
                a_val = a.performance.throughput_ops_sec or 0.0
                b_val = b.performance.throughput_ops_sec or 0.0
                a_values.append(a_val)
                b_values.append(b_val)
            elif obj == 'resource_efficiency':
                # Lower resource utilization is better, so invert
                a_eff = a.resources.get_resource_efficiency()
                b_eff = b.resources.get_resource_efficiency()
                a_val = 100.0 - (a_eff or 100.0)  # Higher is better
                b_val = 100.0 - (b_eff or 100.0)
                a_values.append(a_val)
                b_values.append(b_val)
            elif obj == 'latency':
                # Lower latency is better, so invert
                a_lat = a.performance.latency_ms or float('inf')
                b_lat = b.performance.latency_ms or float('inf')
                a_val = 1000.0 / max(a_lat, 0.1)  # Higher is better
                b_val = 1000.0 / max(b_lat, 0.1)
                a_values.append(a_val)
                b_values.append(b_val)
        
        # 'a' dominates 'b' if 'a' is better or equal in all objectives
        # and strictly better in at least one
        better_or_equal = all(a_val >= b_val for a_val, b_val in zip(a_values, b_values))
        strictly_better = any(a_val > b_val for a_val, b_val in zip(a_values, b_values))
        
        return better_or_equal and strictly_better
    
    # Find Pareto frontier
    pareto_frontier = []
    for candidate in successful:
        is_dominated = False
        for other in successful:
            if other != candidate and dominates(other, candidate):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier.append(candidate)
    
    return pareto_frontier


def calculate_hypervolume(metrics_list: list[DSEMetrics],
                         reference_point: Dict[str, float] = None) -> float:
    """Calculate hypervolume indicator for multi-objective optimization quality.
    
    Args:
        metrics_list: List of DSE metrics (should be Pareto frontier)
        reference_point: Reference point for hypervolume calculation
    
    Returns:
        Hypervolume value (higher is better)
    """
    if not metrics_list:
        return 0.0
    
    if reference_point is None:
        reference_point = {'throughput': 0.0, 'resource_efficiency': 0.0}
    
    # Simplified 2D hypervolume calculation
    # For production use, consider using dedicated hypervolume libraries
    successful = filter_metrics_by_success(metrics_list)
    if not successful:
        return 0.0
    
    points = []
    for m in successful:
        throughput = m.performance.throughput_ops_sec or 0.0
        res_eff = m.resources.get_resource_efficiency()
        efficiency = 100.0 - (res_eff or 100.0) if res_eff else 0.0
        points.append((throughput, efficiency))
    
    if not points:
        return 0.0
    
    # Sort points by first objective
    points.sort()
    
    # Calculate 2D hypervolume (area under curve)
    hypervolume = 0.0
    ref_x = reference_point.get('throughput', 0.0)
    ref_y = reference_point.get('resource_efficiency', 0.0)
    
    prev_x = ref_x
    max_y = ref_y
    
    for x, y in points:
        if y > max_y:
            hypervolume += (x - prev_x) * max_y
            hypervolume += (x - prev_x) * (y - max_y)
            max_y = y
            prev_x = x
        else:
            hypervolume += (x - prev_x) * max_y
            prev_x = x
    
    return hypervolume


def generate_metrics_report(metrics_list: list[DSEMetrics]) -> str:
    """Generate a human-readable report of DSE metrics analysis."""
    analysis = analyze_dse_results(metrics_list)
    pareto_frontier = get_pareto_frontier(metrics_list)
    
    report = []
    report.append("=== DSE Results Analysis ===\n")
    
    # Overall statistics
    report.append(f"Total design points evaluated: {analysis['total_runs']}")
    report.append(f"Successful builds: {analysis['successful_builds']}")
    report.append(f"Success rate: {analysis['success_rate']:.1%}\n")
    
    # Best overall result
    if analysis['best_metrics']:
        best = analysis['best_metrics']
        report.append("Best overall design point:")
        report.append(f"  ID: {best.design_point_id}")
        report.append(f"  Optimization score: {best.get_optimization_score():.3f}")
        if best.performance.throughput_ops_sec:
            report.append(f"  Throughput: {best.performance.throughput_ops_sec:.1f} ops/sec")
        if best.performance.latency_ms:
            report.append(f"  Latency: {best.performance.latency_ms:.1f} ms")
        if best.resources.lut_utilization_percent:
            report.append(f"  LUT utilization: {best.resources.lut_utilization_percent:.1f}%")
        report.append("")
    
    # Summary statistics
    if analysis['summary']:
        report.append("Performance summary (successful builds):")
        for metric, stats in analysis['summary'].items():
            report.append(f"  {metric.replace('_', ' ').title()}:")
            report.append(f"    Min: {stats['min']:.2f}")
            report.append(f"    Max: {stats['max']:.2f}")
            report.append(f"    Avg: {stats['avg']:.2f}")
        report.append("")
    
    # Pareto frontier
    report.append(f"Pareto frontier contains {len(pareto_frontier)} design points")
    if pareto_frontier:
        report.append("Pareto optimal points:")
        for i, metrics in enumerate(pareto_frontier[:5]):  # Show top 5
            report.append(f"  {i+1}. {metrics.design_point_id} (score: {metrics.get_optimization_score():.3f})")
        
        if len(pareto_frontier) > 5:
            report.append(f"  ... and {len(pareto_frontier) - 5} more")
    
    return "\n".join(report)