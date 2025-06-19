"""
Core Data Functions - North Star Aligned

Simple functions for FPGA data collection and processing.
Consolidates metrics and analysis functionality while eliminating enterprise complexity.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union

from .types import (
    BuildMetrics, PerformanceData, ResourceData, QualityData, BuildData,
    DataSummary, ComparisonResult, MetricsList, SelectionCriteria, TradeoffAnalysis
)

logger = logging.getLogger(__name__)

# Import streamlined modules
from ..hooks import log_optimization_event
from ..finn import FINNEvaluationBridge as build_accelerator


def collect_build_metrics(
    build_result: Any,
    model_path: Optional[str] = None,
    blueprint_path: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None
) -> BuildMetrics:
    """
    Core BrainSmith function: Collect unified metrics from any build result.
    
    Consolidates functionality from metrics.collect_build_metrics() with enhanced
    data extraction patterns from analysis.expose_analysis_data().
    
    Args:
        build_result: Result from core.forge(), FINN build, or other build process
        model_path: Path to model file
        blueprint_path: Path to blueprint file  
        parameters: Build parameters used
        
    Returns:
        BuildMetrics containing all collected metrics
        
    Example:
        result = forge('model.onnx', 'blueprint.yaml', **params)
        metrics = collect_build_metrics(result, 'model.onnx', 'blueprint.yaml', params)
    """
    start_time = time.time()
    
    # Initialize metrics container
    metrics = BuildMetrics(
        model_path=model_path,
        blueprint_path=blueprint_path,
        parameters=parameters or {},
        timestamp=start_time
    )
    
    # Log data collection start
    log_optimization_event('data_collection_start', {
        'model_path': model_path,
        'blueprint_path': blueprint_path,
        'parameters': parameters
    })
    
    # Handle None or invalid results first
    if build_result is None:
        metrics.build.build_success = False
        metrics.metadata.update({
            'collection_time_seconds': time.time() - start_time,
            'data_source': 'NoneType',
            'has_performance': False,
            'has_resources': False
        })
        log_optimization_event('data_collection_complete', {
            'success': False,
            'reason': 'None build result',
            'collection_time': time.time() - start_time
        })
        return metrics
    
    try:
        # Extract performance metrics
        metrics.performance = _extract_performance_data(build_result)
        
        # Extract resource metrics
        metrics.resources = _extract_resource_data(build_result)
        
        # Extract quality metrics
        metrics.quality = _extract_quality_data(build_result)
        
        # Extract build information
        metrics.build = _extract_build_data(build_result)
        
        # Add collection metadata
        metrics.metadata.update({
            'collection_time_seconds': time.time() - start_time,
            'data_source': type(build_result).__name__,
            'has_performance': metrics.performance.throughput_ops_sec is not None,
            'has_resources': metrics.resources.lut_utilization_percent is not None
        })
        
        # Log successful collection
        log_optimization_event('data_collection_complete', {
            'success': True,
            'collection_time': time.time() - start_time,
            'metrics_extracted': _count_extracted_metrics(metrics)
        })
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        
        # Log failed collection
        log_optimization_event('data_collection_failed', {
            'error': str(e),
            'collection_time': time.time() - start_time
        })
        
        # Mark build as failed in metrics
        metrics.build.build_success = False
        metrics.metadata['collection_error'] = str(e)
    
    return metrics


def collect_dse_metrics(dse_results: List[Any]) -> MetricsList:
    """
    Process DSE parameter sweep results into unified metrics list.
    
    Consolidates analysis.expose_analysis_data() functionality with 
    metrics collection for parameter sweeps.
    
    Args:
        dse_results: List of DSE results from parameter sweeps
        
    Returns:
        List of BuildMetrics, one for each DSE result
        
    Example:
        dse_results = dse.optimize(model, blueprint, param_ranges)
        all_metrics = collect_dse_metrics(dse_results)
        summary = summarize_data(all_metrics)
    """
    if not dse_results:
        return []
    
    logger.info(f"Processing {len(dse_results)} DSE results")
    
    metrics_list = []
    
    for i, result in enumerate(dse_results):
        try:
            # Extract parameters from result
            parameters = getattr(result, 'design_parameters', {})
            if isinstance(result, dict):
                parameters = result.get('parameters', {})
            
            # Collect metrics for this result
            metrics = collect_build_metrics(
                result,
                parameters=parameters
            )
            
            # Add DSE-specific metadata
            metrics.metadata.update({
                'dse_index': i,
                'is_dse_result': True,
                'parameter_sweep': True
            })
            
            metrics_list.append(metrics)
            
        except Exception as e:
            logger.warning(f"Failed to process DSE result {i}: {e}")
            
            # Create minimal failed metrics
            failed_metrics = BuildMetrics(
                parameters=getattr(result, 'design_parameters', {}),
                timestamp=time.time()
            )
            failed_metrics.build.build_success = False
            failed_metrics.metadata['dse_index'] = i
            failed_metrics.metadata['processing_error'] = str(e)
            
            metrics_list.append(failed_metrics)
    
    logger.info(f"Successfully processed {len(metrics_list)} DSE metrics")
    return metrics_list


def summarize_data(metrics_list: MetricsList) -> DataSummary:
    """
    Create statistical summary of multiple metrics.
    
    Simplified version of metrics.summarize_metrics() with essential statistics.
    
    Args:
        metrics_list: List of BuildMetrics to summarize
        
    Returns:
        DataSummary with statistical analysis
        
    Example:
        summary = summarize_data(dse_results_metrics)
        print(f"Success rate: {summary.success_rate:.1%}")
    """
    if not metrics_list:
        return DataSummary()
    
    summary = DataSummary(metric_count=len(metrics_list))
    
    # Count successful/failed builds
    summary.successful_builds = sum(1 for m in metrics_list if m.is_successful())
    summary.failed_builds = summary.metric_count - summary.successful_builds
    
    # Get successful metrics only for performance analysis
    successful_metrics = [m for m in metrics_list if m.is_successful()]
    
    if successful_metrics:
        # Performance statistics
        throughputs = [m.performance.throughput_ops_sec for m in successful_metrics 
                      if m.performance.throughput_ops_sec is not None]
        if throughputs:
            summary.avg_throughput = sum(throughputs) / len(throughputs)
            summary.max_throughput = max(throughputs)
            summary.min_throughput = min(throughputs)
        
        latencies = [m.performance.latency_ms for m in successful_metrics 
                    if m.performance.latency_ms is not None]
        if latencies:
            summary.avg_latency = sum(latencies) / len(latencies)
            summary.min_latency = min(latencies)
        
        # Resource statistics
        lut_utils = [m.resources.lut_utilization_percent for m in successful_metrics 
                    if m.resources.lut_utilization_percent is not None]
        if lut_utils:
            summary.avg_lut_utilization = sum(lut_utils) / len(lut_utils)
            summary.max_lut_utilization = max(lut_utils)
        
        dsp_utils = [m.resources.dsp_utilization_percent for m in successful_metrics 
                    if m.resources.dsp_utilization_percent is not None]
        if dsp_utils:
            summary.avg_dsp_utilization = sum(dsp_utils) / len(dsp_utils)
            summary.max_dsp_utilization = max(dsp_utils)
        
        # Quality statistics
        accuracies = [m.quality.accuracy_percent for m in successful_metrics 
                     if m.quality.accuracy_percent is not None]
        if accuracies:
            summary.avg_accuracy = sum(accuracies) / len(accuracies)
            summary.min_accuracy = min(accuracies)
        
        # Build time statistics
        build_times = [m.build.build_time_seconds for m in successful_metrics]
        if build_times:
            summary.avg_build_time = sum(build_times) / len(build_times)
            summary.total_build_time = sum(build_times)
    
    return summary


def compare_results(metrics_a: BuildMetrics, metrics_b: BuildMetrics) -> ComparisonResult:
    """
    Compare two sets of metrics.
    
    Simplified version of metrics.compare_metrics() with essential comparisons.
    
    Args:
        metrics_a: First metrics to compare
        metrics_b: Second metrics to compare
        
    Returns:
        ComparisonResult with comparison analysis
    """
    comparison = ComparisonResult()
    
    # Performance comparison
    if (metrics_a.performance.throughput_ops_sec and 
        metrics_b.performance.throughput_ops_sec):
        
        throughput_ratio = (metrics_b.performance.throughput_ops_sec / 
                           metrics_a.performance.throughput_ops_sec)
        comparison.improvement_ratios['throughput'] = throughput_ratio
        
        if throughput_ratio > 1.0:
            comparison.metrics_b_better['throughput'] = f"{(throughput_ratio-1)*100:.1f}% higher"
        else:
            comparison.metrics_a_better['throughput'] = f"{(1/throughput_ratio-1)*100:.1f}% higher"
    
    # Resource comparison
    if (metrics_a.resources.lut_utilization_percent and 
        metrics_b.resources.lut_utilization_percent):
        
        lut_ratio = (metrics_b.resources.lut_utilization_percent / 
                    metrics_a.resources.lut_utilization_percent)
        comparison.improvement_ratios['lut_utilization'] = lut_ratio
        
        if lut_ratio < 1.0:
            comparison.metrics_b_better['lut_efficiency'] = f"{(1-lut_ratio)*100:.1f}% lower utilization"
        else:
            comparison.metrics_a_better['lut_efficiency'] = f"{(1-1/lut_ratio)*100:.1f}% lower utilization"
    
    # Overall efficiency comparison
    eff_a = metrics_a.get_efficiency_score()
    eff_b = metrics_b.get_efficiency_score()
    
    if eff_a and eff_b:
        eff_ratio = eff_b / eff_a
        comparison.improvement_ratios['efficiency'] = eff_ratio
        
        if eff_ratio > 1.0:
            comparison.summary['winner'] = 'metrics_b'
            comparison.summary['efficiency_improvement'] = f"{(eff_ratio-1)*100:.1f}%"
        else:
            comparison.summary['winner'] = 'metrics_a'
            comparison.summary['efficiency_improvement'] = f"{(1/eff_ratio-1)*100:.1f}%"
    
    return comparison


def filter_data(metrics_list: MetricsList, criteria: Dict[str, Any]) -> MetricsList:
    """
    Filter metrics based on criteria.
    
    Args:
        metrics_list: List of BuildMetrics to filter
        criteria: Dictionary with filtering criteria
        
    Returns:
        Filtered list of BuildMetrics
        
    Example:
        good_metrics = filter_data(all_metrics, {
            'min_throughput': 1000,
            'max_lut_utilization': 80,
            'build_success': True
        })
    """
    filtered = []
    
    for metrics in metrics_list:
        include = True
        
        # Check build success
        if 'build_success' in criteria:
            if metrics.build.build_success != criteria['build_success']:
                include = False
                continue
        
        # Check performance criteria
        if 'min_throughput' in criteria:
            if (not metrics.performance.throughput_ops_sec or 
                metrics.performance.throughput_ops_sec < criteria['min_throughput']):
                include = False
                continue
        
        if 'max_latency' in criteria:
            if (not metrics.performance.latency_ms or 
                metrics.performance.latency_ms > criteria['max_latency']):
                include = False
                continue
        
        # Check resource criteria
        if 'max_lut_utilization' in criteria:
            if (not metrics.resources.lut_utilization_percent or 
                metrics.resources.lut_utilization_percent > criteria['max_lut_utilization']):
                include = False
                continue
        
        if 'max_dsp_utilization' in criteria:
            if (not metrics.resources.dsp_utilization_percent or 
                metrics.resources.dsp_utilization_percent > criteria['max_dsp_utilization']):
                include = False
                continue
        
        # Check quality criteria
        if 'min_accuracy' in criteria:
            if (not metrics.quality.accuracy_percent or 
                metrics.quality.accuracy_percent < criteria['min_accuracy']):
                include = False
                continue
        
        if include:
            filtered.append(metrics)
    
    logger.info(f"Filtered {len(metrics_list)} metrics to {len(filtered)} matching criteria")
    return filtered


def validate_data(metrics: BuildMetrics) -> List[str]:
    """
    Validate metrics for completeness and consistency.
    
    Args:
        metrics: BuildMetrics to validate
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    # Check for basic data consistency
    if not metrics.build.build_success and metrics.performance.throughput_ops_sec:
        issues.append("Performance data present for failed build")
    
    # Check performance consistency
    if (metrics.performance.throughput_ops_sec and metrics.performance.latency_ms):
        expected_throughput = 1000.0 / metrics.performance.latency_ms
        actual_throughput = metrics.performance.throughput_ops_sec
        
        if abs(expected_throughput - actual_throughput) / expected_throughput > 0.1:
            issues.append(f"Inconsistent throughput/latency: {actual_throughput:.1f} vs {expected_throughput:.1f}")
    
    # Check resource utilization ranges
    resource_checks = [
        ('LUT', metrics.resources.lut_utilization_percent),
        ('DSP', metrics.resources.dsp_utilization_percent),
        ('BRAM', metrics.resources.bram_utilization_percent),
        ('URAM', metrics.resources.uram_utilization_percent),
        ('FF', metrics.resources.ff_utilization_percent)
    ]
    
    for name, utilization in resource_checks:
        if utilization is not None:
            if utilization < 0 or utilization > 100:
                issues.append(f"{name} utilization out of range: {utilization}%")
    
    # Check quality ranges
    if metrics.quality.accuracy_percent is not None:
        if metrics.quality.accuracy_percent < 0 or metrics.quality.accuracy_percent > 100:
            issues.append(f"Accuracy out of range: {metrics.quality.accuracy_percent}%")
    
    return issues


# Private helper functions for data extraction

def _extract_performance_data(build_result: Any) -> PerformanceData:
    """Extract performance metrics from build result."""
    performance = PerformanceData()
    
    try:
        # Extract from different result formats
        if hasattr(build_result, 'performance'):
            # Standard performance object
            perf = build_result.performance
            # Only extract if the values are actual numbers, not Mock objects
            throughput = getattr(perf, 'throughput_ops_sec', None)
            if throughput is not None and not hasattr(throughput, '_mock_name'):
                performance.throughput_ops_sec = throughput
                
            latency = getattr(perf, 'latency_ms', None)
            if latency is not None and not hasattr(latency, '_mock_name'):
                performance.latency_ms = latency
                
            clock_freq = getattr(perf, 'clock_freq_mhz', None)
            if clock_freq is not None and not hasattr(clock_freq, '_mock_name'):
                performance.clock_freq_mhz = clock_freq
                
            cycles = getattr(perf, 'cycles_per_inference', None)
            if cycles is not None and not hasattr(cycles, '_mock_name'):
                performance.cycles_per_inference = cycles
            
        elif isinstance(build_result, dict):
            # Dictionary format
            perf_data = build_result.get('performance', {})
            performance.throughput_ops_sec = perf_data.get('throughput_ops_sec')
            performance.latency_ms = perf_data.get('latency_ms')
            performance.clock_freq_mhz = perf_data.get('clock_freq_mhz')
            performance.cycles_per_inference = perf_data.get('cycles_per_inference')
            performance.max_batch_size = perf_data.get('max_batch_size')
            performance.inference_time_ms = perf_data.get('inference_time_ms')
        
        # Try to extract from objective_values (DSE results)
        if hasattr(build_result, 'objective_values'):
            try:
                obj_vals = build_result.objective_values
                if obj_vals and len(obj_vals) > 0:
                    performance.throughput_ops_sec = obj_vals[0]  # First objective often throughput
                if obj_vals and len(obj_vals) > 1:
                    performance.latency_ms = obj_vals[1]  # Second objective often latency
            except (TypeError, AttributeError):
                # objective_values might be a Mock or invalid
                pass
        
        # Calculate derived metrics only if we have valid numbers
        if (performance.clock_freq_mhz and performance.cycles_per_inference and
            isinstance(performance.clock_freq_mhz, (int, float)) and
            isinstance(performance.cycles_per_inference, (int, float))):
            performance.inference_time_ms = (performance.cycles_per_inference /
                                           (performance.clock_freq_mhz * 1e6)) * 1000
            
    except Exception as e:
        logger.warning(f"Performance metrics extraction failed: {e}")
    
    return performance


def _extract_resource_data(build_result: Any) -> ResourceData:
    """Extract resource utilization metrics from build result."""
    resources = ResourceData()
    
    try:
        # Extract from different result formats
        if hasattr(build_result, 'resources'):
            # Standard resources object
            res = build_result.resources
            # Only extract if the values are actual numbers, not Mock objects
            lut_util = getattr(res, 'lut_utilization_percent', None)
            if lut_util is not None and not hasattr(lut_util, '_mock_name'):
                resources.lut_utilization_percent = lut_util
                
            dsp_util = getattr(res, 'dsp_utilization_percent', None)
            if dsp_util is not None and not hasattr(dsp_util, '_mock_name'):
                resources.dsp_utilization_percent = dsp_util
                
            bram_util = getattr(res, 'bram_utilization_percent', None)
            if bram_util is not None and not hasattr(bram_util, '_mock_name'):
                resources.bram_utilization_percent = bram_util
                
            uram_util = getattr(res, 'uram_utilization_percent', None)
            if uram_util is not None and not hasattr(uram_util, '_mock_name'):
                resources.uram_utilization_percent = uram_util
                
            ff_util = getattr(res, 'ff_utilization_percent', None)
            if ff_util is not None and not hasattr(ff_util, '_mock_name'):
                resources.ff_utilization_percent = ff_util
            
        elif isinstance(build_result, dict):
            # Dictionary format
            res_data = build_result.get('resources', {})
            resources.lut_utilization_percent = res_data.get('lut_utilization_percent')
            resources.dsp_utilization_percent = res_data.get('dsp_utilization_percent')
            resources.bram_utilization_percent = res_data.get('bram_utilization_percent')
            resources.uram_utilization_percent = res_data.get('uram_utilization_percent')
            resources.ff_utilization_percent = res_data.get('ff_utilization_percent')
            
            # Absolute counts
            resources.lut_count = res_data.get('lut_count')
            resources.dsp_count = res_data.get('dsp_count')
            resources.bram_count = res_data.get('bram_count')
            resources.uram_count = res_data.get('uram_count')
            resources.ff_count = res_data.get('ff_count')
            
    except Exception as e:
        logger.warning(f"Resource metrics extraction failed: {e}")
    
    return resources


def _extract_quality_data(build_result: Any) -> QualityData:
    """Extract quality and accuracy metrics from build result."""
    quality = QualityData()
    
    try:
        # Extract from different result formats
        if hasattr(build_result, 'quality'):
            # Standard quality object
            qual = build_result.quality
            quality.accuracy_percent = getattr(qual, 'accuracy_percent', None)
            quality.precision = getattr(qual, 'precision', None)
            quality.recall = getattr(qual, 'recall', None)
            quality.f1_score = getattr(qual, 'f1_score', None)
            
        elif isinstance(build_result, dict):
            # Dictionary format
            qual_data = build_result.get('quality', {})
            quality.accuracy_percent = qual_data.get('accuracy_percent')
            quality.precision = qual_data.get('precision')
            quality.recall = qual_data.get('recall')
            quality.f1_score = qual_data.get('f1_score')
            quality.inference_error_rate = qual_data.get('inference_error_rate')
            quality.numerical_precision_bits = qual_data.get('numerical_precision_bits')
        
        # Try to extract from validation results
        if hasattr(build_result, 'validation_results'):
            val_results = build_result.validation_results
            if isinstance(val_results, dict):
                quality.accuracy_percent = val_results.get('accuracy', 0) * 100
                quality.precision = val_results.get('precision')
                quality.recall = val_results.get('recall')
                quality.f1_score = val_results.get('f1_score')
            
    except Exception as e:
        logger.warning(f"Quality metrics extraction failed: {e}")
    
    return quality


def _extract_build_data(build_result: Any) -> BuildData:
    """Extract build process information from build result."""
    build_info = BuildData()
    
    try:
        # Extract from different result formats
        if hasattr(build_result, 'build_info'):
            # Standard build info object
            info = build_result.build_info
            success = getattr(info, 'build_success', True)
            if not hasattr(success, '_mock_name'):
                build_info.build_success = success
                
            build_time = getattr(info, 'build_time_seconds', 0.0)
            if not hasattr(build_time, '_mock_name'):
                build_info.build_time_seconds = build_time
                
            synth_time = getattr(info, 'synthesis_time_seconds', None)
            if synth_time is not None and not hasattr(synth_time, '_mock_name'):
                build_info.synthesis_time_seconds = synth_time
                
            pr_time = getattr(info, 'place_route_time_seconds', None)
            if pr_time is not None and not hasattr(pr_time, '_mock_name'):
                build_info.place_route_time_seconds = pr_time
            
        elif isinstance(build_result, dict):
            # Dictionary format
            build_data = build_result.get('build_info', {})
            build_info.build_success = build_data.get('build_success', True)
            build_info.build_time_seconds = build_data.get('build_time_seconds', 0.0)
            build_info.synthesis_time_seconds = build_data.get('synthesis_time_seconds')
            build_info.place_route_time_seconds = build_data.get('place_route_time_seconds')
            build_info.compilation_warnings = build_data.get('compilation_warnings', 0)
            build_info.compilation_errors = build_data.get('compilation_errors', 0)
            build_info.target_device = build_data.get('target_device')
        
        # Check for build success indicators (avoid Mock objects)
        if hasattr(build_result, 'success'):
            success = build_result.success
            if not hasattr(success, '_mock_name'):
                build_info.build_success = success
        elif hasattr(build_result, 'build_successful'):
            success = build_result.build_successful
            if not hasattr(success, '_mock_name'):
                build_info.build_success = success
        
        # Extract timing if available (avoid Mock objects)
        if hasattr(build_result, 'total_time'):
            total_time = build_result.total_time
            if not hasattr(total_time, '_mock_name'):
                build_info.build_time_seconds = total_time
        elif hasattr(build_result, 'elapsed_time'):
            elapsed_time = build_result.elapsed_time
            if not hasattr(elapsed_time, '_mock_name'):
                build_info.build_time_seconds = elapsed_time
            
    except Exception as e:
        logger.warning(f"Build info extraction failed: {e}")
    
    return build_info


def _count_extracted_metrics(metrics: BuildMetrics) -> Dict[str, bool]:
    """Count which types of metrics were successfully extracted."""
    return {
        'performance': metrics.performance.throughput_ops_sec is not None,
        'resources': metrics.resources.lut_utilization_percent is not None,
        'quality': metrics.quality.accuracy_percent is not None,
        'build_info': metrics.build.build_success is not None
    }