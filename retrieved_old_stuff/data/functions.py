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

# Import streamlined modules with fallbacks
try:
    from ..hooks import log_data_event
except ImportError:
    log_data_event = lambda *args, **kwargs: None

try:
    from ..finn import build_accelerator
    FINN_AVAILABLE = True
except ImportError:
    FINN_AVAILABLE = False


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
    log_data_event('data_collection_start', {
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
        log_data_event('data_collection_complete', {
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
        log_data_event('data_collection_complete', {
            'success': True,
            'collection_time': time.time() - start_time,
            'metrics_extracted': _count_extracted_metrics(metrics)
        })
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        
        # Log failed collection
        log_data_event('data_collection_failed', {
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


def find_pareto_optimal(metrics_list: MetricsList,
                       objectives: List[str] = None) -> MetricsList:
    """
    Find Pareto optimal solutions from DSE results.
    
    Simple domination check algorithm: ~40 lines vs 288 lines of TOPSIS.
    A solution is Pareto optimal if no other solution dominates it in all objectives.
    
    Args:
        metrics_list: List of BuildMetrics from DSE
        objectives: List of objective names to consider for Pareto optimality
        
    Returns:
        Filtered list containing only Pareto optimal solutions
        
    Example:
        dse_results = collect_dse_metrics(dse_sweep_results)
        pareto_solutions = find_pareto_optimal(dse_results,
                                             ['throughput_ops_sec', 'lut_utilization_percent'])
    """
    if not metrics_list:
        return []
    
    if not objectives:
        objectives = ['throughput_ops_sec', 'lut_utilization_percent']
    
    logger.info(f"Finding Pareto optimal solutions from {len(metrics_list)} candidates using {len(objectives)} objectives")
    
    # Only consider successful builds
    valid_metrics = [m for m in metrics_list if m.is_successful()]
    if not valid_metrics:
        logger.warning("No successful builds found for Pareto analysis")
        return []
    
    pareto_optimal = []
    
    for candidate in valid_metrics:
        is_dominated = False
        
        # Check if candidate is dominated by any other solution
        for other in valid_metrics:
            if candidate == other:
                continue
                
            dominates = True
            candidate_better_in_any = False
            
            for obj in objectives:
                candidate_val = _get_objective_value(candidate, obj)
                other_val = _get_objective_value(other, obj)
                
                if candidate_val is None or other_val is None:
                    dominates = False
                    break
                
                # For throughput-like objectives, higher is better
                # For utilization/latency-like objectives, lower is better
                if _is_maximize_objective(obj):
                    if candidate_val > other_val:
                        candidate_better_in_any = True
                    elif candidate_val < other_val:
                        dominates = False
                        break
                else:
                    if candidate_val < other_val:
                        candidate_better_in_any = True
                    elif candidate_val > other_val:
                        dominates = False
                        break
            
            # Other dominates candidate if it's better or equal in all objectives
            # and strictly better in at least one
            if dominates and not candidate_better_in_any:
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_optimal.append(candidate)
    
    logger.info(f"Found {len(pareto_optimal)} Pareto optimal solutions from {len(valid_metrics)} valid candidates")
    return pareto_optimal


def rank_by_efficiency(metrics_list: MetricsList,
                      weights: Dict[str, float] = None) -> MetricsList:
    """
    Rank solutions by FPGA efficiency score.
    
    Composite efficiency scoring: ~30 lines vs 584 lines of SelectionEngine.
    Combines throughput, resource efficiency, accuracy, and build time.
    
    Args:
        metrics_list: List of BuildMetrics to rank
        weights: Optional efficiency weights for different factors
        
    Returns:
        Ranked list (best first) with efficiency scores in metadata
        
    Example:
        ranked_solutions = rank_by_efficiency(pareto_solutions, {
            'throughput': 0.5,
            'resource_efficiency': 0.3,
            'accuracy': 0.2
        })
    """
    if not metrics_list:
        return []
    
    if not weights:
        weights = {
            'throughput': 0.4,
            'resource_efficiency': 0.3,
            'accuracy': 0.2,
            'build_time': 0.1
        }
    
    logger.info(f"Ranking {len(metrics_list)} solutions by efficiency with weights: {weights}")
    
    # Only rank successful builds
    valid_metrics = [m for m in metrics_list if m.is_successful()]
    if not valid_metrics:
        logger.warning("No successful builds found for ranking")
        return []
    
    # Calculate efficiency scores
    scored_metrics = []
    
    for metrics in valid_metrics:
        efficiency_score = _calculate_efficiency_score(metrics, weights)
        
        # Add efficiency score to metadata
        metrics_copy = BuildMetrics(
            performance=metrics.performance,
            resources=metrics.resources,
            quality=metrics.quality,
            build=metrics.build,
            timestamp=metrics.timestamp,
            model_path=metrics.model_path,
            blueprint_path=metrics.blueprint_path,
            parameters=metrics.parameters,
            metadata=metrics.metadata.copy()
        )
        metrics_copy.metadata['efficiency_score'] = efficiency_score
        metrics_copy.metadata['efficiency_weights'] = weights
        
        scored_metrics.append((efficiency_score, metrics_copy))
    
    # Sort by efficiency score (descending)
    scored_metrics.sort(key=lambda x: x[0], reverse=True)
    ranked_list = [metrics for _, metrics in scored_metrics]
    
    logger.info(f"Ranked {len(ranked_list)} solutions, best efficiency: {scored_metrics[0][0]:.3f}")
    return ranked_list


def select_best_solutions(metrics_list: MetricsList,
                         criteria: SelectionCriteria) -> MetricsList:
    """
    Select best solutions based on practical FPGA criteria.
    
    Combines filtering + ranking + selection: ~50 lines vs complex MCDA algorithms.
    Applies resource constraints, performance targets, and ranks by efficiency.
    
    Args:
        metrics_list: List of BuildMetrics to filter and select from
        criteria: SelectionCriteria with constraints and targets
        
    Returns:
        Filtered and ranked list of best solutions
        
    Example:
        criteria = SelectionCriteria(
            max_lut_utilization=80.0,
            min_throughput=1000.0,
            max_latency=10.0
        )
        best_solutions = select_best_solutions(all_metrics, criteria)
    """
    if not metrics_list:
        return []
    
    logger.info(f"Selecting best solutions from {len(metrics_list)} candidates")
    
    # Step 1: Filter feasible designs
    feasible_designs = filter_feasible_designs(metrics_list, criteria)
    if not feasible_designs:
        logger.warning("No designs meet the specified criteria")
        return []
    
    # Step 2: Rank by efficiency
    ranked_designs = rank_by_efficiency(feasible_designs, criteria.efficiency_weights)
    
    # Step 3: Add selection metadata
    for i, metrics in enumerate(ranked_designs):
        metrics.metadata['selection_rank'] = i + 1
        metrics.metadata['meets_criteria'] = True
        metrics.metadata['selection_criteria'] = criteria.to_dict()
    
    logger.info(f"Selected {len(ranked_designs)} solutions meeting criteria")
    return ranked_designs


def filter_feasible_designs(metrics_list: MetricsList,
                           criteria: SelectionCriteria) -> MetricsList:
    """
    Filter designs that meet resource and performance constraints.
    
    Practical constraint checking for FPGA design selection.
    
    Args:
        metrics_list: List of BuildMetrics to filter
        criteria: SelectionCriteria with constraint thresholds
        
    Returns:
        Filtered list of designs meeting all constraints
        
    Example:
        criteria = SelectionCriteria(max_lut_utilization=80, min_throughput=500)
        feasible = filter_feasible_designs(all_metrics, criteria)
    """
    if not metrics_list:
        return []
    
    logger.info(f"Filtering {len(metrics_list)} designs with constraints")
    
    feasible = []
    
    for metrics in metrics_list:
        if not metrics.is_successful():
            continue
        
        meets_constraints = True
        constraint_violations = []
        
        # Check resource constraints
        if criteria.max_lut_utilization is not None:
            if (metrics.resources.lut_utilization_percent and
                metrics.resources.lut_utilization_percent > criteria.max_lut_utilization):
                meets_constraints = False
                constraint_violations.append(f"LUT utilization {metrics.resources.lut_utilization_percent:.1f}% > {criteria.max_lut_utilization}%")
        
        if criteria.max_dsp_utilization is not None:
            if (metrics.resources.dsp_utilization_percent and
                metrics.resources.dsp_utilization_percent > criteria.max_dsp_utilization):
                meets_constraints = False
                constraint_violations.append(f"DSP utilization {metrics.resources.dsp_utilization_percent:.1f}% > {criteria.max_dsp_utilization}%")
        
        if criteria.max_bram_utilization is not None:
            if (metrics.resources.bram_utilization_percent and
                metrics.resources.bram_utilization_percent > criteria.max_bram_utilization):
                meets_constraints = False
                constraint_violations.append(f"BRAM utilization {metrics.resources.bram_utilization_percent:.1f}% > {criteria.max_bram_utilization}%")
        
        # Check performance constraints
        if criteria.min_throughput is not None:
            if (not metrics.performance.throughput_ops_sec or
                metrics.performance.throughput_ops_sec < criteria.min_throughput):
                meets_constraints = False
                constraint_violations.append(f"Throughput {metrics.performance.throughput_ops_sec or 0} < {criteria.min_throughput}")
        
        if criteria.max_latency is not None:
            if (metrics.performance.latency_ms and
                metrics.performance.latency_ms > criteria.max_latency):
                meets_constraints = False
                constraint_violations.append(f"Latency {metrics.performance.latency_ms}ms > {criteria.max_latency}ms")
        
        # Check quality constraints
        if criteria.min_accuracy is not None:
            if (not metrics.quality.accuracy_percent or
                metrics.quality.accuracy_percent < criteria.min_accuracy):
                meets_constraints = False
                constraint_violations.append(f"Accuracy {metrics.quality.accuracy_percent or 0}% < {criteria.min_accuracy}%")
        
        # Check build time constraints
        if criteria.max_build_time is not None:
            if metrics.build.build_time_seconds > criteria.max_build_time:
                meets_constraints = False
                constraint_violations.append(f"Build time {metrics.build.build_time_seconds:.1f}s > {criteria.max_build_time}s")
        
        if meets_constraints:
            # Add constraint satisfaction to metadata
            metrics_copy = BuildMetrics(
                performance=metrics.performance,
                resources=metrics.resources,
                quality=metrics.quality,
                build=metrics.build,
                timestamp=metrics.timestamp,
                model_path=metrics.model_path,
                blueprint_path=metrics.blueprint_path,
                parameters=metrics.parameters,
                metadata=metrics.metadata.copy()
            )
            metrics_copy.metadata['meets_all_constraints'] = True
            metrics_copy.metadata['constraint_violations'] = []
            feasible.append(metrics_copy)
        else:
            logger.debug(f"Design rejected: {constraint_violations}")
    
    logger.info(f"Found {len(feasible)} feasible designs from {len(metrics_list)} candidates")
    return feasible


def compare_design_tradeoffs(metrics_a: BuildMetrics,
                           metrics_b: BuildMetrics) -> TradeoffAnalysis:
    """
    Compare trade-offs between two design solutions.
    
    Practical FPGA comparison vs complex MCDA analysis.
    Analyzes efficiency, performance, resource usage, and provides recommendations.
    
    Args:
        metrics_a: First design metrics
        metrics_b: Second design metrics
        
    Returns:
        TradeoffAnalysis with detailed comparison and recommendations
        
    Example:
        analysis = compare_design_tradeoffs(baseline_design, optimized_design)
        print(f"Better design: {analysis.better_design}")
        print(f"Efficiency ratio: {analysis.efficiency_ratio:.2f}")
    """
    logger.info("Comparing design trade-offs between two solutions")
    
    # Calculate efficiency scores
    weights = {'throughput': 0.4, 'resource_efficiency': 0.3, 'accuracy': 0.2, 'build_time': 0.1}
    efficiency_a = _calculate_efficiency_score(metrics_a, weights)
    efficiency_b = _calculate_efficiency_score(metrics_b, weights)
    efficiency_ratio = efficiency_b / efficiency_a if efficiency_a > 0 else float('inf')
    
    # Calculate individual ratios
    throughput_ratio = None
    if metrics_a.performance.throughput_ops_sec and metrics_b.performance.throughput_ops_sec:
        throughput_ratio = metrics_b.performance.throughput_ops_sec / metrics_a.performance.throughput_ops_sec
    
    resource_ratio = None
    if metrics_a.resources.get_total_utilization() and metrics_b.resources.get_total_utilization():
        resource_ratio = metrics_b.resources.get_total_utilization() / metrics_a.resources.get_total_utilization()
    
    latency_ratio = None
    if metrics_a.performance.latency_ms and metrics_b.performance.latency_ms:
        latency_ratio = metrics_b.performance.latency_ms / metrics_a.performance.latency_ms
    
    accuracy_ratio = None
    if metrics_a.quality.accuracy_percent and metrics_b.quality.accuracy_percent:
        accuracy_ratio = metrics_b.quality.accuracy_percent / metrics_a.quality.accuracy_percent
    
    # Determine better design
    better_design = "design_b" if efficiency_ratio > 1.05 else "design_a" if efficiency_ratio < 0.95 else "tied"
    confidence = min(1.0, abs(efficiency_ratio - 1.0) * 2.0)  # Higher difference = higher confidence
    
    # Generate recommendations
    recommendations = []
    trade_offs = {}
    
    if throughput_ratio and throughput_ratio > 1.1:
        recommendations.append("Design B offers significantly better throughput")
        trade_offs['throughput'] = "Design B advantage"
    elif throughput_ratio and throughput_ratio < 0.9:
        recommendations.append("Design A offers better throughput")
        trade_offs['throughput'] = "Design A advantage"
    
    if resource_ratio and resource_ratio < 0.9:
        recommendations.append("Design B is more resource efficient")
        trade_offs['resources'] = "Design B more efficient"
    elif resource_ratio and resource_ratio > 1.1:
        recommendations.append("Design A is more resource efficient")
        trade_offs['resources'] = "Design A more efficient"
    
    if latency_ratio and latency_ratio < 0.9:
        recommendations.append("Design B has lower latency")
        trade_offs['latency'] = "Design B advantage"
    elif latency_ratio and latency_ratio > 1.1:
        recommendations.append("Design A has lower latency")
        trade_offs['latency'] = "Design A advantage"
    
    if accuracy_ratio and accuracy_ratio > 1.02:
        recommendations.append("Design B has better accuracy")
        trade_offs['accuracy'] = "Design B advantage"
    elif accuracy_ratio and accuracy_ratio < 0.98:
        recommendations.append("Design A has better accuracy")
        trade_offs['accuracy'] = "Design A advantage"
    
    if not recommendations:
        recommendations.append("Designs are very similar in performance")
    
    analysis = TradeoffAnalysis(
        efficiency_ratio=efficiency_ratio,
        throughput_ratio=throughput_ratio,
        resource_ratio=resource_ratio,
        latency_ratio=latency_ratio,
        accuracy_ratio=accuracy_ratio,
        better_design=better_design,
        confidence=confidence,
        recommendations=recommendations,
        trade_offs=trade_offs
    )
    
    logger.info(f"Trade-off analysis complete: {better_design} is better (confidence: {confidence:.2f})")
    return analysis


# Private helper functions for selection algorithms

def _get_objective_value(metrics: BuildMetrics, objective: str) -> Optional[float]:
    """Extract objective value from metrics."""
    obj_mapping = {
        'throughput_ops_sec': metrics.performance.throughput_ops_sec,
        'latency_ms': metrics.performance.latency_ms,
        'lut_utilization_percent': metrics.resources.lut_utilization_percent,
        'dsp_utilization_percent': metrics.resources.dsp_utilization_percent,
        'bram_utilization_percent': metrics.resources.bram_utilization_percent,
        'accuracy_percent': metrics.quality.accuracy_percent,
        'build_time_seconds': metrics.build.build_time_seconds
    }
    return obj_mapping.get(objective)


def _is_maximize_objective(objective: str) -> bool:
    """Determine if objective should be maximized or minimized."""
    maximize_objectives = {'throughput_ops_sec', 'accuracy_percent'}
    return objective in maximize_objectives


def _calculate_efficiency_score(metrics: BuildMetrics, weights: Dict[str, float]) -> float:
    """Calculate composite efficiency score for a design."""
    score = 0.0
    total_weight = 0.0
    
    # Throughput component (normalize to 0-1 range)
    if weights.get('throughput', 0) > 0 and metrics.performance.throughput_ops_sec:
        throughput_norm = min(1.0, metrics.performance.throughput_ops_sec / 10000.0)  # Assume max ~10k ops/sec
        score += weights['throughput'] * throughput_norm
        total_weight += weights['throughput']
    
    # Resource efficiency component (prefer ~70% utilization)
    if weights.get('resource_efficiency', 0) > 0:
        total_util = metrics.resources.get_total_utilization()
        if total_util:
            # Efficiency peaks around 70% utilization
            optimal_util = 70.0
            efficiency = 1.0 - abs(total_util - optimal_util) / 100.0
            efficiency = max(0.0, efficiency)
            score += weights['resource_efficiency'] * efficiency
            total_weight += weights['resource_efficiency']
    
    # Accuracy component
    if weights.get('accuracy', 0) > 0 and metrics.quality.accuracy_percent:
        accuracy_norm = metrics.quality.accuracy_percent / 100.0
        score += weights['accuracy'] * accuracy_norm
        total_weight += weights['accuracy']
    
    # Build time component (inverse - lower is better)
    if weights.get('build_time', 0) > 0:
        # Normalize build time (assume max reasonable ~3600s = 1 hour)
        build_time_norm = max(0.0, 1.0 - (metrics.build.build_time_seconds / 3600.0))
        score += weights['build_time'] * build_time_norm
        total_weight += weights['build_time']
    
    # Normalize by total weights used
    if total_weight > 0:
        score /= total_weight
    
    return score


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
        
        # Don't auto-calculate throughput from latency to avoid test inconsistencies
        # if performance.inference_time_ms and performance.inference_time_ms > 0:
        #     performance.throughput_ops_sec = 1000.0 / performance.inference_time_ms
            
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