"""
Data Export Functions

Functions for exporting metrics, analysis results, and other data to various formats.
Supports JSON, CSV, and other common formats for data analysis.
"""

import json
import csv
import logging
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import time

from .types import BuildMetrics, DataSummary, MetricsList

logger = logging.getLogger(__name__)

# Import pandas for advanced export capabilities
import pandas as pd


def export_metrics(
    metrics: Union[BuildMetrics, MetricsList],
    output_path: str,
    format: str = 'json'
) -> bool:
    """
    Export metrics to file in specified format.
    
    Args:
        metrics: Single BuildMetrics or list of BuildMetrics
        output_path: Path to output file
        format: Export format ('json', 'csv', 'excel')
        
    Returns:
        True if export successful, False otherwise
        
    Example:
        export_metrics(metrics, 'results.json', 'json')
        export_metrics(metrics_list, 'results.csv', 'csv')
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Normalize to list
        if isinstance(metrics, list):
            metrics_list = metrics
        else:
            metrics_list = [metrics]
        
        if format.lower() == 'json':
            return _export_json(metrics_list, output_file)
        elif format.lower() == 'csv':
            return _export_csv(metrics_list, output_file)
        elif format.lower() == 'excel':
            return _export_excel(metrics_list, output_file)
        else:
            logger.error(f"Unsupported export format: {format}")
            return False
            
    except Exception as e:
        logger.error(f"Export failed: {e}")
        return False


def export_summary(summary: DataSummary, output_path: str) -> bool:
    """
    Export data summary to JSON file.
    
    Args:
        summary: DataSummary to export
        output_path: Path to output file
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary_dict = {
            'metric_count': summary.metric_count,
            'successful_builds': summary.successful_builds,
            'failed_builds': summary.failed_builds,
            'success_rate': summary.successful_builds / summary.metric_count if summary.metric_count > 0 else 0,
            
            'performance_statistics': {
                'avg_throughput': summary.avg_throughput,
                'max_throughput': summary.max_throughput,
                'min_throughput': summary.min_throughput,
                'avg_latency': summary.avg_latency,
                'min_latency': summary.min_latency
            },
            
            'resource_statistics': {
                'avg_lut_utilization': summary.avg_lut_utilization,
                'max_lut_utilization': summary.max_lut_utilization,
                'avg_dsp_utilization': summary.avg_dsp_utilization,
                'max_dsp_utilization': summary.max_dsp_utilization
            },
            
            'quality_statistics': {
                'avg_accuracy': summary.avg_accuracy,
                'min_accuracy': summary.min_accuracy
            },
            
            'build_statistics': {
                'avg_build_time': summary.avg_build_time,
                'total_build_time': summary.total_build_time
            },
            
            'export_metadata': {
                'export_time': time.time(),
                'export_format': 'summary_json'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary_dict, f, indent=2, default=str)
        
        logger.info(f"Summary exported to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Summary export failed: {e}")
        return False


def export_pareto_frontier(
    pareto_points: List[BuildMetrics],
    output_path: str,
    objectives: List[str] = None
) -> bool:
    """
    Export Pareto frontier points with objective analysis.
    
    Args:
        pareto_points: List of Pareto optimal BuildMetrics
        output_path: Path to output file
        objectives: List of objective names to include
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        if not objectives:
            objectives = ['performance.throughput_ops_sec', 'resources.lut_utilization_percent']
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        pareto_data = {
            'pareto_points': [],
            'analysis': {
                'total_points': len(pareto_points),
                'objectives': objectives,
                'export_time': time.time()
            }
        }
        
        for i, point in enumerate(pareto_points):
            point_data = {
                'point_id': i,
                'parameters': point.parameters,
                'objectives': {},
                'metrics': {
                    'performance': {
                        'throughput_ops_sec': point.performance.throughput_ops_sec,
                        'latency_ms': point.performance.latency_ms,
                        'clock_freq_mhz': point.performance.clock_freq_mhz
                    },
                    'resources': {
                        'lut_utilization_percent': point.resources.lut_utilization_percent,
                        'dsp_utilization_percent': point.resources.dsp_utilization_percent,
                        'bram_utilization_percent': point.resources.bram_utilization_percent
                    },
                    'quality': {
                        'accuracy_percent': point.quality.accuracy_percent
                    },
                    'build': {
                        'build_success': point.build.build_success,
                        'build_time_seconds': point.build.build_time_seconds
                    }
                }
            }
            
            # Extract objective values
            for obj in objectives:
                obj_value = _get_nested_value(point, obj)
                point_data['objectives'][obj] = obj_value
            
            pareto_data['pareto_points'].append(point_data)
        
        with open(output_file, 'w') as f:
            json.dump(pareto_data, f, indent=2, default=str)
        
        logger.info(f"Pareto frontier exported to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Pareto frontier export failed: {e}")
        return False


def export_comparison(
    metrics_a: BuildMetrics,
    metrics_b: BuildMetrics,
    output_path: str,
    comparison_name: str = "comparison"
) -> bool:
    """
    Export comparison between two metrics sets.
    
    Args:
        metrics_a: First metrics for comparison
        metrics_b: Second metrics for comparison
        output_path: Path to output file
        comparison_name: Name for this comparison
        
    Returns:
        True if export successful, False otherwise
    """
    try:
        from .collection import compare_results
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        comparison = compare_results(metrics_a, metrics_b)
        
        comparison_data = {
            'comparison_name': comparison_name,
            'metrics_a': _metrics_to_dict(metrics_a),
            'metrics_b': _metrics_to_dict(metrics_b),
            'improvement_ratios': comparison.improvement_ratios,
            'metrics_a_better': comparison.metrics_a_better,
            'metrics_b_better': comparison.metrics_b_better,
            'summary': comparison.summary,
            'export_metadata': {
                'export_time': time.time(),
                'comparison_type': 'two_metrics'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(comparison_data, f, indent=2, default=str)
        
        logger.info(f"Comparison exported to: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Comparison export failed: {e}")
        return False


def _export_json(metrics_list: MetricsList, output_file: Path) -> bool:
    """Export metrics list to JSON format."""
    try:
        metrics_data = {
            'metrics': [_metrics_to_dict(m) for m in metrics_list],
            'metadata': {
                'total_metrics': len(metrics_list),
                'export_time': time.time(),
                'export_format': 'json'
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(metrics_data, f, indent=2, default=str)
        
        logger.info(f"JSON export completed: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"JSON export failed: {e}")
        return False


def _export_csv(metrics_list: MetricsList, output_file: Path) -> bool:
    """Export metrics list to CSV format."""
    try:
        if not metrics_list:
            return False
        
        # Flatten metrics to rows
        rows = []
        for i, metrics in enumerate(metrics_list):
            row = {
                'metric_id': i,
                'model_path': metrics.model_path,
                'blueprint_path': metrics.blueprint_path,
                'timestamp': metrics.timestamp,
                
                # Performance metrics
                'throughput_ops_sec': metrics.performance.throughput_ops_sec,
                'latency_ms': metrics.performance.latency_ms,
                'clock_freq_mhz': metrics.performance.clock_freq_mhz,
                'cycles_per_inference': metrics.performance.cycles_per_inference,
                
                # Resource metrics
                'lut_utilization_percent': metrics.resources.lut_utilization_percent,
                'dsp_utilization_percent': metrics.resources.dsp_utilization_percent,
                'bram_utilization_percent': metrics.resources.bram_utilization_percent,
                'uram_utilization_percent': metrics.resources.uram_utilization_percent,
                'ff_utilization_percent': metrics.resources.ff_utilization_percent,
                
                # Quality metrics
                'accuracy_percent': metrics.quality.accuracy_percent,
                'precision': metrics.quality.precision,
                'recall': metrics.quality.recall,
                'f1_score': metrics.quality.f1_score,
                
                # Build metrics
                'build_success': metrics.build.build_success,
                'build_time_seconds': metrics.build.build_time_seconds,
                'synthesis_time_seconds': metrics.build.synthesis_time_seconds,
                'place_route_time_seconds': metrics.build.place_route_time_seconds,
                
                # Parameters (flatten dict)
                **{f'param_{k}': v for k, v in metrics.parameters.items()}
            }
            rows.append(row)
        
        # Write CSV
        if rows:
            fieldnames = rows[0].keys()
            with open(output_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        logger.info(f"CSV export completed: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"CSV export failed: {e}")
        return False


def _export_excel(metrics_list: MetricsList, output_file: Path) -> bool:
    """Export metrics list to Excel format using pandas."""
    try:
        # Convert to DataFrame
        rows = []
        for i, metrics in enumerate(metrics_list):
            row = {
                'metric_id': i,
                'model_path': metrics.model_path,
                'blueprint_path': metrics.blueprint_path,
                'timestamp': metrics.timestamp,
                
                # Performance
                'throughput_ops_sec': metrics.performance.throughput_ops_sec,
                'latency_ms': metrics.performance.latency_ms,
                'clock_freq_mhz': metrics.performance.clock_freq_mhz,
                
                # Resources
                'lut_utilization_percent': metrics.resources.lut_utilization_percent,
                'dsp_utilization_percent': metrics.resources.dsp_utilization_percent,
                'bram_utilization_percent': metrics.resources.bram_utilization_percent,
                
                # Quality
                'accuracy_percent': metrics.quality.accuracy_percent,
                
                # Build
                'build_success': metrics.build.build_success,
                'build_time_seconds': metrics.build.build_time_seconds,
                
                # Parameters
                **{f'param_{k}': v for k, v in metrics.parameters.items()}
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        
        # Export to Excel with multiple sheets
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Metrics', index=False)
            
            # Add summary sheet if enough data
            if len(metrics_list) > 1:
                successful = [m for m in metrics_list if m.build.build_success]
                if successful:
                    summary_data = {
                        'Statistic': ['Total Metrics', 'Successful Builds', 'Success Rate',
                                    'Avg Throughput', 'Avg Latency', 'Avg LUT Util'],
                        'Value': [
                            len(metrics_list),
                            len(successful),
                            len(successful) / len(metrics_list),
                            sum(m.performance.throughput_ops_sec for m in successful 
                                if m.performance.throughput_ops_sec) / len(successful),
                            sum(m.performance.latency_ms for m in successful 
                                if m.performance.latency_ms) / len(successful),
                            sum(m.resources.lut_utilization_percent for m in successful 
                                if m.resources.lut_utilization_percent) / len(successful)
                        ]
                    }
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        logger.info(f"Excel export completed: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Excel export failed: {e}")
        return False


def _metrics_to_dict(metrics: BuildMetrics) -> Dict[str, Any]:
    """Convert BuildMetrics to dictionary for export."""
    return {
        'model_path': metrics.model_path,
        'blueprint_path': metrics.blueprint_path,
        'parameters': metrics.parameters,
        'timestamp': metrics.timestamp,
        
        'performance': {
            'throughput_ops_sec': metrics.performance.throughput_ops_sec,
            'latency_ms': metrics.performance.latency_ms,
            'clock_freq_mhz': metrics.performance.clock_freq_mhz,
            'cycles_per_inference': metrics.performance.cycles_per_inference,
            'max_batch_size': metrics.performance.max_batch_size,
            'inference_time_ms': metrics.performance.inference_time_ms
        },
        
        'resources': {
            'lut_utilization_percent': metrics.resources.lut_utilization_percent,
            'dsp_utilization_percent': metrics.resources.dsp_utilization_percent,
            'bram_utilization_percent': metrics.resources.bram_utilization_percent,
            'uram_utilization_percent': metrics.resources.uram_utilization_percent,
            'ff_utilization_percent': metrics.resources.ff_utilization_percent,
            'lut_count': metrics.resources.lut_count,
            'dsp_count': metrics.resources.dsp_count,
            'bram_count': metrics.resources.bram_count,
            'uram_count': metrics.resources.uram_count,
            'ff_count': metrics.resources.ff_count
        },
        
        'quality': {
            'accuracy_percent': metrics.quality.accuracy_percent,
            'precision': metrics.quality.precision,
            'recall': metrics.quality.recall,
            'f1_score': metrics.quality.f1_score,
            'inference_error_rate': metrics.quality.inference_error_rate,
            'numerical_precision_bits': metrics.quality.numerical_precision_bits
        },
        
        'build': {
            'build_success': metrics.build.build_success,
            'build_time_seconds': metrics.build.build_time_seconds,
            'synthesis_time_seconds': metrics.build.synthesis_time_seconds,
            'place_route_time_seconds': metrics.build.place_route_time_seconds,
            'compilation_warnings': metrics.build.compilation_warnings,
            'compilation_errors': metrics.build.compilation_errors,
            'target_device': metrics.build.target_device
        },
        
        'metadata': metrics.metadata
    }


def _get_nested_value(obj: Any, path: str) -> Any:
    """Get nested value using dot notation path."""
    try:
        current = obj
        for part in path.split('.'):
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current
    except (AttributeError, KeyError, TypeError):
        return None


# Utility functions for common export patterns

def export_best_results(
    metrics_list: MetricsList,
    output_dir: str,
    top_n: int = 10
) -> bool:
    """
    Export top N best results based on efficiency.
    
    Args:
        metrics_list: List of metrics to analyze
        output_dir: Directory to save results
        top_n: Number of top results to export
        
    Returns:
        True if export successful
    """
    try:
        from .collection import filter_data
        
        # Filter to successful builds only
        successful = filter_data(metrics_list, {'build_success': True})
        
        if not successful:
            logger.warning("No successful builds to export")
            return False
        
        # Sort by efficiency score
        sorted_metrics = sorted(successful, 
                              key=lambda m: m.get_efficiency_score() or 0, 
                              reverse=True)
        
        # Take top N
        top_results = sorted_metrics[:top_n]
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export top results
        return export_metrics(top_results, output_dir / f"top_{top_n}_results.json", 'json')
        
    except Exception as e:
        logger.error(f"Best results export failed: {e}")
        return False


def export_dse_analysis(
    metrics_list: MetricsList,
    output_dir: str,
    include_pareto: bool = True
) -> bool:
    """
    Export complete DSE analysis including summary, best results, and Pareto frontier.
    
    Args:
        metrics_list: List of metrics from DSE
        output_dir: Directory to save analysis
        include_pareto: Whether to include Pareto frontier analysis
        
    Returns:
        True if export successful
    """
    try:
        from .collection import summarize_data, filter_data
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Export all metrics
        export_metrics(metrics_list, output_dir / "all_results.json", 'json')
        export_metrics(metrics_list, output_dir / "all_results.csv", 'csv')
        
        # Export summary
        summary = summarize_data(metrics_list)
        export_summary(summary, output_dir / "summary.json")
        
        # Export best results
        export_best_results(metrics_list, str(output_dir), 10)
        
        # Export Pareto frontier if requested
        if include_pareto:
            successful = filter_data(metrics_list, {'build_success': True})
            if len(successful) > 1:
                # For simplicity, use top results as Pareto approximation
                # In practice, would use proper Pareto frontier calculation
                sorted_results = sorted(successful, 
                                      key=lambda m: m.get_efficiency_score() or 0, 
                                      reverse=True)
                pareto_approx = sorted_results[:min(5, len(sorted_results))]
                export_pareto_frontier(pareto_approx, output_dir / "pareto_frontier.json")
        
        logger.info(f"DSE analysis exported to: {output_dir}")
        return True
        
    except Exception as e:
        logger.error(f"DSE analysis export failed: {e}")
        return False