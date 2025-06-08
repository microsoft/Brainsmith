"""
Enhanced result classes for Brainsmith platform with comprehensive metrics.

This module provides structured result objects that support both simple
compilation and advanced design space exploration workflows.
"""

import os
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from pathlib import Path

from .metrics import BrainsmithMetrics
from .design_space import DesignPoint


@dataclass
class BrainsmithResult:
    """Enhanced result object with comprehensive metrics and metadata."""
    
    # Core result information
    success: bool = False
    output_dir: str = ""
    build_time: float = 0.0
    
    # Enhanced metrics
    metrics: Optional[BrainsmithMetrics] = None
    design_point: Optional[DesignPoint] = None
    
    # Build artifacts
    final_model_path: Optional[str] = None
    stitched_ip_path: Optional[str] = None
    reports_dir: Optional[str] = None
    
    # Error tracking
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    # Build metadata
    blueprint_name: str = ""
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    # Configuration used
    config_data: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize timing if not set."""
        if self.start_time is None:
            self.start_time = datetime.now()
    
    def add_error(self, error: str):
        """Add an error message."""
        self.errors.append(error)
    
    def add_warning(self, warning: str):
        """Add a warning message."""
        self.warnings.append(warning)
    
    def start_timing(self):
        """Start timing the build."""
        self.start_time = datetime.now()
    
    def end_timing(self):
        """End timing the build."""
        self.end_time = datetime.now()
        if self.start_time:
            self.build_time = (self.end_time - self.start_time).total_seconds()
    
    def get_artifacts(self) -> Dict[str, str]:
        """Get dictionary of build artifacts."""
        artifacts = {}
        
        if self.final_model_path and os.path.exists(self.final_model_path):
            artifacts['final_model'] = self.final_model_path
        
        if self.stitched_ip_path and os.path.exists(self.stitched_ip_path):
            artifacts['stitched_ip'] = self.stitched_ip_path
        
        if self.reports_dir and os.path.exists(self.reports_dir):
            artifacts['reports'] = self.reports_dir
        
        # Look for common FINN output files
        if self.output_dir and os.path.exists(self.output_dir):
            common_files = [
                'output.onnx',
                'estimate_report.json',
                'rtlsim_performance.json',
                'folding_config.json'
            ]
            
            for filename in common_files:
                filepath = os.path.join(self.output_dir, filename)
                if os.path.exists(filepath):
                    artifacts[filename.replace('.', '_')] = filepath
        
        return artifacts
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of the build results."""
        summary = {
            'success': self.success,
            'build_time': self.build_time,
            'output_dir': self.output_dir,
            'blueprint': self.blueprint_name,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }
        
        # Add metrics summary if available
        if self.metrics:
            if self.metrics.performance.throughput_ops_sec:
                summary['throughput_ops_sec'] = self.metrics.performance.throughput_ops_sec
            
            if self.metrics.resources.lut_utilization_percent:
                summary['lut_utilization_percent'] = self.metrics.resources.lut_utilization_percent
            
            if self.metrics.resources.dsp_utilization_percent:
                summary['dsp_utilization_percent'] = self.metrics.resources.dsp_utilization_percent
            
            if self.metrics.resources.estimated_power_w:
                summary['estimated_power_w'] = self.metrics.resources.estimated_power_w
        
        # Add design point summary if available
        if self.design_point:
            summary['design_point_hash'] = getattr(self.design_point, 'hash', 'unknown')
            summary['parameters'] = self.design_point.parameters
        
        return summary
    
    def to_research_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format for research data export."""
        research_data = {
            'build_id': self.metrics.build_id if self.metrics else f"build_{int(time.time())}",
            'success': self.success,
            'build_time': self.build_time,
            'blueprint': self.blueprint_name,
            'config': self.config_data,
            'artifacts': self.get_artifacts()
        }
        
        # Add comprehensive metrics if available
        if self.metrics:
            research_data.update(self.metrics.to_research_dataset())
        
        # Add design point data if available
        if self.design_point:
            research_data['design_point'] = self.design_point.to_dict()
        
        # Add error/warning info
        research_data['errors'] = self.errors
        research_data['warnings'] = self.warnings
        
        return research_data
    
    def save_result(self, filepath: Optional[str] = None) -> str:
        """Save result to JSON file."""
        if filepath is None:
            filepath = os.path.join(self.output_dir, "brainsmith_result.json")
        
        result_data = {
            'success': self.success,
            'build_time': self.build_time,
            'output_dir': self.output_dir,
            'blueprint_name': self.blueprint_name,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'config_data': self.config_data,
            'errors': self.errors,
            'warnings': self.warnings,
            'artifacts': self.get_artifacts(),
            'summary': self.get_summary()
        }
        
        # Add metrics if available
        if self.metrics:
            result_data['metrics'] = self.metrics.to_dict()
        
        # Add design point if available
        if self.design_point:
            result_data['design_point'] = self.design_point.to_dict()
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
        
        return filepath
    
    def save(self, filepath: Union[str, Path]):
        """Save result to file (compatibility method)."""
        return self.save_result(str(filepath))
    
    @classmethod
    def load_result(cls, filepath: str) -> 'BrainsmithResult':
        """Load result from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        result = cls(
            success=data.get('success', False),
            build_time=data.get('build_time', 0.0),
            output_dir=data.get('output_dir', ''),
            blueprint_name=data.get('blueprint_name', ''),
            config_data=data.get('config_data', {}),
            errors=data.get('errors', []),
            warnings=data.get('warnings', [])
        )
        
        # Restore timestamps
        if data.get('start_time'):
            result.start_time = datetime.fromisoformat(data['start_time'])
        if data.get('end_time'):
            result.end_time = datetime.fromisoformat(data['end_time'])
        
        # Restore metrics if available
        if 'metrics' in data:
            result.metrics = BrainsmithMetrics.from_dict(data['metrics'])
        
        # Restore design point if available
        if 'design_point' in data:
            result.design_point = DesignPoint.from_dict(data['design_point'])
        
        return result


@dataclass
class ParameterSweepResult:
    """Result object for parameter sweep operations."""
    
    results: List[BrainsmithResult] = field(default_factory=list)
    sweep_parameters: Dict[str, List[Any]] = field(default_factory=dict)
    total_time: float = 0.0
    success_count: int = 0
    
    def __post_init__(self):
        """Calculate derived metrics."""
        self.success_count = sum(1 for r in self.results if r.success)
    
    def get_best_result(self, metric: str = "throughput_ops_sec") -> Optional[BrainsmithResult]:
        """Get the best result according to specified metric."""
        successful_results = [r for r in self.results if r.success and r.metrics]
        
        if not successful_results:
            return None
        
        def get_metric_value(result: BrainsmithResult) -> float:
            if not result.metrics:
                return 0.0
            
            if metric == "throughput_ops_sec":
                return result.metrics.performance.throughput_ops_sec or 0.0
            elif metric == "efficiency":
                return result.metrics.custom_metrics.get("throughput_per_lut", 0.0)
            elif metric == "power_efficiency":
                return result.metrics.custom_metrics.get("throughput_per_watt", 0.0)
            elif metric == "resource_usage":
                # Lower is better for resource usage
                return -(result.metrics.resources.get_total_resource_score() or 1.0)
            else:
                return result.metrics.custom_metrics.get(metric, 0.0)
        
        return max(successful_results, key=get_metric_value)
    
    def get_pareto_frontier(self, objectives: List[str], 
                           directions: List[str]) -> List[BrainsmithResult]:
        """
        Get Pareto frontier for multi-objective optimization.
        
        Args:
            objectives: List of metric names
            directions: List of "maximize" or "minimize" for each objective
        """
        successful_results = [r for r in self.results if r.success and r.metrics]
        
        if not successful_results:
            return []
        
        def get_objective_values(result: BrainsmithResult) -> List[float]:
            values = []
            for obj in objectives:
                if obj == "throughput_ops_sec":
                    val = result.metrics.performance.throughput_ops_sec or 0.0
                elif obj == "lut_utilization":
                    val = result.metrics.resources.lut_utilization_percent or 0.0
                elif obj == "power":
                    val = result.metrics.resources.estimated_power_w or 0.0
                else:
                    val = result.metrics.custom_metrics.get(obj, 0.0)
                values.append(val)
            return values
        
        def dominates(a_values: List[float], b_values: List[float]) -> bool:
            """Check if a dominates b in Pareto sense."""
            better_in_any = False
            for i, (a_val, b_val, direction) in enumerate(zip(a_values, b_values, directions)):
                if direction == "maximize":
                    if a_val < b_val:
                        return False
                    elif a_val > b_val:
                        better_in_any = True
                else:  # minimize
                    if a_val > b_val:
                        return False
                    elif a_val < b_val:
                        better_in_any = True
            return better_in_any
        
        # Find Pareto frontier
        pareto_results = []
        for result in successful_results:
            result_values = get_objective_values(result)
            is_dominated = False
            
            for other_result in successful_results:
                if other_result == result:
                    continue
                other_values = get_objective_values(other_result)
                if dominates(other_values, result_values):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_results.append(result)
        
        return pareto_results
    
    def export_comparison_data(self) -> Dict[str, Any]:
        """Export data for comparison and analysis."""
        comparison_data = {
            'sweep_parameters': self.sweep_parameters,
            'total_time': self.total_time,
            'success_count': self.success_count,
            'total_count': len(self.results),
            'success_rate': self.success_count / len(self.results) if self.results else 0.0,
            'results': []
        }
        
        for result in self.results:
            result_data = result.to_research_dict()
            comparison_data['results'].append(result_data)
        
        return comparison_data
    
    def to_csv(self, filepath: str):
        """Export results to CSV file for analysis."""
        import csv
        
        if not self.results:
            return
        
        # Determine columns from first successful result
        successful_results = [r for r in self.results if r.success and r.metrics]
        if not successful_results:
            return
        
        sample_result = successful_results[0]
        research_data = sample_result.to_research_dict()
        
        # Define column order
        columns = [
            'build_id', 'success', 'build_time',
            'throughput_ops_sec', 'lut_utilization_percent', 'dsp_utilization_percent',
            'estimated_power_w', 'target_achievement_ratio', 'throughput_per_lut'
        ]
        
        # Add parameter columns
        if sample_result.design_point:
            for param_name in sample_result.design_point.parameters.keys():
                columns.append(f"param_{param_name}")
        
        with open(filepath, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=columns)
            writer.writeheader()
            
            for result in self.results:
                row_data = {}
                research_data = result.to_research_dict()
                
                # Basic columns
                for col in columns:
                    if col.startswith('param_'):
                        param_name = col[6:]  # Remove 'param_' prefix
                        if result.design_point and param_name in result.design_point.parameters:
                            row_data[col] = result.design_point.parameters[param_name]
                        else:
                            row_data[col] = None
                    else:
                        row_data[col] = research_data.get(col)
                
                writer.writerow(row_data)


@dataclass 
class DSEResult:
    """Result object for design space exploration."""
    
    results: List[BrainsmithResult] = field(default_factory=list)
    design_space_info: Dict[str, Any] = field(default_factory=dict)
    exploration_time: float = 0.0
    strategy_used: str = ""
    analysis: Dict[str, Any] = field(default_factory=dict)  # Add analysis field for compatibility
    
    # Analysis results (to be populated by analysis tools)
    pareto_frontier: Optional[List[BrainsmithResult]] = None
    best_configurations: Dict[str, BrainsmithResult] = field(default_factory=dict)
    analysis_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_successful_results(self) -> List[BrainsmithResult]:
        """Get only successful build results."""
        return [r for r in self.results if r.success]
    
    def get_coverage_report(self) -> Dict[str, Any]:
        """Generate design space coverage report."""
        successful_results = self.get_successful_results()
        
        coverage = {
            'total_points_evaluated': len(self.results),
            'successful_points': len(successful_results),
            'success_rate': len(successful_results) / len(self.results) if self.results else 0.0,
            'exploration_time': self.exploration_time,
            'strategy': self.strategy_used
        }
        
        # Parameter coverage analysis
        if successful_results and successful_results[0].design_point:
            param_coverage = {}
            for result in successful_results:
                for param_name, param_value in result.design_point.parameters.items():
                    if param_name not in param_coverage:
                        param_coverage[param_name] = set()
                    param_coverage[param_name].add(param_value)
            
            coverage['parameter_coverage'] = {
                name: len(values) for name, values in param_coverage.items()
            }
        
        return coverage
    
    def save(self, filepath: Union[str, Path]):
        """Save DSE result to file."""
        result_data = {
            'results': [r.to_research_dict() for r in self.results],
            'design_space_info': self.design_space_info,
            'exploration_time': self.exploration_time,
            'strategy_used': self.strategy_used,
            'analysis': self.analysis,
            'analysis_metadata': self.analysis_metadata,
            'coverage_report': self.get_coverage_report()
        }
        
        if self.pareto_frontier:
            result_data['pareto_frontier'] = [r.to_research_dict() for r in self.pareto_frontier]
        
        if self.best_configurations:
            result_data['best_configurations'] = {
                metric: result.to_research_dict() 
                for metric, result in self.best_configurations.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(result_data, f, indent=2, default=str)
    
    def export_research_dataset(self, filepath: str):
        """Export comprehensive research dataset."""
        dataset = {
            'design_space_info': self.design_space_info,
            'exploration_metadata': {
                'strategy': self.strategy_used,
                'exploration_time': self.exploration_time,
                'total_evaluations': len(self.results),
                'successful_evaluations': len(self.get_successful_results())
            },
            'coverage_report': self.get_coverage_report(),
            'analysis': self.analysis,
            'results': [r.to_research_dict() for r in self.results]
        }
        
        if self.pareto_frontier:
            dataset['pareto_frontier'] = [r.to_research_dict() for r in self.pareto_frontier]
        
        if self.best_configurations:
            dataset['best_configurations'] = {
                metric: result.to_research_dict() 
                for metric, result in self.best_configurations.items()
            }
        
        with open(filepath, 'w') as f:
            json.dump(dataset, f, indent=2, default=str)