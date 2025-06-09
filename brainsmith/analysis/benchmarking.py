"""
Benchmarking Framework
Reference design database and comparative analysis tools.
"""

import json
import logging
from typing import Dict, List, Any, Optional
import numpy as np
from pathlib import Path
from datetime import datetime

from .models import (
    BenchmarkResult, BenchmarkCategory, AnalysisConfiguration,
    ParetoSolution, RankedSolution
)

logger = logging.getLogger(__name__)


class ReferenceDesignDB:
    """Database of reference designs for benchmarking."""
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize reference design database."""
        self.db_path = db_path
        self.designs = {}
        self.categories = {}
        
        if db_path and Path(db_path).exists():
            self.load_database()
        else:
            self._initialize_default_database()
    
    def load_database(self) -> None:
        """Load reference designs from database file."""
        try:
            with open(self.db_path, 'r') as f:
                data = json.load(f)
                self.designs = data.get('designs', {})
                self.categories = data.get('categories', {})
            logger.info(f"Loaded reference database with {len(self.designs)} designs")
        except Exception as e:
            logger.error(f"Failed to load reference database: {e}")
            self._initialize_default_database()
    
    def save_database(self) -> None:
        """Save reference designs to database file."""
        if not self.db_path:
            return
        
        try:
            data = {
                'designs': self.designs,
                'categories': self.categories,
                'last_updated': datetime.now().isoformat()
            }
            
            with open(self.db_path, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Saved reference database to {self.db_path}")
        except Exception as e:
            logger.error(f"Failed to save reference database: {e}")
    
    def _initialize_default_database(self) -> None:
        """Initialize database with default reference designs."""
        
        # CNN Inference reference designs
        self.categories[BenchmarkCategory.CNN_INFERENCE.value] = {
            'description': 'Convolutional Neural Network inference designs',
            'metrics': ['throughput_fps', 'power_watts', 'latency_ms', 'accuracy_percent', 'dsp_utilization', 'lut_utilization']
        }
        
        # Default CNN designs (representative examples)
        cnn_designs = [
            {
                'name': 'ResNet50_Reference',
                'category': BenchmarkCategory.CNN_INFERENCE.value,
                'metrics': {
                    'throughput_fps': 120.0,
                    'power_watts': 15.5,
                    'latency_ms': 8.3,
                    'accuracy_percent': 76.1,
                    'dsp_utilization': 85.2,
                    'lut_utilization': 78.9
                },
                'parameters': {
                    'pe_parallelism': 16,
                    'memory_bandwidth': 256,
                    'precision': 'int8'
                },
                'source': 'industry_baseline'
            },
            {
                'name': 'MobileNet_Reference',
                'category': BenchmarkCategory.CNN_INFERENCE.value,
                'metrics': {
                    'throughput_fps': 250.0,
                    'power_watts': 8.2,
                    'latency_ms': 4.0,
                    'accuracy_percent': 71.8,
                    'dsp_utilization': 65.3,
                    'lut_utilization': 45.7
                },
                'parameters': {
                    'pe_parallelism': 8,
                    'memory_bandwidth': 128,
                    'precision': 'int8'
                },
                'source': 'industry_baseline'
            },
            {
                'name': 'EfficientNet_Reference',
                'category': BenchmarkCategory.CNN_INFERENCE.value,
                'metrics': {
                    'throughput_fps': 95.0,
                    'power_watts': 12.1,
                    'latency_ms': 10.5,
                    'accuracy_percent': 82.3,
                    'dsp_utilization': 92.1,
                    'lut_utilization': 88.4
                },
                'parameters': {
                    'pe_parallelism': 24,
                    'memory_bandwidth': 512,
                    'precision': 'int8'
                },
                'source': 'industry_baseline'
            }
        ]
        
        # Store designs
        for design in cnn_designs:
            design_id = f"{design['category']}_{design['name']}"
            self.designs[design_id] = design
        
        # Signal Processing reference designs
        self.categories[BenchmarkCategory.SIGNAL_PROCESSING.value] = {
            'description': 'Digital signal processing designs',
            'metrics': ['throughput_msps', 'power_watts', 'latency_us', 'snr_db', 'dsp_utilization', 'bram_utilization']
        }
        
        # Add more categories as needed
        logger.info("Initialized default reference database")
    
    def add_design(self, 
                   design_name: str,
                   category: BenchmarkCategory,
                   metrics: Dict[str, float],
                   parameters: Dict[str, Any],
                   source: str = "user_contributed") -> None:
        """Add a new reference design to the database."""
        
        design_id = f"{category.value}_{design_name}"
        
        design_data = {
            'name': design_name,
            'category': category.value,
            'metrics': metrics,
            'parameters': parameters,
            'source': source,
            'added_date': datetime.now().isoformat()
        }
        
        self.designs[design_id] = design_data
        logger.info(f"Added reference design: {design_id}")
    
    def get_designs_by_category(self, category: BenchmarkCategory) -> List[Dict[str, Any]]:
        """Get all reference designs for a category."""
        return [
            design for design_id, design in self.designs.items()
            if design.get('category') == category.value
        ]
    
    def get_design(self, design_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific reference design."""
        return self.designs.get(design_id)


class IndustryBenchmark:
    """Industry benchmark standards and comparisons."""
    
    @staticmethod
    def get_industry_standards(category: BenchmarkCategory) -> Dict[str, Any]:
        """Get industry standard benchmarks for a category."""
        
        standards = {
            BenchmarkCategory.CNN_INFERENCE: {
                'throughput_fps': {
                    'excellent': 200.0,
                    'good': 100.0,
                    'average': 50.0,
                    'poor': 25.0
                },
                'power_watts': {
                    'excellent': 5.0,
                    'good': 10.0,
                    'average': 20.0,
                    'poor': 40.0
                },
                'latency_ms': {
                    'excellent': 2.0,
                    'good': 5.0,
                    'average': 10.0,
                    'poor': 20.0
                },
                'accuracy_percent': {
                    'excellent': 85.0,
                    'good': 80.0,
                    'average': 75.0,
                    'poor': 70.0
                }
            },
            BenchmarkCategory.SIGNAL_PROCESSING: {
                'throughput_msps': {
                    'excellent': 1000.0,
                    'good': 500.0,
                    'average': 100.0,
                    'poor': 50.0
                },
                'power_watts': {
                    'excellent': 2.0,
                    'good': 5.0,
                    'average': 10.0,
                    'poor': 20.0
                },
                'latency_us': {
                    'excellent': 1.0,
                    'good': 5.0,
                    'average': 10.0,
                    'poor': 50.0
                }
            }
        }
        
        return standards.get(category, {})


class BenchmarkingEngine:
    """Engine for benchmarking designs against reference database."""
    
    def __init__(self, 
                 reference_db: Optional[ReferenceDesignDB] = None,
                 configuration: Optional[AnalysisConfiguration] = None):
        """Initialize benchmarking engine."""
        self.reference_db = reference_db or ReferenceDesignDB()
        self.config = configuration or AnalysisConfiguration()
        
    def benchmark_design(self, 
                        design: Union[ParetoSolution, RankedSolution],
                        category: BenchmarkCategory,
                        design_id: Optional[str] = None) -> BenchmarkResult:
        """
        Benchmark a design against reference designs in the category.
        
        Args:
            design: Design to benchmark
            category: Benchmark category
            design_id: Optional identifier for the design
            
        Returns:
            BenchmarkResult: Comprehensive benchmark analysis
        """
        
        design_id = design_id or f"design_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Extract performance metrics from design
        design_metrics = self._extract_design_metrics(design)
        
        # Get reference designs for category
        reference_designs = self.reference_db.get_designs_by_category(category)
        
        if not reference_designs:
            logger.warning(f"No reference designs found for category {category.value}")
            return self._create_empty_benchmark_result(design_id, category, design_metrics)
        
        # Calculate relative performance
        relative_performance = self._calculate_relative_performance(
            design_metrics, reference_designs
        )
        
        # Calculate percentile rankings
        percentile_ranking = self._calculate_percentile_ranking(
            design_metrics, reference_designs
        )
        
        # Get industry comparison
        industry_comparison = self._compare_with_industry_standards(
            design_metrics, category
        )
        
        # Generate recommendation
        recommendation = self._generate_benchmark_recommendation(
            relative_performance, percentile_ranking, industry_comparison
        )
        
        return BenchmarkResult(
            design_id=design_id,
            benchmark_category=category,
            reference_designs=reference_designs,
            performance_metrics=design_metrics,
            relative_performance=relative_performance,
            percentile_ranking=percentile_ranking,
            industry_comparison=industry_comparison,
            recommendation=recommendation
        )
    
    def _extract_design_metrics(self, 
                               design: Union[ParetoSolution, RankedSolution]) -> Dict[str, float]:
        """Extract performance metrics from design."""
        
        if isinstance(design, RankedSolution):
            # Extract from RankedSolution
            if hasattr(design.solution, 'objective_values'):
                objective_values = design.solution.objective_values
            else:
                objective_values = []
            
            # Try to get metric names from selection criteria
            if hasattr(design, 'selection_criteria') and hasattr(design.selection_criteria, 'objectives'):
                objective_names = design.selection_criteria.objectives
            else:
                objective_names = [f"objective_{i}" for i in range(len(objective_values))]
            
        elif isinstance(design, ParetoSolution):
            # Extract from ParetoSolution
            objective_values = design.objective_values
            objective_names = [f"objective_{i}" for i in range(len(objective_values))]
            
        else:
            # Unknown design type
            return {}
        
        # Create metrics dictionary
        metrics = {}
        for i, (name, value) in enumerate(zip(objective_names, objective_values)):
            metrics[name] = float(value)
        
        return metrics
    
    def _calculate_relative_performance(self, 
                                      design_metrics: Dict[str, float],
                                      reference_designs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate relative performance compared to reference designs."""
        
        relative_performance = {}
        
        for metric_name, metric_value in design_metrics.items():
            # Collect reference values for this metric
            reference_values = []
            for ref_design in reference_designs:
                if metric_name in ref_design.get('metrics', {}):
                    reference_values.append(ref_design['metrics'][metric_name])
            
            if not reference_values:
                continue
            
            # Calculate relative performance
            ref_mean = np.mean(reference_values)
            ref_best = self._get_best_reference_value(metric_name, reference_values)
            
            # Ratio to mean
            ratio_to_mean = metric_value / ref_mean if ref_mean != 0 else 1.0
            
            # Ratio to best
            ratio_to_best = metric_value / ref_best if ref_best != 0 else 1.0
            
            relative_performance[metric_name] = {
                'ratio_to_mean': ratio_to_mean,
                'ratio_to_best': ratio_to_best,
                'reference_mean': ref_mean,
                'reference_best': ref_best,
                'design_value': metric_value
            }
        
        # Flatten for easier access
        flat_performance = {}
        for metric_name, perf_data in relative_performance.items():
            flat_performance[f"{metric_name}_ratio_to_mean"] = perf_data['ratio_to_mean']
            flat_performance[f"{metric_name}_ratio_to_best"] = perf_data['ratio_to_best']
        
        return flat_performance
    
    def _get_best_reference_value(self, metric_name: str, values: List[float]) -> float:
        """Get best reference value (max for benefit metrics, min for cost metrics)."""
        
        # Simple heuristic: metrics with 'power', 'latency', 'delay' are cost metrics (lower is better)
        cost_keywords = ['power', 'latency', 'delay', 'energy', 'area', 'utilization']
        
        is_cost_metric = any(keyword in metric_name.lower() for keyword in cost_keywords)
        
        if is_cost_metric:
            return min(values)  # Lower is better
        else:
            return max(values)  # Higher is better
    
    def _calculate_percentile_ranking(self, 
                                    design_metrics: Dict[str, float],
                                    reference_designs: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate percentile ranking for each metric."""
        
        percentile_ranking = {}
        
        for metric_name, metric_value in design_metrics.items():
            # Collect reference values
            reference_values = []
            for ref_design in reference_designs:
                if metric_name in ref_design.get('metrics', {}):
                    reference_values.append(ref_design['metrics'][metric_name])
            
            if not reference_values:
                continue
            
            # Add design value to reference values for percentile calculation
            all_values = reference_values + [metric_value]
            all_values.sort()
            
            # Find percentile rank
            design_rank = all_values.index(metric_value) + 1
            percentile = (design_rank / len(all_values)) * 100
            
            # Adjust for cost metrics (invert percentile)
            cost_keywords = ['power', 'latency', 'delay', 'energy', 'area', 'utilization']
            is_cost_metric = any(keyword in metric_name.lower() for keyword in cost_keywords)
            
            if is_cost_metric:
                percentile = 100 - percentile  # Invert for cost metrics
            
            percentile_ranking[metric_name] = percentile
        
        return percentile_ranking
    
    def _compare_with_industry_standards(self, 
                                       design_metrics: Dict[str, float],
                                       category: BenchmarkCategory) -> Dict[str, Any]:
        """Compare design with industry standards."""
        
        industry_standards = IndustryBenchmark.get_industry_standards(category)
        comparison = {}
        
        for metric_name, metric_value in design_metrics.items():
            if metric_name in industry_standards:
                standards = industry_standards[metric_name]
                
                # Determine performance level
                if metric_value >= standards.get('excellent', float('inf')):
                    level = 'excellent'
                elif metric_value >= standards.get('good', float('inf')):
                    level = 'good'
                elif metric_value >= standards.get('average', float('inf')):
                    level = 'average'
                else:
                    level = 'poor'
                
                comparison[metric_name] = {
                    'level': level,
                    'value': metric_value,
                    'standards': standards
                }
        
        return comparison
    
    def _generate_benchmark_recommendation(self, 
                                         relative_performance: Dict[str, float],
                                         percentile_ranking: Dict[str, float],
                                         industry_comparison: Dict[str, Any]) -> str:
        """Generate benchmark recommendation."""
        
        recommendations = []
        
        # Overall performance assessment
        if percentile_ranking:
            avg_percentile = np.mean(list(percentile_ranking.values()))
            
            if avg_percentile >= 80:
                recommendations.append("Excellent overall performance compared to references.")
            elif avg_percentile >= 60:
                recommendations.append("Good performance with some optimization opportunities.")
            elif avg_percentile >= 40:
                recommendations.append("Average performance - consider targeted improvements.")
            else:
                recommendations.append("Below-average performance - significant optimization needed.")
        
        # Specific metric recommendations
        for metric, percentile in percentile_ranking.items():
            if percentile < 25:
                recommendations.append(f"Focus on improving {metric} (bottom quartile performance).")
        
        # Industry comparison insights
        excellent_metrics = [
            metric for metric, comparison in industry_comparison.items()
            if comparison.get('level') == 'excellent'
        ]
        
        if excellent_metrics:
            recommendations.append(f"Industry-leading performance in: {', '.join(excellent_metrics)}")
        
        poor_metrics = [
            metric for metric, comparison in industry_comparison.items()
            if comparison.get('level') == 'poor'
        ]
        
        if poor_metrics:
            recommendations.append(f"Below industry standards in: {', '.join(poor_metrics)}")
        
        return ". ".join(recommendations) if recommendations else "No specific recommendations available."
    
    def _create_empty_benchmark_result(self, 
                                     design_id: str,
                                     category: BenchmarkCategory,
                                     design_metrics: Dict[str, float]) -> BenchmarkResult:
        """Create empty benchmark result when no references available."""
        
        return BenchmarkResult(
            design_id=design_id,
            benchmark_category=category,
            reference_designs=[],
            performance_metrics=design_metrics,
            relative_performance={},
            percentile_ranking={},
            industry_comparison={},
            recommendation="No reference designs available for comparison."
        )
    
    def add_design_to_reference(self, 
                               design: Union[ParetoSolution, RankedSolution],
                               category: BenchmarkCategory,
                               design_name: str,
                               source: str = "user_contributed") -> None:
        """Add a design to the reference database."""
        
        design_metrics = self._extract_design_metrics(design)
        
        # Extract design parameters if available
        if isinstance(design, RankedSolution) and hasattr(design.solution, 'design_parameters'):
            parameters = design.solution.design_parameters
        elif isinstance(design, ParetoSolution):
            parameters = design.design_parameters
        else:
            parameters = {}
        
        self.reference_db.add_design(
            design_name=design_name,
            category=category,
            metrics=design_metrics,
            parameters=parameters,
            source=source
        )
    
    def save_reference_database(self) -> None:
        """Save the reference database."""
        self.reference_db.save_database()