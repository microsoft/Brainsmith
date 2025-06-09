"""
Data models for the Performance Analysis Framework
Defines core data structures used throughout the analysis system.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime

# Import from selection framework
try:
    from ..selection.models import RankedSolution, ParetoSolution
except ImportError:
    # Fallback definitions if selection framework not available
    @dataclass
    class RankedSolution:
        solution: Any
        rank: int
        score: float
        
    @dataclass
    class ParetoSolution:
        design_parameters: Dict[str, Any]
        objective_values: List[float]
        constraint_violations: List[float] = field(default_factory=list)
        metadata: Dict[str, Any] = field(default_factory=dict)


class AnalysisType(Enum):
    """Types of performance analysis."""
    DESCRIPTIVE = "descriptive"
    COMPARATIVE = "comparative"
    PREDICTIVE = "predictive"
    BENCHMARKING = "benchmarking"
    STATISTICAL = "statistical"
    CORRELATION = "correlation"
    TREND = "trend"
    OUTLIER = "outlier"


class DistributionType(Enum):
    """Statistical distribution types."""
    NORMAL = "normal"
    LOGNORMAL = "lognormal"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    BETA = "beta"
    UNIFORM = "uniform"
    WEIBULL = "weibull"
    STUDENT_T = "student_t"


class BenchmarkCategory(Enum):
    """Categories for benchmarking."""
    CNN_INFERENCE = "cnn_inference"
    TRANSFORMER = "transformer"
    SIGNAL_PROCESSING = "signal_processing"
    COMPUTER_VISION = "computer_vision"
    NATURAL_LANGUAGE = "natural_language"
    CUSTOM = "custom"


class PredictionModel(Enum):
    """Types of prediction models."""
    LINEAR_REGRESSION = "linear_regression"
    POLYNOMIAL_REGRESSION = "polynomial_regression"
    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    NEURAL_NETWORK = "neural_network"
    GAUSSIAN_PROCESS = "gaussian_process"
    ENSEMBLE = "ensemble"


@dataclass
class PerformanceData:
    """Container for performance data."""
    metric_name: str
    values: np.ndarray
    units: str = ""
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate performance data after initialization."""
        if isinstance(self.values, list):
            self.values = np.array(self.values)
        
        if self.values.size == 0:
            raise ValueError("Performance data cannot be empty")
    
    @property
    def statistics(self) -> Dict[str, float]:
        """Get basic statistics for the performance data."""
        return {
            'count': len(self.values),
            'mean': float(np.mean(self.values)),
            'std': float(np.std(self.values)),
            'min': float(np.min(self.values)),
            'max': float(np.max(self.values)),
            'median': float(np.median(self.values)),
            'q25': float(np.percentile(self.values, 25)),
            'q75': float(np.percentile(self.values, 75))
        }


@dataclass
class StatisticalSummary:
    """Statistical summary of performance data."""
    metric_name: str
    count: int
    mean: float
    std: float
    min_value: float
    max_value: float
    median: float
    q25: float
    q75: float
    skewness: float = 0.0
    kurtosis: float = 0.0
    coefficient_of_variation: float = 0.0
    
    def __post_init__(self):
        """Calculate derived statistics."""
        if self.mean != 0:
            self.coefficient_of_variation = self.std / abs(self.mean)


@dataclass
class DistributionAnalysis:
    """Results of distribution analysis."""
    metric_name: str
    best_fit_distribution: DistributionType
    distribution_parameters: Dict[str, float]
    goodness_of_fit: float  # p-value or R-squared
    confidence_level: float
    tested_distributions: List[DistributionType] = field(default_factory=list)
    fit_scores: Dict[DistributionType, float] = field(default_factory=dict)
    
    def get_distribution_info(self) -> Dict[str, Any]:
        """Get comprehensive distribution information."""
        return {
            'distribution': self.best_fit_distribution.value,
            'parameters': self.distribution_parameters,
            'goodness_of_fit': self.goodness_of_fit,
            'confidence': self.confidence_level,
            'alternatives': {dist.value: score for dist, score in self.fit_scores.items()}
        }


@dataclass
class ConfidenceInterval:
    """Confidence interval for a metric."""
    metric_name: str
    confidence_level: float
    lower_bound: float
    upper_bound: float
    mean: float
    std_error: float
    method: str = "t_distribution"  # bootstrap, t_distribution, normal
    
    @property
    def width(self) -> float:
        """Get confidence interval width."""
        return self.upper_bound - self.lower_bound
    
    @property
    def margin_of_error(self) -> float:
        """Get margin of error."""
        return self.width / 2.0


@dataclass
class HypothesisTest:
    """Results of hypothesis testing."""
    test_name: str
    null_hypothesis: str
    alternative_hypothesis: str
    test_statistic: float
    p_value: float
    critical_value: float
    significance_level: float
    reject_null: bool
    confidence_interval: Optional[ConfidenceInterval] = None
    effect_size: Optional[float] = None
    
    @property
    def is_significant(self) -> bool:
        """Check if result is statistically significant."""
        return self.p_value < self.significance_level


@dataclass
class CorrelationAnalysis:
    """Results of correlation analysis."""
    metric_pairs: List[Tuple[str, str]]
    correlation_matrix: np.ndarray
    correlation_method: str = "pearson"  # pearson, spearman, kendall
    p_values: Optional[np.ndarray] = None
    significance_level: float = 0.05
    significant_correlations: List[Tuple[str, str, float]] = field(default_factory=list)
    
    def get_strong_correlations(self, threshold: float = 0.7) -> List[Tuple[str, str, float]]:
        """Get correlations above threshold."""
        strong_corrs = []
        n = len(self.metric_pairs)
        
        for i in range(n):
            for j in range(i + 1, n):
                if i < len(self.correlation_matrix) and j < len(self.correlation_matrix[0]):
                    corr = self.correlation_matrix[i, j]
                    if abs(corr) >= threshold:
                        metric1 = self.metric_pairs[i][0] if isinstance(self.metric_pairs[i], tuple) else str(i)
                        metric2 = self.metric_pairs[j][0] if isinstance(self.metric_pairs[j], tuple) else str(j)
                        strong_corrs.append((metric1, metric2, corr))
        
        return strong_corrs


@dataclass
class OutlierDetection:
    """Results of outlier detection."""
    metric_name: str
    outlier_indices: List[int]
    outlier_values: np.ndarray
    detection_method: str
    threshold: float
    outlier_scores: np.ndarray
    total_samples: int
    
    @property
    def outlier_percentage(self) -> float:
        """Get percentage of outliers."""
        return (len(self.outlier_indices) / self.total_samples) * 100.0 if self.total_samples > 0 else 0.0
    
    @property
    def num_outliers(self) -> int:
        """Get number of outliers."""
        return len(self.outlier_indices)


@dataclass
class BenchmarkResult:
    """Results of benchmarking analysis."""
    design_id: str
    benchmark_category: BenchmarkCategory
    reference_designs: List[Dict[str, Any]]
    performance_metrics: Dict[str, float]
    relative_performance: Dict[str, float]  # ratio to best/mean reference
    percentile_ranking: Dict[str, float]    # percentile in reference distribution
    industry_comparison: Dict[str, Any]
    recommendation: str = ""
    
    def get_overall_ranking(self) -> float:
        """Get overall percentile ranking across all metrics."""
        if not self.percentile_ranking:
            return 0.0
        return np.mean(list(self.percentile_ranking.values()))
    
    def is_competitive(self, threshold: float = 50.0) -> bool:
        """Check if design is competitive (above threshold percentile)."""
        return self.get_overall_ranking() >= threshold


@dataclass
class PredictionResult:
    """Results of performance prediction."""
    target_metric: str
    predicted_value: float
    prediction_interval: ConfidenceInterval
    model_type: PredictionModel
    model_accuracy: float  # R-squared or similar
    feature_importance: Dict[str, float]
    training_size: int
    validation_score: float
    uncertainty: float = 0.0
    
    @property
    def prediction_quality(self) -> str:
        """Get qualitative assessment of prediction quality."""
        if self.model_accuracy >= 0.9:
            return "Excellent"
        elif self.model_accuracy >= 0.8:
            return "Good"
        elif self.model_accuracy >= 0.7:
            return "Fair"
        else:
            return "Poor"


@dataclass
class TrendAnalysis:
    """Results of trend analysis."""
    metric_name: str
    trend_direction: str  # "increasing", "decreasing", "stable", "cyclic"
    trend_strength: float  # 0-1, strength of trend
    trend_equation: str   # mathematical representation
    trend_parameters: Dict[str, float]
    future_predictions: List[Tuple[float, float]]  # (time, predicted_value)
    seasonality_detected: bool = False
    change_points: List[int] = field(default_factory=list)
    
    @property
    def is_significant_trend(self) -> bool:
        """Check if trend is statistically significant."""
        return self.trend_strength >= 0.3  # Threshold for significance


@dataclass
class AnalysisContext:
    """Context for performance analysis."""
    solutions: List[Union[ParetoSolution, RankedSolution]]
    performance_data: Dict[str, PerformanceData]
    analysis_types: List[AnalysisType]
    configuration: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def get_metric_values(self, metric_name: str) -> Optional[np.ndarray]:
        """Get values for a specific metric."""
        if metric_name in self.performance_data:
            return self.performance_data[metric_name].values
        
        # Try to extract from solutions
        values = []
        for solution in self.solutions:
            if hasattr(solution, 'solution') and hasattr(solution.solution, 'objective_values'):
                # RankedSolution
                values.extend(solution.solution.objective_values)
            elif hasattr(solution, 'objective_values'):
                # ParetoSolution
                values.extend(solution.objective_values)
        
        return np.array(values) if values else None
    
    def get_all_metrics(self) -> List[str]:
        """Get list of all available metrics."""
        metrics = list(self.performance_data.keys())
        
        # Add objective names if available
        if self.solutions and hasattr(self.solutions[0], 'solution'):
            # Check if we have selection criteria with objective names
            solution = self.solutions[0]
            if hasattr(solution, 'selection_criteria') and hasattr(solution.selection_criteria, 'objectives'):
                metrics.extend(solution.selection_criteria.objectives)
        
        return list(set(metrics))  # Remove duplicates


@dataclass
class PerformanceAnalysis:
    """Comprehensive performance analysis results."""
    analysis_id: str
    timestamp: datetime
    solutions_analyzed: int
    statistical_summary: Dict[str, StatisticalSummary]
    distribution_analysis: Dict[str, DistributionAnalysis]
    correlation_analysis: CorrelationAnalysis
    outlier_detection: Dict[str, OutlierDetection]
    hypothesis_tests: List[HypothesisTest]
    confidence_intervals: Dict[str, ConfidenceInterval]
    performance_insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get comprehensive summary for a specific metric."""
        summary = {}
        
        if metric_name in self.statistical_summary:
            summary['statistics'] = self.statistical_summary[metric_name]
        
        if metric_name in self.distribution_analysis:
            summary['distribution'] = self.distribution_analysis[metric_name].get_distribution_info()
        
        if metric_name in self.outlier_detection:
            summary['outliers'] = {
                'count': self.outlier_detection[metric_name].num_outliers,
                'percentage': self.outlier_detection[metric_name].outlier_percentage
            }
        
        if metric_name in self.confidence_intervals:
            ci = self.confidence_intervals[metric_name]
            summary['confidence_interval'] = {
                'lower': ci.lower_bound,
                'upper': ci.upper_bound,
                'level': ci.confidence_level,
                'width': ci.width
            }
        
        return summary if summary else None


@dataclass
class BenchmarkComparison:
    """Comprehensive benchmarking comparison results."""
    design_id: str
    benchmark_category: BenchmarkCategory
    benchmark_results: Dict[str, BenchmarkResult]
    industry_ranking: Dict[str, float]
    competitive_analysis: Dict[str, Any]
    improvement_opportunities: List[str] = field(default_factory=list)
    strengths: List[str] = field(default_factory=list)
    weaknesses: List[str] = field(default_factory=list)
    
    def get_overall_competitiveness(self) -> float:
        """Get overall competitiveness score."""
        if not self.benchmark_results:
            return 0.0
        
        rankings = [result.get_overall_ranking() for result in self.benchmark_results.values()]
        return np.mean(rankings)
    
    def get_top_metrics(self, n: int = 3) -> List[Tuple[str, float]]:
        """Get top N performing metrics."""
        metric_rankings = []
        
        for metric, result in self.benchmark_results.items():
            overall_ranking = result.get_overall_ranking()
            metric_rankings.append((metric, overall_ranking))
        
        metric_rankings.sort(key=lambda x: x[1], reverse=True)
        return metric_rankings[:n]


@dataclass
class PredictionAnalysis:
    """Comprehensive prediction analysis results."""
    analysis_id: str
    target_metrics: List[str]
    prediction_results: Dict[str, PredictionResult]
    model_comparison: Dict[str, Dict[str, float]]
    feature_analysis: Dict[str, Dict[str, float]]
    uncertainty_analysis: Dict[str, float]
    trend_analysis: Dict[str, TrendAnalysis]
    validation_metrics: Dict[str, float]
    
    def get_best_model(self, metric: str) -> Optional[PredictionModel]:
        """Get best performing model for a metric."""
        if metric not in self.prediction_results:
            return None
        return self.prediction_results[metric].model_type
    
    def get_prediction_confidence(self, metric: str) -> float:
        """Get prediction confidence for a metric."""
        if metric not in self.prediction_results:
            return 0.0
        return self.prediction_results[metric].model_accuracy


@dataclass
class AnalysisConfiguration:
    """Configuration for performance analysis."""
    analysis_types: List[AnalysisType] = field(default_factory=lambda: [AnalysisType.DESCRIPTIVE])
    confidence_level: float = 0.95
    significance_level: float = 0.05
    outlier_threshold: float = 2.0  # Number of standard deviations
    correlation_threshold: float = 0.3
    distribution_tests: List[DistributionType] = field(default_factory=lambda: [
        DistributionType.NORMAL, DistributionType.LOGNORMAL, DistributionType.EXPONENTIAL
    ])
    prediction_models: List[PredictionModel] = field(default_factory=lambda: [
        PredictionModel.LINEAR_REGRESSION, PredictionModel.RANDOM_FOREST
    ])
    enable_bootstrapping: bool = True
    bootstrap_samples: int = 1000
    random_seed: Optional[int] = None
    parallel_processing: bool = True
    max_workers: Optional[int] = None
    
    def validate(self) -> bool:
        """Validate configuration parameters."""
        if not (0 < self.confidence_level < 1):
            return False
        if not (0 < self.significance_level < 1):
            return False
        if self.outlier_threshold <= 0:
            return False
        if self.bootstrap_samples <= 0:
            return False
        return True


# Type aliases for better code readability
MetricValues = Dict[str, np.ndarray]
AnalysisResults = Dict[str, Any]
BenchmarkData = Dict[str, Any]
PredictionData = Dict[str, Union[float, np.ndarray]]