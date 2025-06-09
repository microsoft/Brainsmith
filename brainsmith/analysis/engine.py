"""
Core Performance Analysis Engine
Main orchestration engine for comprehensive performance analysis and evaluation.
"""

import time
import logging
from typing import Dict, List, Any, Optional, Union
import numpy as np
from datetime import datetime
import uuid

from .models import (
    AnalysisContext, AnalysisConfiguration, PerformanceAnalysis,
    StatisticalSummary, DistributionAnalysis, CorrelationAnalysis,
    OutlierDetection, HypothesisTest, ConfidenceInterval,
    PerformanceData, AnalysisType, PredictionAnalysis
)

logger = logging.getLogger(__name__)


class PerformanceMetrics:
    """Container for performance metrics calculation."""
    
    @staticmethod
    def calculate_basic_metrics(values: np.ndarray) -> Dict[str, float]:
        """Calculate basic statistical metrics."""
        if len(values) == 0:
            return {}
        
        return {
            'count': len(values),
            'mean': float(np.mean(values)),
            'std': float(np.std(values, ddof=1) if len(values) > 1 else 0),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'median': float(np.median(values)),
            'q25': float(np.percentile(values, 25)),
            'q75': float(np.percentile(values, 75)),
            'range': float(np.max(values) - np.min(values)),
            'iqr': float(np.percentile(values, 75) - np.percentile(values, 25))
        }
    
    @staticmethod
    def calculate_advanced_metrics(values: np.ndarray) -> Dict[str, float]:
        """Calculate advanced statistical metrics."""
        if len(values) < 2:
            return {}
        
        try:
            from scipy import stats
            
            skewness = float(stats.skew(values))
            kurtosis = float(stats.kurtosis(values))
            
        except ImportError:
            # Fallback calculation if scipy not available
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            
            if std > 0:
                normalized = (values - mean) / std
                skewness = float(np.mean(normalized ** 3))
                kurtosis = float(np.mean(normalized ** 4) - 3)
            else:
                skewness = 0.0
                kurtosis = 0.0
        
        mean = np.mean(values)
        cv = np.std(values, ddof=1) / abs(mean) if mean != 0 else 0.0
        
        return {
            'skewness': skewness,
            'kurtosis': kurtosis,
            'coefficient_of_variation': float(cv)
        }


class AnalysisResult:
    """Result container for performance analysis operations."""
    
    def __init__(self, 
                 analysis: PerformanceAnalysis,
                 analysis_time: float,
                 context: AnalysisContext):
        self.analysis = analysis
        self.analysis_time = analysis_time
        self.context = context
        self.timestamp = datetime.now()
    
    def get_metric_summary(self, metric_name: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific metric."""
        return self.analysis.get_metric_summary(metric_name)
    
    def get_insights(self) -> List[str]:
        """Get performance insights."""
        return self.analysis.performance_insights
    
    def get_recommendations(self) -> List[str]:
        """Get analysis recommendations."""
        return self.analysis.recommendations


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis engine.
    
    Provides statistical analysis, distribution fitting, correlation analysis,
    outlier detection, and hypothesis testing capabilities.
    """
    
    def __init__(self, configuration: Optional[AnalysisConfiguration] = None):
        """Initialize performance analyzer with configuration."""
        self.config = configuration or AnalysisConfiguration()
        self.analysis_history = []
        
        # Validate configuration
        if not self.config.validate():
            raise ValueError("Invalid analysis configuration")
        
        logger.info(f"Performance analyzer initialized with {len(self.config.analysis_types)} analysis types")
    
    def analyze_performance(self, 
                           context: AnalysisContext) -> AnalysisResult:
        """
        Perform comprehensive performance analysis.
        
        Args:
            context: Analysis context with solutions and data
            
        Returns:
            AnalysisResult: Comprehensive analysis results
        """
        start_time = time.time()
        
        # Validate context
        self._validate_context(context)
        
        try:
            # Initialize analysis components
            analysis_id = str(uuid.uuid4())
            timestamp = datetime.now()
            
            statistical_summary = {}
            distribution_analysis = {}
            outlier_detection = {}
            confidence_intervals = {}
            hypothesis_tests = []
            
            # Get all available metrics
            metrics = context.get_all_metrics()
            logger.info(f"Analyzing {len(metrics)} metrics for {len(context.solutions)} solutions")
            
            # Perform statistical analysis for each metric
            for metric_name in metrics:
                values = context.get_metric_values(metric_name)
                if values is not None and len(values) > 0:
                    
                    # Statistical summary
                    if AnalysisType.DESCRIPTIVE in self.config.analysis_types:
                        statistical_summary[metric_name] = self._calculate_statistical_summary(
                            metric_name, values
                        )
                    
                    # Distribution analysis
                    if AnalysisType.STATISTICAL in self.config.analysis_types:
                        distribution_analysis[metric_name] = self._analyze_distribution(
                            metric_name, values
                        )
                    
                    # Outlier detection
                    if AnalysisType.OUTLIER in self.config.analysis_types:
                        outlier_detection[metric_name] = self._detect_outliers(
                            metric_name, values
                        )
                    
                    # Confidence intervals
                    confidence_intervals[metric_name] = self._calculate_confidence_interval(
                        metric_name, values
                    )
            
            # Correlation analysis
            correlation_analysis = None
            if AnalysisType.CORRELATION in self.config.analysis_types and len(metrics) > 1:
                correlation_analysis = self._analyze_correlations(context, metrics)
            
            # Hypothesis tests
            if AnalysisType.COMPARATIVE in self.config.analysis_types:
                hypothesis_tests = self._perform_hypothesis_tests(context, metrics)
            
            # Generate insights and recommendations
            insights = self._generate_insights(
                statistical_summary, distribution_analysis, 
                correlation_analysis, outlier_detection
            )
            
            recommendations = self._generate_recommendations(
                statistical_summary, distribution_analysis,
                correlation_analysis, outlier_detection
            )
            
            # Create comprehensive analysis
            analysis = PerformanceAnalysis(
                analysis_id=analysis_id,
                timestamp=timestamp,
                solutions_analyzed=len(context.solutions),
                statistical_summary=statistical_summary,
                distribution_analysis=distribution_analysis,
                correlation_analysis=correlation_analysis or self._create_empty_correlation(),
                outlier_detection=outlier_detection,
                hypothesis_tests=hypothesis_tests,
                confidence_intervals=confidence_intervals,
                performance_insights=insights,
                recommendations=recommendations
            )
            
            # Create result
            analysis_time = time.time() - start_time
            result = AnalysisResult(analysis, analysis_time, context)
            
            # Store in history
            self.analysis_history.append(result)
            
            logger.info(f"Performance analysis completed in {analysis_time:.3f}s")
            logger.info(f"Generated {len(insights)} insights and {len(recommendations)} recommendations")
            
            return result
            
        except Exception as e:
            logger.error(f"Performance analysis failed: {e}")
            raise
    
    def _validate_context(self, context: AnalysisContext) -> None:
        """Validate analysis context."""
        if not context.solutions:
            raise ValueError("No solutions provided for analysis")
        
        if not context.performance_data and not hasattr(context.solutions[0], 'objective_values'):
            raise ValueError("No performance data available for analysis")
    
    def _calculate_statistical_summary(self, 
                                     metric_name: str,
                                     values: np.ndarray) -> StatisticalSummary:
        """Calculate comprehensive statistical summary."""
        basic_metrics = PerformanceMetrics.calculate_basic_metrics(values)
        advanced_metrics = PerformanceMetrics.calculate_advanced_metrics(values)
        
        return StatisticalSummary(
            metric_name=metric_name,
            count=basic_metrics.get('count', 0),
            mean=basic_metrics.get('mean', 0.0),
            std=basic_metrics.get('std', 0.0),
            min_value=basic_metrics.get('min', 0.0),
            max_value=basic_metrics.get('max', 0.0),
            median=basic_metrics.get('median', 0.0),
            q25=basic_metrics.get('q25', 0.0),
            q75=basic_metrics.get('q75', 0.0),
            skewness=advanced_metrics.get('skewness', 0.0),
            kurtosis=advanced_metrics.get('kurtosis', 0.0),
            coefficient_of_variation=advanced_metrics.get('coefficient_of_variation', 0.0)
        )
    
    def _analyze_distribution(self, 
                            metric_name: str,
                            values: np.ndarray) -> DistributionAnalysis:
        """Analyze distribution characteristics."""
        from .statistics import StatisticalAnalyzer
        
        # Use statistical analyzer for distribution fitting
        analyzer = StatisticalAnalyzer(self.config)
        return analyzer.fit_distribution(metric_name, values)
    
    def _detect_outliers(self, 
                        metric_name: str,
                        values: np.ndarray) -> OutlierDetection:
        """Detect outliers using statistical methods."""
        from .statistics import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(self.config)
        return analyzer.detect_outliers(metric_name, values)
    
    def _calculate_confidence_interval(self, 
                                     metric_name: str,
                                     values: np.ndarray) -> ConfidenceInterval:
        """Calculate confidence interval for metric."""
        if len(values) < 2:
            return ConfidenceInterval(
                metric_name=metric_name,
                confidence_level=self.config.confidence_level,
                lower_bound=float(values[0]) if len(values) > 0 else 0.0,
                upper_bound=float(values[0]) if len(values) > 0 else 0.0,
                mean=float(values[0]) if len(values) > 0 else 0.0,
                std_error=0.0
            )
        
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        std_error = std / np.sqrt(len(values))
        
        # Calculate confidence interval using t-distribution
        try:
            from scipy import stats
            alpha = 1 - self.config.confidence_level
            df = len(values) - 1
            t_critical = stats.t.ppf(1 - alpha/2, df)
        except ImportError:
            # Fallback to normal approximation
            from math import sqrt
            # Approximate t-critical for common confidence levels
            if self.config.confidence_level >= 0.99:
                t_critical = 2.576
            elif self.config.confidence_level >= 0.95:
                t_critical = 1.96
            else:
                t_critical = 1.645
        
        margin_of_error = t_critical * std_error
        
        return ConfidenceInterval(
            metric_name=metric_name,
            confidence_level=self.config.confidence_level,
            lower_bound=float(mean - margin_of_error),
            upper_bound=float(mean + margin_of_error),
            mean=float(mean),
            std_error=float(std_error),
            method="t_distribution"
        )
    
    def _analyze_correlations(self, 
                            context: AnalysisContext,
                            metrics: List[str]) -> CorrelationAnalysis:
        """Analyze correlations between metrics."""
        from .statistics import StatisticalAnalyzer
        
        analyzer = StatisticalAnalyzer(self.config)
        return analyzer.analyze_correlations(context, metrics)
    
    def _perform_hypothesis_tests(self, 
                                context: AnalysisContext,
                                metrics: List[str]) -> List[HypothesisTest]:
        """Perform hypothesis tests."""
        tests = []
        
        # Example: Test if metric means are significantly different from zero
        for metric_name in metrics:
            values = context.get_metric_values(metric_name)
            if values is not None and len(values) > 1:
                test = self._one_sample_t_test(metric_name, values, 0.0)
                if test:
                    tests.append(test)
        
        return tests
    
    def _one_sample_t_test(self, 
                          metric_name: str,
                          values: np.ndarray,
                          null_value: float) -> Optional[HypothesisTest]:
        """Perform one-sample t-test."""
        if len(values) < 2:
            return None
        
        try:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(values, null_value)
            
            # Calculate critical value
            alpha = self.config.significance_level
            df = len(values) - 1
            critical_value = stats.t.ppf(1 - alpha/2, df)
            
        except ImportError:
            # Fallback calculation
            mean = np.mean(values)
            std = np.std(values, ddof=1)
            std_error = std / np.sqrt(len(values))
            
            t_stat = (mean - null_value) / std_error if std_error > 0 else 0.0
            
            # Approximate p-value (very rough)
            p_value = 2 * (1 - 0.95) if abs(t_stat) > 1.96 else 0.5
            critical_value = 1.96  # Approximate
        
        return HypothesisTest(
            test_name=f"One-sample t-test: {metric_name}",
            null_hypothesis=f"{metric_name} mean = {null_value}",
            alternative_hypothesis=f"{metric_name} mean ≠ {null_value}",
            test_statistic=float(t_stat),
            p_value=float(p_value),
            critical_value=float(critical_value),
            significance_level=self.config.significance_level,
            reject_null=p_value < self.config.significance_level
        )
    
    def _create_empty_correlation(self) -> CorrelationAnalysis:
        """Create empty correlation analysis."""
        return CorrelationAnalysis(
            metric_pairs=[],
            correlation_matrix=np.array([]),
            correlation_method="pearson"
        )
    
    def _generate_insights(self, 
                          statistical_summary: Dict[str, StatisticalSummary],
                          distribution_analysis: Dict[str, DistributionAnalysis],
                          correlation_analysis: Optional[CorrelationAnalysis],
                          outlier_detection: Dict[str, OutlierDetection]) -> List[str]:
        """Generate performance insights."""
        insights = []
        
        # Statistical insights
        for metric_name, stats in statistical_summary.items():
            if stats.coefficient_of_variation > 0.5:
                insights.append(
                    f"{metric_name} shows high variability (CV={stats.coefficient_of_variation:.2f})"
                )
            
            if abs(stats.skewness) > 1.0:
                direction = "right" if stats.skewness > 0 else "left"
                insights.append(
                    f"{metric_name} distribution is significantly {direction}-skewed"
                )
        
        # Outlier insights
        for metric_name, outliers in outlier_detection.items():
            if outliers.outlier_percentage > 10.0:
                insights.append(
                    f"{metric_name} has {outliers.outlier_percentage:.1f}% outliers, "
                    f"indicating potential data quality issues"
                )
        
        # Correlation insights
        if correlation_analysis:
            strong_corrs = correlation_analysis.get_strong_correlations(0.7)
            for metric1, metric2, corr in strong_corrs:
                insights.append(
                    f"Strong correlation detected between {metric1} and {metric2} (r={corr:.2f})"
                )
        
        return insights
    
    def _generate_recommendations(self, 
                                statistical_summary: Dict[str, StatisticalSummary],
                                distribution_analysis: Dict[str, DistributionAnalysis],
                                correlation_analysis: Optional[CorrelationAnalysis],
                                outlier_detection: Dict[str, OutlierDetection]) -> List[str]:
        """Generate analysis recommendations."""
        recommendations = []
        
        # Variability recommendations
        high_var_metrics = [
            name for name, stats in statistical_summary.items()
            if stats.coefficient_of_variation > 0.3
        ]
        
        if high_var_metrics:
            recommendations.append(
                f"Consider optimization focus on reducing variability in: {', '.join(high_var_metrics)}"
            )
        
        # Outlier recommendations
        outlier_metrics = [
            name for name, outliers in outlier_detection.items()
            if outliers.outlier_percentage > 5.0
        ]
        
        if outlier_metrics:
            recommendations.append(
                f"Investigate outliers in: {', '.join(outlier_metrics)} for potential design improvements"
            )
        
        # Distribution recommendations
        for metric_name, dist_analysis in distribution_analysis.items():
            if dist_analysis.goodness_of_fit < 0.05:  # Poor fit
                recommendations.append(
                    f"Consider alternative modeling approaches for {metric_name} "
                    f"due to poor distribution fit"
                )
        
        return recommendations
    
    def compare_solutions(self, 
                         context1: AnalysisContext,
                         context2: AnalysisContext,
                         comparison_name: str = "Comparison") -> Dict[str, Any]:
        """Compare performance between two solution sets."""
        
        # Analyze both contexts
        result1 = self.analyze_performance(context1)
        result2 = self.analyze_performance(context2)
        
        comparison = {
            'comparison_name': comparison_name,
            'context1_size': len(context1.solutions),
            'context2_size': len(context2.solutions),
            'metric_comparisons': {},
            'insights': [],
            'recommendations': []
        }
        
        # Compare metrics
        common_metrics = set(result1.analysis.statistical_summary.keys()) & \
                        set(result2.analysis.statistical_summary.keys())
        
        for metric in common_metrics:
            stats1 = result1.analysis.statistical_summary[metric]
            stats2 = result2.analysis.statistical_summary[metric]
            
            mean_diff = stats2.mean - stats1.mean
            mean_pct_change = (mean_diff / stats1.mean * 100) if stats1.mean != 0 else 0
            
            comparison['metric_comparisons'][metric] = {
                'mean_difference': mean_diff,
                'percent_change': mean_pct_change,
                'context1_mean': stats1.mean,
                'context2_mean': stats2.mean,
                'context1_std': stats1.std,
                'context2_std': stats2.std
            }
            
            # Generate insights
            if abs(mean_pct_change) > 10:
                direction = "increased" if mean_pct_change > 0 else "decreased"
                comparison['insights'].append(
                    f"{metric} {direction} by {abs(mean_pct_change):.1f}%"
                )
        
        return comparison