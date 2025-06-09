"""
Statistical Analysis Tools
Advanced statistical analysis capabilities including distribution fitting,
correlation analysis, and hypothesis testing.
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
import logging

from .models import (
    DistributionAnalysis, CorrelationAnalysis, OutlierDetection,
    HypothesisTest, DistributionType, AnalysisConfiguration,
    AnalysisContext
)

logger = logging.getLogger(__name__)


class StatisticalAnalyzer:
    """Advanced statistical analysis tools."""
    
    def __init__(self, configuration: AnalysisConfiguration):
        self.config = configuration
    
    def fit_distribution(self, 
                        metric_name: str,
                        values: np.ndarray) -> DistributionAnalysis:
        """Fit statistical distributions to data and find best fit."""
        if len(values) < 3:
            # Not enough data for distribution fitting
            return DistributionAnalysis(
                metric_name=metric_name,
                best_fit_distribution=DistributionType.NORMAL,
                distribution_parameters={'mean': float(np.mean(values)), 'std': 1.0},
                goodness_of_fit=0.0,
                confidence_level=self.config.confidence_level
            )
        
        best_distribution = DistributionType.NORMAL
        best_params = {}
        best_score = 0.0
        fit_scores = {}
        
        # Test different distributions
        for dist_type in self.config.distribution_tests:
            try:
                params, score = self._fit_single_distribution(values, dist_type)
                fit_scores[dist_type] = score
                
                if score > best_score:
                    best_score = score
                    best_distribution = dist_type
                    best_params = params
                    
            except Exception as e:
                logger.debug(f"Failed to fit {dist_type.value}: {e}")
                fit_scores[dist_type] = 0.0
        
        return DistributionAnalysis(
            metric_name=metric_name,
            best_fit_distribution=best_distribution,
            distribution_parameters=best_params,
            goodness_of_fit=best_score,
            confidence_level=self.config.confidence_level,
            tested_distributions=list(self.config.distribution_tests),
            fit_scores=fit_scores
        )
    
    def _fit_single_distribution(self, 
                                values: np.ndarray,
                                dist_type: DistributionType) -> Tuple[Dict[str, float], float]:
        """Fit a single distribution type."""
        
        if dist_type == DistributionType.NORMAL:
            return self._fit_normal(values)
        elif dist_type == DistributionType.LOGNORMAL:
            return self._fit_lognormal(values)
        elif dist_type == DistributionType.EXPONENTIAL:
            return self._fit_exponential(values)
        elif dist_type == DistributionType.UNIFORM:
            return self._fit_uniform(values)
        else:
            # Fallback to normal
            return self._fit_normal(values)
    
    def _fit_normal(self, values: np.ndarray) -> Tuple[Dict[str, float], float]:
        """Fit normal distribution."""
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=1))
        
        params = {'mean': mean, 'std': std}
        
        # Calculate goodness of fit using simplified approach
        score = self._calculate_normal_fit_score(values, mean, std)
        
        return params, score
    
    def _fit_lognormal(self, values: np.ndarray) -> Tuple[Dict[str, float], float]:
        """Fit lognormal distribution."""
        if np.any(values <= 0):
            # Lognormal requires positive values
            return {'mean': 0.0, 'std': 1.0}, 0.0
        
        log_values = np.log(values)
        mean_log = float(np.mean(log_values))
        std_log = float(np.std(log_values, ddof=1))
        
        params = {'mean_log': mean_log, 'std_log': std_log}
        
        # Calculate fit score
        score = self._calculate_lognormal_fit_score(values, mean_log, std_log)
        
        return params, score
    
    def _fit_exponential(self, values: np.ndarray) -> Tuple[Dict[str, float], float]:
        """Fit exponential distribution."""
        if np.any(values < 0):
            return {'lambda': 1.0}, 0.0
        
        lambda_param = 1.0 / float(np.mean(values)) if np.mean(values) > 0 else 1.0
        params = {'lambda': lambda_param}
        
        score = self._calculate_exponential_fit_score(values, lambda_param)
        
        return params, score
    
    def _fit_uniform(self, values: np.ndarray) -> Tuple[Dict[str, float], float]:
        """Fit uniform distribution."""
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        params = {'min': min_val, 'max': max_val}
        
        score = self._calculate_uniform_fit_score(values, min_val, max_val)
        
        return params, score
    
    def _calculate_normal_fit_score(self, 
                                   values: np.ndarray,
                                   mean: float,
                                   std: float) -> float:
        """Calculate fit score for normal distribution."""
        if std <= 0:
            return 0.0
        
        # Simplified Kolmogorov-Smirnov-like test
        # Generate expected quantiles
        n = len(values)
        sorted_values = np.sort(values)
        
        # Calculate empirical CDF
        empirical_cdf = np.arange(1, n + 1) / n
        
        # Calculate theoretical CDF for normal distribution
        theoretical_cdf = np.array([
            0.5 * (1 + np.sign(val - mean) * np.sqrt(1 - np.exp(-2 * ((val - mean) / std) ** 2)))
            for val in sorted_values
        ])
        
        # Ensure CDF values are in valid range
        theoretical_cdf = np.clip(theoretical_cdf, 0.001, 0.999)
        
        # Calculate maximum difference (KS statistic)
        ks_statistic = np.max(np.abs(empirical_cdf - theoretical_cdf))
        
        # Convert to score (lower KS statistic = better fit)
        score = max(0.0, 1.0 - ks_statistic * 2)
        
        return score
    
    def _calculate_lognormal_fit_score(self, 
                                      values: np.ndarray,
                                      mean_log: float,
                                      std_log: float) -> float:
        """Calculate fit score for lognormal distribution."""
        if std_log <= 0:
            return 0.0
        
        log_values = np.log(values)
        
        # Use normal fit score on log-transformed data
        return self._calculate_normal_fit_score(log_values, mean_log, std_log)
    
    def _calculate_exponential_fit_score(self, 
                                        values: np.ndarray,
                                        lambda_param: float) -> float:
        """Calculate fit score for exponential distribution."""
        if lambda_param <= 0:
            return 0.0
        
        # For exponential distribution, check if data follows exponential decay
        sorted_values = np.sort(values)
        n = len(values)
        
        # Calculate empirical survival function
        empirical_sf = 1 - np.arange(1, n + 1) / n
        
        # Calculate theoretical survival function: S(x) = exp(-λx)
        theoretical_sf = np.exp(-lambda_param * sorted_values)
        
        # Calculate fit score
        ks_statistic = np.max(np.abs(empirical_sf - theoretical_sf))
        score = max(0.0, 1.0 - ks_statistic * 2)
        
        return score
    
    def _calculate_uniform_fit_score(self, 
                                    values: np.ndarray,
                                    min_val: float,
                                    max_val: float) -> float:
        """Calculate fit score for uniform distribution."""
        if max_val <= min_val:
            return 0.0
        
        # For uniform distribution, check if values are evenly distributed
        sorted_values = np.sort(values)
        n = len(values)
        
        # Calculate expected uniform spacing
        expected_spacing = (max_val - min_val) / (n - 1) if n > 1 else 0
        
        if expected_spacing == 0:
            return 0.0
        
        # Calculate actual spacings
        if n > 1:
            actual_spacings = np.diff(sorted_values)
            spacing_variance = np.var(actual_spacings)
            
            # Lower variance in spacing indicates better uniform fit
            score = max(0.0, 1.0 - spacing_variance / (expected_spacing ** 2))
        else:
            score = 1.0
        
        return score
    
    def detect_outliers(self, 
                       metric_name: str,
                       values: np.ndarray) -> OutlierDetection:
        """Detect outliers using multiple methods."""
        
        outlier_indices = []
        outlier_scores = np.zeros(len(values))
        
        if len(values) < 3:
            return OutlierDetection(
                metric_name=metric_name,
                outlier_indices=[],
                outlier_values=np.array([]),
                detection_method="insufficient_data",
                threshold=self.config.outlier_threshold,
                outlier_scores=outlier_scores,
                total_samples=len(values)
            )
        
        # Z-score method
        z_outliers, z_scores = self._detect_outliers_zscore(values)
        
        # IQR method
        iqr_outliers = self._detect_outliers_iqr(values)
        
        # Combine methods (intersection for conservative approach)
        outlier_indices = list(set(z_outliers) & set(iqr_outliers))
        outlier_values = values[outlier_indices] if outlier_indices else np.array([])
        
        return OutlierDetection(
            metric_name=metric_name,
            outlier_indices=outlier_indices,
            outlier_values=outlier_values,
            detection_method="zscore_iqr_combined",
            threshold=self.config.outlier_threshold,
            outlier_scores=z_scores,
            total_samples=len(values)
        )
    
    def _detect_outliers_zscore(self, values: np.ndarray) -> Tuple[List[int], np.ndarray]:
        """Detect outliers using Z-score method."""
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            return [], np.zeros(len(values))
        
        z_scores = np.abs((values - mean) / std)
        outlier_indices = np.where(z_scores > self.config.outlier_threshold)[0].tolist()
        
        return outlier_indices, z_scores
    
    def _detect_outliers_iqr(self, values: np.ndarray) -> List[int]:
        """Detect outliers using IQR method."""
        q25 = np.percentile(values, 25)
        q75 = np.percentile(values, 75)
        iqr = q75 - q25
        
        if iqr == 0:
            return []
        
        # Standard IQR outlier detection
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        
        outlier_indices = np.where((values < lower_bound) | (values > upper_bound))[0].tolist()
        
        return outlier_indices
    
    def analyze_correlations(self, 
                           context: AnalysisContext,
                           metrics: List[str]) -> CorrelationAnalysis:
        """Analyze correlations between metrics."""
        
        if len(metrics) < 2:
            return CorrelationAnalysis(
                metric_pairs=[],
                correlation_matrix=np.array([]),
                correlation_method="pearson"
            )
        
        # Collect metric values
        metric_data = {}
        for metric in metrics:
            values = context.get_metric_values(metric)
            if values is not None and len(values) > 0:
                metric_data[metric] = values
        
        valid_metrics = list(metric_data.keys())
        if len(valid_metrics) < 2:
            return CorrelationAnalysis(
                metric_pairs=[],
                correlation_matrix=np.array([]),
                correlation_method="pearson"
            )
        
        # Create data matrix
        min_length = min(len(values) for values in metric_data.values())
        data_matrix = np.zeros((min_length, len(valid_metrics)))
        
        for i, metric in enumerate(valid_metrics):
            data_matrix[:, i] = metric_data[metric][:min_length]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(data_matrix.T)
        
        # Create metric pairs
        metric_pairs = [(metric, metric) for metric in valid_metrics]
        
        # Find significant correlations
        significant_correlations = []
        for i in range(len(valid_metrics)):
            for j in range(i + 1, len(valid_metrics)):
                corr = correlation_matrix[i, j]
                if abs(corr) >= self.config.correlation_threshold:
                    significant_correlations.append((valid_metrics[i], valid_metrics[j], corr))
        
        return CorrelationAnalysis(
            metric_pairs=metric_pairs,
            correlation_matrix=correlation_matrix,
            correlation_method="pearson",
            significant_correlations=significant_correlations
        )
    
    def perform_normality_test(self, values: np.ndarray) -> HypothesisTest:
        """Perform Shapiro-Wilk normality test (simplified version)."""
        
        if len(values) < 3:
            return HypothesisTest(
                test_name="Normality Test",
                null_hypothesis="Data is normally distributed",
                alternative_hypothesis="Data is not normally distributed",
                test_statistic=0.0,
                p_value=1.0,
                critical_value=0.05,
                significance_level=self.config.significance_level,
                reject_null=False
            )
        
        # Simplified normality test based on skewness and kurtosis
        n = len(values)
        mean = np.mean(values)
        std = np.std(values, ddof=1)
        
        if std == 0:
            # No variation - not normal
            return HypothesisTest(
                test_name="Normality Test",
                null_hypothesis="Data is normally distributed",
                alternative_hypothesis="Data is not normally distributed",
                test_statistic=float('inf'),
                p_value=0.0,
                critical_value=0.05,
                significance_level=self.config.significance_level,
                reject_null=True
            )
        
        # Calculate skewness and kurtosis
        normalized_values = (values - mean) / std
        skewness = np.mean(normalized_values ** 3)
        kurtosis = np.mean(normalized_values ** 4) - 3  # Excess kurtosis
        
        # Combined test statistic (simplified)
        test_statistic = abs(skewness) + abs(kurtosis) / 2
        
        # Rough p-value estimation
        if test_statistic < 0.5:
            p_value = 0.8
        elif test_statistic < 1.0:
            p_value = 0.3
        elif test_statistic < 2.0:
            p_value = 0.1
        else:
            p_value = 0.01
        
        return HypothesisTest(
            test_name="Normality Test (Simplified)",
            null_hypothesis="Data is normally distributed",
            alternative_hypothesis="Data is not normally distributed",
            test_statistic=test_statistic,
            p_value=p_value,
            critical_value=0.05,
            significance_level=self.config.significance_level,
            reject_null=p_value < self.config.significance_level
        )