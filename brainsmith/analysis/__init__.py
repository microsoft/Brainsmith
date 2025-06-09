"""
Comprehensive Performance Analysis Framework

This module provides deep performance analysis, benchmarking, and statistical insights
for FPGA design optimization results. It builds on the selection framework to provide
comprehensive analysis capabilities.

Key Features:
- Statistical performance analysis with distribution fitting
- Benchmarking against reference designs and industry standards
- Performance prediction using machine learning models
- Uncertainty quantification and confidence intervals
- Regression analysis and correlation detection
- Outlier detection and anomaly analysis

Main Components:
1. Performance Analysis Engine: Core statistical analysis and performance evaluation
2. Benchmarking Framework: Reference design database and comparative analysis
3. Statistical Analysis Tools: Distribution analysis, hypothesis testing, correlations
4. Predictive Modeling: ML-based performance prediction and trend analysis

Example Usage:
    from brainsmith.analysis import PerformanceAnalyzer, BenchmarkingEngine
    
    # Analyze solution performance
    analyzer = PerformanceAnalyzer()
    analysis = analyzer.analyze_performance(solutions, metrics=['throughput', 'power'])
    
    # Benchmark against reference designs
    benchmarker = BenchmarkingEngine('reference_db.json')
    benchmark_result = benchmarker.benchmark_design(best_solution, 'cnn_inference')
    
    # Get statistical insights
    stats = analysis.statistical_summary
    print(f"Mean throughput: {stats['throughput']['mean']:.2f}")
    print(f"95% CI: [{stats['throughput']['ci_lower']:.2f}, {stats['throughput']['ci_upper']:.2f}]")
"""

import logging
from typing import List, Dict, Any, Optional, Tuple, Union

# Core analysis components
from .engine import (
    PerformanceAnalyzer,
    AnalysisResult,
    AnalysisConfiguration,
    PerformanceMetrics
)

# Benchmarking framework
from .benchmarking import (
    BenchmarkingEngine,
    ReferenceDesignDB,
    BenchmarkResult,
    BenchmarkCategory,
    IndustryBenchmark
)

# Statistical analysis tools
from .statistics import (
    StatisticalAnalyzer,
    DistributionAnalysis,
    HypothesisTest,
    CorrelationAnalysis,
    OutlierDetection,
    ConfidenceInterval
)

# Predictive modeling
from .prediction import (
    PerformancePredictionModel,
    PredictionResult,
    ModelTrainer,
    UncertaintyQuantification,
    TrendAnalysis
)

# Data models
from .models import (
    PerformanceAnalysis,
    BenchmarkComparison,
    StatisticalSummary,
    PredictionAnalysis,
    AnalysisContext,
    PerformanceData
)

# Utilities
from .utils import (
    calculate_statistics,
    fit_distributions,
    detect_outliers,
    normalize_performance_data,
    create_analysis_context
)

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Core engine
    'PerformanceAnalyzer',
    'AnalysisResult',
    'AnalysisConfiguration',
    'PerformanceMetrics',
    
    # Benchmarking
    'BenchmarkingEngine',
    'ReferenceDesignDB',
    'BenchmarkResult',
    'BenchmarkCategory',
    'IndustryBenchmark',
    
    # Statistical analysis
    'StatisticalAnalyzer',
    'DistributionAnalysis',
    'HypothesisTest',
    'CorrelationAnalysis',
    'OutlierDetection',
    'ConfidenceInterval',
    
    # Predictive modeling
    'PerformancePredictionModel',
    'PredictionResult',
    'ModelTrainer',
    'UncertaintyQuantification',
    'TrendAnalysis',
    
    # Data models
    'PerformanceAnalysis',
    'BenchmarkComparison',
    'StatisticalSummary',
    'PredictionAnalysis',
    'AnalysisContext',
    'PerformanceData',
    
    # Utilities
    'calculate_statistics',
    'fit_distributions',
    'detect_outliers',
    'normalize_performance_data',
    'create_analysis_context'
]

# Initialize logging
logger.info(f"Performance Analysis Framework v{__version__} initialized")
logger.info("Available analysis capabilities: Statistics, Benchmarking, Prediction, ML Models")