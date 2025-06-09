"""
Brainsmith Enhanced Metrics Collection Framework
Comprehensive performance, resource, and quality metrics for FINN-based accelerator design.
"""

from .performance import (
    AdvancedPerformanceMetrics, TimingAnalyzer, ThroughputProfiler, 
    LatencyAnalyzer, PowerEstimator, PerformanceCollector
)
from .resources import (
    ResourceUtilizationTracker, UtilizationMonitor, ResourcePredictor,
    EfficiencyAnalyzer, FPGAResourceAnalyzer
)
from .analysis import (
    HistoricalAnalysisEngine, TrendAnalyzer, RegressionDetector,
    BaselineManager, AlertSystem, MetricsDatabase
)
from .quality import (
    QualityMetricsCollector, AccuracyAnalyzer, PrecisionTracker,
    ReliabilityAssessment, ValidationMetrics
)
from .core import (
    MetricsCollector, MetricsAggregator, MetricsExporter,
    MetricsConfiguration, MetricsRegistry
)

__all__ = [
    # Performance Metrics
    'AdvancedPerformanceMetrics',
    'TimingAnalyzer',
    'ThroughputProfiler',
    'LatencyAnalyzer', 
    'PowerEstimator',
    'PerformanceCollector',
    
    # Resource Metrics
    'ResourceUtilizationTracker',
    'UtilizationMonitor',
    'ResourcePredictor',
    'EfficiencyAnalyzer',
    'FPGAResourceAnalyzer',
    
    # Historical Analysis
    'HistoricalAnalysisEngine',
    'TrendAnalyzer',
    'RegressionDetector',
    'BaselineManager',
    'AlertSystem',
    'MetricsDatabase',
    
    # Quality Metrics
    'QualityMetricsCollector',
    'AccuracyAnalyzer',
    'PrecisionTracker',
    'ReliabilityAssessment',
    'ValidationMetrics',
    
    # Core Framework
    'MetricsCollector',
    'MetricsAggregator',
    'MetricsExporter',
    'MetricsConfiguration',
    'MetricsRegistry'
]

__version__ = '2.0.0'