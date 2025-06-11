"""
Data Management Infrastructure

Complete data collection, processing, and export system for BrainSmith.
Provides unified interface for metrics collection, analysis, and data lifecycle management.
"""

# Core data collection functions
from .collection import (
    collect_build_metrics,
    collect_dse_metrics,
    summarize_data,
    compare_results,
    filter_data,
    validate_data
)

# Data export functions
from .export import (
    export_metrics,
    export_summary,
    export_pareto_frontier,
    export_comparison,
    export_best_results,
    export_dse_analysis
)

# Data management and lifecycle
from .management import (
    DataManager,
    get_data_manager,
    set_data_manager,
    collect_and_cache,
    export_complete_analysis,
    process_batch_results,
    validate_metrics_quality,
    cleanup_old_cache
)

# Data types and structures
from .types import (
    BuildMetrics,
    PerformanceData,
    ResourceData,
    QualityData,
    BuildData,
    DataSummary,
    ComparisonResult,
    SelectionCriteria,
    TradeoffAnalysis,
    MetricsList,
    DataList,
    MetricsData
)

__all__ = [
    # Core collection functions
    'collect_build_metrics',
    'collect_dse_metrics',
    'summarize_data',
    'compare_results',
    'filter_data',
    'validate_data',
    
    # Export functions
    'export_metrics',
    'export_summary',
    'export_pareto_frontier',
    'export_comparison',
    'export_best_results',
    'export_dse_analysis',
    
    # Data management
    'DataManager',
    'get_data_manager',
    'set_data_manager',
    'collect_and_cache',
    'export_complete_analysis',
    'process_batch_results',
    'validate_metrics_quality',
    'cleanup_old_cache',
    
    # Data types
    'BuildMetrics',
    'PerformanceData',
    'ResourceData',
    'QualityData',
    'BuildData',
    'DataSummary',
    'ComparisonResult',
    'SelectionCriteria',
    'TradeoffAnalysis',
    'MetricsList',
    'DataList',
    'MetricsData'
]

# Version info
__version__ = "2.0.0"  # Updated for complete data infrastructure