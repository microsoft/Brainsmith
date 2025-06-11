"""
BrainSmith Data Module - North Star Aligned Implementation

Unified data collection, analysis, and selection for FPGA build results.
Consolidates metrics, analysis, and selection modules while eliminating enterprise complexity.

Core Philosophy:
- Functions Over Frameworks: Simple function calls for data operations
- Data Exposure: Clean integration with external analysis tools
- Simplicity Over Sophistication: Essential functionality only
- Practical Focus: FPGA-specific selection vs academic MCDA algorithms

Example Usage:
    # Collect metrics from build result
    metrics = collect_build_metrics(build_result, 'model.onnx', 'blueprint.yaml')
    
    # DSE workflow with selection
    dse_results = collect_dse_metrics(dse_sweep_results)
    pareto_solutions = find_pareto_optimal(dse_results)
    best_designs = select_best_solutions(pareto_solutions, SelectionCriteria(
        max_lut_utilization=80, min_throughput=1000
    ))
    
    # Export for external analysis
    df = to_pandas(best_designs)
    df.plot(x='pe_count', y='throughput_ops_sec', kind='scatter')
"""

# Core data collection functions
from .functions import (
    collect_build_metrics,
    collect_dse_metrics,
    summarize_data,
    compare_results,
    validate_data,
    filter_data,
    
    # Selection functions (NEW - replaces complex selection module)
    find_pareto_optimal,
    rank_by_efficiency,
    select_best_solutions,
    filter_feasible_designs,
    compare_design_tradeoffs
)

# Data export functions for external tools
from .export import (
    export_for_analysis,
    to_pandas,
    to_csv,
    to_json,
    create_report
)

# Simple data types
from .types import (
    BuildMetrics,
    PerformanceData,
    ResourceData,
    QualityData,
    BuildData,
    DataSummary,
    ComparisonResult,
    
    # Selection types (NEW)
    SelectionCriteria,
    TradeoffAnalysis
)

__all__ = [
    # Core functions (6 functions)
    'collect_build_metrics',
    'collect_dse_metrics',
    'summarize_data',
    'compare_results',
    'validate_data',
    'filter_data',
    
    # Selection functions (5 NEW functions - replaces selection module)
    'find_pareto_optimal',
    'rank_by_efficiency',
    'select_best_solutions',
    'filter_feasible_designs',
    'compare_design_tradeoffs',
    
    # Export functions (5 functions)
    'export_for_analysis',
    'to_pandas',
    'to_csv',
    'to_json',
    'create_report',
    
    # Data types (7 + 2 NEW types)
    'BuildMetrics',
    'PerformanceData',
    'ResourceData',
    'QualityData',
    'BuildData',
    'DataSummary',
    'ComparisonResult',
    
    # Selection types (NEW)
    'SelectionCriteria',
    'TradeoffAnalysis'
]

# Version info
__version__ = "3.0.0-unified"  # North Star aligned consolidation

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Data v{__version__} - Unified data collection and export")
logger.info(f"API surface: {len(__all__)} exports (down from 25+ in separate modules)")
logger.info("Integration: core, dse, finn, hooks + external analysis tools")

# Backwards compatibility for metrics module
def collect_performance_metrics(build_result):
    """Deprecated: Use collect_build_metrics() instead."""
    import warnings
    warnings.warn(
        "collect_performance_metrics is deprecated. Use collect_build_metrics() for unified metrics collection.",
        DeprecationWarning,
        stacklevel=2
    )
    return collect_build_metrics(build_result).performance

def collect_resource_metrics(build_result):
    """Deprecated: Use collect_build_metrics() instead."""
    import warnings
    warnings.warn(
        "collect_resource_metrics is deprecated. Use collect_build_metrics() for unified metrics collection.",
        DeprecationWarning,
        stacklevel=2
    )
    return collect_build_metrics(build_result).resources

# Backwards compatibility for analysis module
def expose_analysis_data(dse_results):
    """Deprecated: Use export_for_analysis() instead."""
    import warnings
    warnings.warn(
        "expose_analysis_data is deprecated. Use export_for_analysis() for data export.",
        DeprecationWarning,
        stacklevel=2
    )
    
    # Legacy function returned raw input, not processed dict
    return dse_results