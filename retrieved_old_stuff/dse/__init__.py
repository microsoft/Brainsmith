"""
Simplified DSE: Functions Over Frameworks

North Star aligned design space exploration for FPGA accelerators.
Simple functions that integrate seamlessly with streamlined BrainSmith modules.

Core Philosophy:
- Functions Over Frameworks: Simple function calls instead of enterprise objects
- Simplicity Over Sophistication: 8 functions instead of 50+ classes
- Focus Over Feature Creep: Core FPGA DSE only, no academic algorithms
- Hooks Over Implementation: Export data for external analysis tools
- Performance Over Purity: Fast parameter sweeps, practical results

Example Usage:
    # Basic parameter sweep
    parameters = {
        'pe_count': [1, 2, 4, 8],
        'simd_factor': [1, 2, 4],
        'precision': [8, 16]
    }
    results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)
    
    # Find best performing configuration
    best = find_best_result(results, 'throughput_ops_sec')
    
    # Export for external analysis
    df = export_results(results, 'pandas')
    df.plot(x='pe_count', y='throughput', kind='scatter')
"""

# Core DSE functions - North Star aligned
from .functions import (
    parameter_sweep,
    batch_evaluate,
    find_best_result,
    compare_results,
    sample_design_space
)

# Helper functions for practical DSE workflows
from .helpers import (
    generate_parameter_grid,
    create_parameter_samples,
    export_results,
    estimate_runtime,
    count_parameter_combinations,
    validate_parameter_space,
    create_parameter_subsets,
    filter_results,
    sort_results
)

# Simple data types
from .types import (
    DSEResult,
    ParameterSet,
    ComparisonResult,
    DSEConfiguration,
    ParameterSpace,
    ParameterCombination,
    MetricName
)

# Public API - 8 core functions + 9 helpers + 7 types = 24 total exports
# vs 50+ classes in the previous enterprise framework
__all__ = [
    # Core DSE functions (5 functions)
    'parameter_sweep',
    'batch_evaluate', 
    'find_best_result',
    'compare_results',
    'sample_design_space',
    
    # Helper functions (9 functions)
    'generate_parameter_grid',
    'create_parameter_samples',
    'export_results',
    'estimate_runtime',
    'count_parameter_combinations',
    'validate_parameter_space',
    'create_parameter_subsets',
    'filter_results',
    'sort_results',
    
    # Data types (7 types)
    'DSEResult',
    'ParameterSet',
    'ComparisonResult', 
    'DSEConfiguration',
    'ParameterSpace',
    'ParameterCombination',
    'MetricName'
]

# Backwards compatibility warnings for deprecated enterprise interfaces
import warnings

def create_dse_engine(*args, **kwargs):
    """Deprecated: Use parameter_sweep() instead."""
    warnings.warn(
        "create_dse_engine is deprecated. Use parameter_sweep() for simple DSE workflows.",
        DeprecationWarning,
        stacklevel=2
    )
    # Return a simple function wrapper for basic compatibility
    return parameter_sweep

def DSEInterface(*args, **kwargs):
    """Deprecated: Use parameter_sweep() instead."""
    warnings.warn(
        "DSEInterface is deprecated. Use parameter_sweep() for simple DSE workflows.",
        DeprecationWarning,
        stacklevel=2
    )
    return parameter_sweep

def DSEEngine(*args, **kwargs):
    """Deprecated: Use parameter_sweep() instead."""
    warnings.warn(
        "DSEEngine is deprecated. Use parameter_sweep() for simple DSE workflows.", 
        DeprecationWarning,
        stacklevel=2
    )
    return parameter_sweep

# Version and metadata
__version__ = "2.0.0-simplified"
__author__ = "BrainSmith Development Team"
__description__ = "Simple FPGA design space exploration functions"

# Module initialization logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith DSE v{__version__} - North Star aligned simple functions loaded")
logger.info(f"API surface: {len(__all__)} exports (vs 50+ in enterprise framework)")
logger.info("Integration: core, blueprints, hooks, finn, metrics modules")