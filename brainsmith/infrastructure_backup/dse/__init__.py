"""
Design Space Exploration Infrastructure

Complete DSE engine with parameter sweep, optimization, and analysis capabilities.
Provides the infrastructure layer for design space exploration in BrainSmith.
"""

# Core DSE functionality
from .engine import (
    parameter_sweep,
    batch_evaluate, 
    find_best_result,
    compare_results,
    sample_design_space
)

# Helper functions
from .helpers import (
    generate_parameter_grid,
    create_parameter_samples,
    estimate_runtime,
    validate_parameter_space,
    optimize_parameter_selection,
    create_parameter_ranges,
    analyze_parameter_coverage
)

# Type definitions
from .types import (
    DSEConfiguration,
    DSEResult,
    DSEResults,
    DSEObjective,
    OptimizationObjective,
    SamplingStrategy,
    ParameterSpace,
    ParameterSet,
    ParameterDefinition,
    ComparisonResult,
    ExplorationStatistics
)

# Main interface
from .interface import (
    DSEInterface,
    create_dse_config_for_strategy,
    run_simple_dse,
    quick_parameter_sweep,
    find_best_throughput,
    find_best_efficiency
)

# Blueprint management
from .blueprint_manager import (
    BlueprintManager,
    load_blueprint,
    list_available_blueprints,
    create_design_point_from_blueprint
)

# Design space management (existing)
from .design_space import DesignSpace, DesignPoint

__all__ = [
    # Core engine functions
    'parameter_sweep',
    'batch_evaluate',
    'find_best_result', 
    'compare_results',
    'sample_design_space',
    
    # Helper functions
    'generate_parameter_grid',
    'create_parameter_samples',
    'estimate_runtime',
    'validate_parameter_space',
    'optimize_parameter_selection',
    'create_parameter_ranges',
    'analyze_parameter_coverage',
    
    # Type definitions
    'DSEConfiguration',
    'DSEResult',
    'DSEResults',
    'DSEObjective',
    'OptimizationObjective',
    'SamplingStrategy',
    'ParameterSpace',
    'ParameterSet',
    'ParameterDefinition',
    'ComparisonResult',
    'ExplorationStatistics',
    
    # Main interface
    'DSEInterface',
    'create_dse_config_for_strategy',
    'run_simple_dse',
    'quick_parameter_sweep',
    'find_best_throughput',
    'find_best_efficiency',
    
    # Blueprint management
    'BlueprintManager',
    'load_blueprint',
    'list_available_blueprints',
    'create_design_point_from_blueprint',
    
    # Design space management
    'DesignSpace',
    'DesignPoint'
]

# Version info
__version__ = "2.0.0"  # Updated for complete DSE infrastructure