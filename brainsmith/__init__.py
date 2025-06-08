# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
Brainsmith: Extensible FPGA accelerator design space exploration platform.

This package provides comprehensive tools for exploring FPGA accelerator design spaces
with support for multiple optimization strategies, blueprint-based configurations,
and advanced analysis capabilities.

Week 1 Implementation: Core architecture with existing component integration.
"""

# Core platform components (Week 1 implementation)
try:
    from .core.config import BrainsmithConfig
except ImportError:
    BrainsmithConfig = None

try:
    from .core.result import BrainsmithResult, DSEResult
except ImportError:
    BrainsmithResult = None
    DSEResult = None

try:
    from .core.metrics import BrainsmithMetrics
except ImportError:
    BrainsmithMetrics = None

from .core.design_space import DesignSpace, DesignPoint, ParameterDefinition

try:
    from .core.compiler import BrainsmithCompiler
except ImportError:
    BrainsmithCompiler = None

# Week 1 Core API (new extensible architecture)
from .core.api import (
    brainsmith_explore, brainsmith_roofline, brainsmith_dataflow_analysis,
    brainsmith_generate, brainsmith_workflow, validate_blueprint,
    explore_design_space  # Legacy compatibility wrapper
)

from .core import get_core_status

# Blueprint system (optional for Week 1)
try:
    from .blueprints import (
        Blueprint, get_blueprint, load_blueprint, 
        list_blueprints, get_design_space as get_blueprint_design_space
    )
except ImportError:
    Blueprint = None
    get_blueprint = None
    load_blueprint = None
    list_blueprints = None
    get_blueprint_design_space = None

# DSE system (Phase 3 - optional for Week 1)
try:
    from .dse import (
        DSEInterface, SimpleDSEEngine, ExternalDSEAdapter,
        DSEAnalyzer, ParetoAnalyzer
    )
    from .dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective
    from .dse.strategies import (
        SamplingStrategy, OptimizationStrategy, 
        create_dse_config_for_strategy, StrategySelector,
        COMMON_CONFIGS
    )
except ImportError:
    DSEInterface = None
    SimpleDSEEngine = None
    ExternalDSEAdapter = None
    DSEAnalyzer = None
    ParetoAnalyzer = None
    DSEConfiguration = None
    DSEObjective = None
    OptimizationObjective = None
    SamplingStrategy = None
    OptimizationStrategy = None
    create_dse_config_for_strategy = None
    StrategySelector = None
    COMMON_CONFIGS = None

# Simple API functions (enhanced for Week 1, fallback compatible)
def build_model(model_path: str, blueprint_name: str, 
               parameters: dict = None, config=None):
    """
    Build a single model configuration.
    
    Args:
        model_path: Path to input model
        blueprint_name: Name of blueprint to use
        parameters: Optional parameter overrides
        config: Optional configuration override
        
    Returns:
        Build result with metrics
    """
    if BrainsmithCompiler is None:
        # Fallback for Week 1 when compiler not available
        return {
            'status': 'fallback',
            'message': 'BrainsmithCompiler not available - using Week 1 fallback',
            'model_path': model_path,
            'blueprint_name': blueprint_name,
            'parameters': parameters or {},
            'week1_fallback': True
        }
    
    if config is None and BrainsmithConfig is not None:
        config = BrainsmithConfig()
    
    # Apply parameter overrides if provided
    if parameters and config:
        for param, value in parameters.items():
            setattr(config, param, value)
    
    compiler = BrainsmithCompiler(config)
    return compiler.compile_model(model_path, blueprint_name)


def optimize_model(model_path: str, blueprint_name: str,
                  parameters: dict = None,
                  max_evaluations: int = 100,
                  strategy: str = "auto",
                  objectives: list = None):
    """
    Optimize model with automatic strategy selection.
    
    Week 1 implementation routes to brainsmith_explore.
    """
    # Route to Week 1 API
    results, analysis = brainsmith_explore(
        model_path=model_path,
        blueprint_path=blueprint_name,  # Assume blueprint_name is path for Week 1
        exit_point="dataflow_generation"
    )
    
    return results


def load_design_space(blueprint_name: str) -> DesignSpace:
    """
    Load design space from blueprint.
    
    Args:
        blueprint_name: Name of blueprint
        
    Returns:
        Design space object
    """
    if get_blueprint_design_space is not None:
        return get_blueprint_design_space(blueprint_name)
    else:
        # Week 1 fallback
        return DesignSpace(name=f"fallback_{blueprint_name}")


def sample_design_space(design_space: DesignSpace, n_samples: int = 10,
                       strategy: str = "latin_hypercube", seed: int = None) -> list:
    """
    Sample points from design space.
    
    Args:
        design_space: Design space to sample from
        n_samples: Number of samples to generate
        strategy: Sampling strategy
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled design points
    """
    from .core.design_space import sample_design_space as sample_func
    return sample_func(design_space, n_samples, strategy, seed)


def analyze_dse_results(dse_result, export_path: str = None) -> dict:
    """
    Analyze DSE results with comprehensive analytics.
    
    Args:
        dse_result: DSE result to analyze
        export_path: Optional path to export analysis
        
    Returns:
        Analysis dictionary
    """
    if DSEAnalyzer is None:
        # Week 1 fallback
        return {
            'status': 'fallback',
            'message': 'DSEAnalyzer not available - using Week 1 fallback',
            'week1_implementation': True,
            'basic_analysis': {
                'total_results': len(getattr(dse_result, 'results', [])),
                'best_result': getattr(dse_result, 'best_result', None)
            }
        }
    
    if not hasattr(dse_result, 'design_space') or not dse_result.design_space:
        raise ValueError("DSE result must contain design space information")
    
    # Create analyzer
    objectives = []
    if hasattr(dse_result, 'objectives') and dse_result.objectives:
        for obj_name in dse_result.objectives:
            objectives.append(DSEObjective(obj_name, OptimizationObjective.MAXIMIZE))
    else:
        objectives = [DSEObjective("performance.throughput_ops_sec", OptimizationObjective.MAXIMIZE)]
    
    analyzer = DSEAnalyzer(dse_result.design_space, objectives)
    analysis = analyzer.analyze_dse_result(dse_result)
    
    if export_path:
        analyzer.export_analysis(analysis, export_path)
    
    return analysis


def get_pareto_frontier(dse_result) -> list:
    """
    Extract Pareto frontier from DSE results.
    
    Args:
        dse_result: DSE result to analyze
        
    Returns:
        List of Pareto-optimal results
    """
    if not hasattr(dse_result, 'objectives') or len(dse_result.objectives) < 2:
        # Single objective: return best result
        if hasattr(dse_result, 'best_result') and dse_result.best_result:
            return [dse_result.best_result]
        else:
            return []
    
    if ParetoAnalyzer is None:
        # Week 1 fallback
        return [getattr(dse_result, 'best_result', None)] if hasattr(dse_result, 'best_result') else []
    
    # Multi-objective: compute Pareto frontier
    objectives = [DSEObjective(obj_name, OptimizationObjective.MAXIMIZE) for obj_name in dse_result.objectives]
    pareto_analyzer = ParetoAnalyzer(objectives)
    pareto_points = pareto_analyzer.compute_pareto_frontier(dse_result.results)
    
    return [point.result for point in pareto_points]


def list_available_strategies() -> dict:
    """
    List all available DSE strategies.
    
    Returns:
        Dictionary of available strategies with descriptions
    """
    if create_dse_config_for_strategy is None:
        # Week 1 fallback
        return {
            "week1_fallback": {
                "description": "Week 1 implementation using existing components",
                "available": True,
                "supports_multi_objective": False,
                "recommended_max_evaluations": 50,
                "external_library": None
            }
        }
    
    try:
        from .dse.external import check_framework_availability
        from .dse.strategies import STRATEGY_CONFIGS
        
        framework_status = check_framework_availability()
        
        strategies = {}
        for strategy_enum, config in STRATEGY_CONFIGS.items():
            strategy_name = strategy_enum.value if hasattr(strategy_enum, 'value') else str(strategy_enum)
            
            available = True
            if config.requires_external_library:
                if config.external_library == "scipy":
                    available = framework_status.get("scipy", False)
                elif config.external_library in framework_status:
                    available = framework_status[config.external_library]
                else:
                    available = False
            
            strategies[strategy_name] = {
                "description": config.description,
                "available": available,
                "supports_multi_objective": config.supports_multi_objective,
                "recommended_max_evaluations": config.recommended_max_evaluations,
                "external_library": config.external_library if config.requires_external_library else None
            }
        
        return strategies
    except ImportError:
        return list_available_strategies()  # Recursively call fallback


def recommend_strategy(n_parameters: int = None, 
                      max_evaluations: int = 100,
                      n_objectives: int = 1,
                      blueprint_name: str = None) -> str:
    """
    Recommend DSE strategy based on problem characteristics.
    
    Args:
        n_parameters: Number of parameters (auto-detected from blueprint if not provided)
        max_evaluations: Evaluation budget
        n_objectives: Number of objectives
        blueprint_name: Blueprint name for auto-detection
        
    Returns:
        Recommended strategy name
    """
    if StrategySelector is None:
        # Week 1 fallback
        return "week1_fallback"
    
    # Auto-detect parameters from blueprint
    if n_parameters is None and blueprint_name:
        try:
            design_space = load_design_space(blueprint_name)
            n_parameters = len(design_space.parameters)
        except:
            n_parameters = 5  # Default assumption
    elif n_parameters is None:
        n_parameters = 5  # Default assumption
    
    selector = StrategySelector()
    return selector.select_best_strategy(
        n_parameters=n_parameters,
        max_evaluations=max_evaluations,
        n_objectives=n_objectives,
        problem_type="fpga"
    )


# Enhanced exports for Week 1
__all__ = [
    # Week 1 Core API
    'brainsmith_explore',
    'brainsmith_roofline', 
    'brainsmith_dataflow_analysis',
    'brainsmith_generate',
    'brainsmith_workflow',
    'validate_blueprint',
    'explore_design_space',  # Legacy compatibility
    'get_core_status',
    
    # Core components (may be None if not available)
    'BrainsmithConfig',
    'BrainsmithResult', 
    'DSEResult',
    'BrainsmithMetrics',
    'DesignSpace',
    'DesignPoint',
    'ParameterDefinition',
    'BrainsmithCompiler',
    
    # Blueprint system (may be None if not available)
    'Blueprint',
    'get_blueprint',
    'load_blueprint',
    'list_blueprints',
    
    # DSE system (may be None if not available)
    'DSEInterface',
    'SimpleDSEEngine', 
    'ExternalDSEAdapter',
    'DSEAnalyzer',
    'ParetoAnalyzer',
    'DSEConfiguration',
    'DSEObjective',
    'OptimizationObjective',
    'SamplingStrategy',
    'OptimizationStrategy',
    
    # Simple API (with Week 1 fallbacks)
    'build_model',
    'optimize_model',
    'load_design_space',
    'sample_design_space',
    'analyze_dse_results',
    'get_pareto_frontier',
    
    # Utility functions (with Week 1 fallbacks)
    'list_available_strategies',
    'recommend_strategy',
    
    # Common configurations (may be None)
    'COMMON_CONFIGS'
]

# Version information (Week 1)
__version__ = "0.4.0"  # Week 1 version
__author__ = "Microsoft Research"
__description__ = "Extensible FPGA accelerator design space exploration platform - Week 1 Implementation"
