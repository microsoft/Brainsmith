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

from .core.dse.types import DesignPoint, ParameterSpace
from .core.dse.design_space import DesignSpace
try:
    from .core.dse.types import ParameterDefinition
except ImportError:
    ParameterDefinition = None

try:
    from .core.compiler import BrainsmithCompiler
except ImportError:
    BrainsmithCompiler = None

# Simplified Core API (new forge-based architecture)
from .core.api import forge, validate_blueprint

try:
    from .core import get_core_status
except ImportError:
    get_core_status = None

# Blueprint system (updated for new structure)
try:
    from .core.dse.blueprint_manager import BlueprintManager, load_blueprint, list_available_blueprints
    from libraries.blueprints import discover_all_blueprints, get_blueprint_by_name
    
    # Compatibility wrappers
    def get_blueprint(name):
        return get_blueprint_by_name(name)
    
    def list_blueprints():
        return list_available_blueprints()
        
    def get_blueprint_design_space(name):
        manager = BlueprintManager()
        return manager.get_blueprint_parameter_space(name)
        
    Blueprint = None  # Legacy class not available
except ImportError:
    Blueprint = None
    get_blueprint = None
    load_blueprint = None
    list_blueprints = None
    get_blueprint_design_space = None

# DSE system (updated for new infrastructure)
try:
    from .core.dse import (
        DSEInterface, parameter_sweep, batch_evaluate,
        discover_all_steps, optimize_design_space
    )
    from .core.dse.interface import DSEInterface as DSEInterfaceClass
    from .core.dse.types import (
        DSEConfiguration, DSEResult, DSEObjective, OptimizationObjective,
        SamplingStrategy, ParameterSpace
    )
    
    # Legacy compatibility
    SimpleDSEEngine = DSEInterfaceClass
    ExternalDSEAdapter = None  # Not implemented in new structure
    DSEAnalyzer = None  # Not implemented in new structure
    ParetoAnalyzer = None  # Not implemented in new structure
    OptimizationStrategy = SamplingStrategy
    create_dse_config_for_strategy = None  # Not implemented in new structure
    StrategySelector = None  # Not implemented in new structure
    COMMON_CONFIGS = None  # Not implemented in new structure
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

# Tools module (moved from core API) - backward compatibility
try:
    from .libraries.analysis.profiling import roofline_analysis, RooflineProfiler
except ImportError:
    roofline_analysis = None
    RooflineProfiler = None

# Backward compatibility imports for moved components
try:
    # Automation tools
    from .libraries.automation import batch_process, parameter_sweep, find_best, aggregate_stats
    # Legacy compatibility
    class batch:
        process = staticmethod(batch_process)
    class sweep:
        parameter_sweep = staticmethod(parameter_sweep)
        find_best = staticmethod(find_best)
        aggregate_stats = staticmethod(aggregate_stats)
except ImportError:
    batch = None
    sweep = None

try:
    # Kernel libraries (updated structure)
    from .libraries.kernels import (
        discover_all_kernels, get_kernel_by_name, find_kernels_for_operator,
        list_available_kernels, KernelRegistry
    )
    from .libraries.kernels import (
        load_kernel_package, find_compatible_kernels, select_optimal_kernel,
        validate_kernel_package, generate_finn_config
    )
    from .libraries.kernels.types import (
        KernelPackage, KernelRequirements, KernelSelection,
        OperatorType, BackendType
    )
    
    # Legacy compatibility
    class kernel_functions:
        discover_all_kernels = staticmethod(discover_all_kernels)
        load_kernel_package = staticmethod(load_kernel_package)
        find_compatible_kernels = staticmethod(find_compatible_kernels)
        select_optimal_kernel = staticmethod(select_optimal_kernel)
        
    class kernel_types:
        KernelPackage = KernelPackage
        KernelRequirements = KernelRequirements
        OperatorType = OperatorType
        BackendType = BackendType
        
except ImportError:
    kernel_functions = None
    kernel_types = None

try:
    # Transform libraries (updated structure)
    from .libraries.transforms import (
        discover_all_transforms, get_transform_by_name, find_transforms_by_type,
        list_available_transforms, TransformRegistry, TransformType
    )
    from .libraries.transforms import (
        cleanup_step, cleanup_advanced_step, qonnx_to_finn_step,
        streamlining_step, infer_hardware_step, generate_reference_io_step,
        shell_metadata_handover_step, remove_head_step, remove_tail_step,
        get_step, validate_step_sequence, discover_all_steps
    )
    
    # Legacy compatibility - create modules with step functions
    class bert:
        remove_head_step = staticmethod(remove_head_step)
        remove_tail_step = staticmethod(remove_tail_step)
    
    class cleanup:
        cleanup_step = staticmethod(cleanup_step)
        cleanup_advanced_step = staticmethod(cleanup_advanced_step)
    
    class conversion:
        qonnx_to_finn_step = staticmethod(qonnx_to_finn_step)
    
    class hardware:
        infer_hardware_step = staticmethod(infer_hardware_step)
    
    class metadata:
        shell_metadata_handover_step = staticmethod(shell_metadata_handover_step)
    
    class optimizations:
        pass  # Legacy placeholder
    
    class streamlining:
        streamlining_step = staticmethod(streamlining_step)
    
    class validation:
        generate_reference_io_step = staticmethod(generate_reference_io_step)
    
    # Operations (these would need to be imported from operations modules)
    convert_to_hw_layers = None  # Not directly available in new structure
    expand_norms = None  # Not directly available in new structure
    shuffle_helpers = None  # Not directly available in new structure
    
except ImportError:
    bert = cleanup = conversion = hardware = None
    metadata = optimizations = streamlining = validation = None
    convert_to_hw_layers = expand_norms = shuffle_helpers = None

try:
    # Hooks system (updated structure)
    from .core.hooks import (
        log_optimization_event, log_parameter_change, log_performance_metric,
        log_strategy_decision, log_dse_event, get_recent_events, get_events_by_type,
        register_event_handler, register_global_handler, HooksRegistry,
        discover_all_plugins, get_plugin_by_name, install_hook_plugin
    )
    from .core.hooks.types import (
        OptimizationEvent, EventHandler, SimpleMetric, ParameterChange,
        EventTypes, HooksPlugin
    )
    
    # Legacy compatibility
    class events:
        log_optimization_event = staticmethod(log_optimization_event)
        log_parameter_change = staticmethod(log_parameter_change)
        log_performance_metric = staticmethod(log_performance_metric)
        get_recent_events = staticmethod(get_recent_events)
        get_events_by_type = staticmethod(get_events_by_type)
    
    class hook_types:
        OptimizationEvent = OptimizationEvent
        EventHandler = EventHandler
        SimpleMetric = SimpleMetric
        ParameterChange = ParameterChange
    
    hook_examples = None  # Examples not directly exposed
    
except ImportError:
    events = hook_types = hook_examples = None
    
    # Legacy compatibility
    class events:
        log_optimization_event = staticmethod(log_optimization_event)
        log_parameter_change = staticmethod(log_parameter_change)
        log_performance_metric = staticmethod(log_performance_metric)
        get_recent_events = staticmethod(get_recent_events)
        get_events_by_type = staticmethod(get_events_by_type)
    
    class hook_types:
        OptimizationEvent = OptimizationEvent
        EventHandler = EventHandler
        SimpleMetric = SimpleMetric
        ParameterChange = ParameterChange
# New registry systems for auto-discovery and management
try:
    from .libraries.kernels import KernelRegistry, discover_all_kernels
    from .libraries.transforms import TransformRegistry, discover_all_transforms  
    from .libraries.analysis import AnalysisRegistry, discover_all_analysis_tools
    from .libraries.automation import AutomationRegistry, discover_all_automation_components
    from libraries.blueprints import BlueprintLibraryRegistry, discover_all_blueprints
    from .core.hooks import HooksRegistry, discover_all_plugins
    
    # Convenience function to discover all components
    def discover_all_components():
        """Discover all available BrainSmith components."""
        return {
            'kernels': discover_all_kernels(),
            'transforms': discover_all_transforms(), 
            'analysis_tools': discover_all_analysis_tools(),
            'automation_tools': discover_all_automation_components(),
            'blueprints': discover_all_blueprints(),
            'hooks_plugins': discover_all_plugins()
        }
        
except ImportError:
    KernelRegistry = TransformRegistry = AnalysisRegistry = None
    AutomationRegistry = BlueprintLibraryRegistry = HooksRegistry = None
    discover_all_components = lambda: {}
    
    hook_examples = None  # Examples not directly exposed
    
except ImportError:
    events = hook_types = hook_examples = None


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


# Simplified exports for API simplification
__all__ = [
    # Core toolchain
    'forge',
    'validate_blueprint',
    
    # Core data structures
    'DesignSpace',
    'DesignPoint',
    'ParameterDefinition',
    
    # Blueprint system (may be None if not available)
    'Blueprint',
    'get_blueprint',
    'load_blueprint',
    'list_blueprints',
    
    # DSE system (may be None if not available)
    'DSEInterface',
    'DSEAnalyzer',
    'ParetoAnalyzer',
    'DSEConfiguration',
    'DSEObjective',
    'OptimizationObjective',
    
    # Supplementary tools (moved from core API)
    'roofline_analysis',
    'RooflineProfiler',
    
    # Utility functions (preserved)
    'load_design_space',
    'sample_design_space',
    'analyze_dse_results',
    'get_pareto_frontier',
    'list_available_strategies',
    'recommend_strategy',
    
    # Core status
    'get_core_status'
]

# Version information (API Simplification)
__version__ = "0.5.0"  # Updated for API simplification
__author__ = "Microsoft Research"
__description__ = "Simplified FPGA accelerator design space exploration platform"
