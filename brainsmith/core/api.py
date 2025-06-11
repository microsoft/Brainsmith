"""
Simplified BrainSmith Core API - Single `forge` Function

This module provides the main Python API for BrainSmith,
implementing a single unified `forge` function that serves as
the core toolchain for FPGA accelerator design space exploration.
"""

from typing import Dict, Any, Optional
from pathlib import Path
import logging
import json

logger = logging.getLogger(__name__)

# Import hooks for optimization event logging
try:
    from ..hooks import log_optimization_event, log_strategy_decision, log_dse_event
    HOOKS_AVAILABLE = True
except ImportError:
    HOOKS_AVAILABLE = False
    log_optimization_event = lambda *args, **kwargs: None
    log_strategy_decision = lambda *args, **kwargs: None
    log_dse_event = lambda *args, **kwargs: None


def forge(
    model_path: str,
    blueprint_path: str,
    objectives: Dict[str, Any] = None,
    constraints: Dict[str, Any] = None,
    target_device: str = None,
    is_hw_graph: bool = False,
    build_core: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Core BrainSmith toolchain: DSE on input model to produce Dataflow Core.
    
    Args:
        model_path: Path to pre-quantized ONNX model
        blueprint_path: Path to blueprint YAML (design space specification)
        objectives: Target objectives (latency/throughput requirements)
        constraints: Hardware resource budgets, optimization priorities
        target_device: Target FPGA device specification
        is_hw_graph: If True, input is already a Dataflow Graph, skip to HW optimization
        build_core: If False, exit after Dataflow Graph generation
        output_dir: Optional output directory for results
        
    Returns:
        Dict containing:
        - dataflow_graph: ONNX graph of HWCustomOps describing Dataflow Core
        - dataflow_core: Stitched IP design (if build_core=True)
        - metrics: Performance and resource utilization metrics
        - analysis: DSE analysis and recommendations
    """
    logger.info(f"Starting forge with model: {model_path}, blueprint: {blueprint_path}")
    
    # 1. Input validation
    _validate_inputs(model_path, blueprint_path, objectives, constraints)
    
    # 2. Load and validate blueprint (hard error)
    blueprint = _load_and_validate_blueprint(blueprint_path)
    
    # 3. Setup DSE configuration
    dse_config = _setup_dse_configuration(blueprint, objectives, constraints, target_device)
    
    # 4. Branch based on is_hw_graph flag
    if is_hw_graph:
        # Input is already Dataflow Graph, skip to HW optimization
        logger.info("Hardware graph mode: Skipping to HW optimization")
        if HOOKS_AVAILABLE:
            log_strategy_decision('hw_optimization_only', 'Input is already a dataflow graph')
            log_dse_event('hw_optimization_start', {'mode': 'hw_graph'})
        
        dataflow_graph = _load_dataflow_graph(model_path)
        dse_results = _run_hw_optimization_dse(dataflow_graph, dse_config)
    else:
        # Standard flow: Model -> DSE -> Dataflow Graph
        logger.info("Standard mode: Running full model-to-hardware DSE")
        if HOOKS_AVAILABLE:
            log_strategy_decision('full_dse', 'Standard model-to-hardware flow')
            log_dse_event('full_dse_start', {'mode': 'model_to_hw'})
        
        dse_results = _run_full_dse(model_path, dse_config)
        dataflow_graph = dse_results.get('best_result', {}).get('dataflow_graph')
    
    # 5. Generate Dataflow Core if requested
    dataflow_core = None
    if build_core and dataflow_graph:
        logger.info("Generating Dataflow Core (stitched IP design)")
        dataflow_core = _generate_dataflow_core(dataflow_graph, dse_config)
    elif not build_core:
        logger.info("Checkpoint mode: Exiting after Dataflow Graph generation")
    
    # 6. Prepare results
    results = _assemble_results(dataflow_graph, dataflow_core, dse_results)
    
    # 7. Save results if output directory specified
    if output_dir:
        _save_forge_results(results, output_dir)
    
    # Log optimization completion
    if HOOKS_AVAILABLE:
        log_dse_event('optimization_complete', {
            'dataflow_graph_generated': dataflow_graph is not None,
            'dataflow_core_generated': dataflow_core is not None,
            'output_saved': output_dir is not None
        })
        log_optimization_event('optimization_end', {
            'success': True,
            'duration_info': 'completed_successfully'
        })
    
    logger.info("Forge process completed successfully")
    return results


def validate_blueprint(blueprint_path: str) -> tuple[bool, list[str]]:
    """
    Validate blueprint configuration - hard error if invalid.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    logger.info(f"Validating blueprint: {blueprint_path}")
    
    try:
        blueprint_data = _load_and_validate_blueprint(blueprint_path)
        logger.info(f"Blueprint validation successful: {blueprint_data.get('name', 'unnamed')}")
        return True, []
    except Exception as e:
        error_msg = f"Blueprint validation failed: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]


# Helper function implementations

def _validate_inputs(model_path: str, blueprint_path: str, objectives: Dict, constraints: Dict):
    """Validate all input parameters with descriptive error messages."""
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    if not model_path.lower().endswith('.onnx'):
        raise ValueError(f"Model must be ONNX format, got: {model_path}")
    
    if not blueprint_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError(f"Blueprint must be YAML format, got: {blueprint_path}")
    
    # Validate objectives format
    if objectives:
        for obj_name, obj_config in objectives.items():
            if not isinstance(obj_config, dict):
                raise ValueError(f"Objective '{obj_name}' must be a dictionary")
            if 'direction' not in obj_config:
                raise ValueError(f"Objective '{obj_name}' missing 'direction' field")
            if obj_config['direction'] not in ['maximize', 'minimize']:
                raise ValueError(f"Objective '{obj_name}' direction must be 'maximize' or 'minimize'")
    
    # Validate constraints format  
    if constraints:
        numeric_constraints = ['max_luts', 'max_dsps', 'max_brams', 'max_power', 'target_frequency']
        for key, value in constraints.items():
            if key in numeric_constraints and not isinstance(value, (int, float)):
                raise ValueError(f"Constraint '{key}' must be numeric, got {type(value)}")


def _load_and_validate_blueprint(blueprint_path: str):
    """Load and validate blueprint using simplified functions."""
    try:
        from ..blueprints.functions import load_blueprint_yaml, validate_blueprint_yaml
        
        # Load blueprint as simple dictionary
        blueprint_data = load_blueprint_yaml(blueprint_path)
        
        # Validate blueprint configuration
        is_valid, errors = validate_blueprint_yaml(blueprint_data)
        if not is_valid:
            raise ValueError(f"Blueprint validation failed:\n" + "\n".join(f"  - {error}" for error in errors))
        
        logger.info(f"Successfully loaded blueprint: {blueprint_data.get('name', 'unnamed')}")
        return blueprint_data
        
    except ImportError:
        raise RuntimeError("Blueprint system not available. Cannot proceed without valid blueprint.")
    except FileNotFoundError:
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    except Exception as e:
        raise ValueError(f"Failed to load blueprint '{blueprint_path}': {str(e)}")


def _setup_dse_configuration(blueprint_data, objectives, constraints, target_device):
    """Setup comprehensive DSE configuration using simplified blueprint data."""
    try:
        from ..dse.interface import DSEConfiguration, DSEObjective, OptimizationObjective
        from ..blueprints.functions import get_objectives, get_constraints
        
        # Use simple blueprint functions to extract configuration
        blueprint_objectives = get_objectives(blueprint_data)
        blueprint_constraints = get_constraints(blueprint_data)
        
        # Setup objectives (use provided objectives or blueprint defaults)
        dse_objectives = []
        final_objectives = objectives or blueprint_objectives
        
        for obj_name, obj_config in final_objectives.items():
            direction = OptimizationObjective.MAXIMIZE if obj_config['direction'] == 'maximize' else OptimizationObjective.MINIMIZE
            weight = obj_config.get('weight', 1.0)
            target = obj_config.get('target', None)
            
            dse_objectives.append(DSEObjective(
                name=obj_name,
                direction=direction,
                weight=weight,
                target_value=target
            ))
        
        # Setup constraints (merge provided constraints with blueprint defaults)
        dse_constraints = blueprint_constraints.copy()
        if constraints:
            dse_constraints.update(constraints)
        if target_device:
            dse_constraints['target_device'] = target_device
        
        return DSEConfiguration(
            design_space={},  # Simplified - no complex design space
            objectives=dse_objectives,
            constraints=dse_constraints,
            blueprint=blueprint_data
        )
    except ImportError:
        # Fallback configuration for when DSE system not available
        logger.warning("DSE system not available, using fallback configuration")
        from ..blueprints.functions import get_objectives, get_constraints
        
        return {
            'design_space': {},
            'objectives': objectives or get_objectives(blueprint_data),
            'constraints': constraints or get_constraints(blueprint_data),
            'blueprint': blueprint_data,
            'fallback_mode': True
        }


def _run_full_dse(model_path: str, dse_config):
    """Execute full model-to-hardware DSE pipeline."""
    try:
        from ..dse.interface import DSEInterface
        
        logger.info("Starting full DSE: Model analysis -> Transformation -> Kernel mapping -> HW optimization")
        
        dse_engine = DSEInterface(dse_config)
        
        # Execute complete pipeline
        results = dse_engine.explore_design_space(
            model_path=model_path,
            stages=['analysis', 'transformation', 'kernel_mapping', 'hw_optimization']
        )
        
        logger.info(f"DSE completed: {len(results.results) if hasattr(results, 'results') else 0} design points evaluated")
        return results
        
    except ImportError:
        logger.warning("DSE interface not available, using fallback DSE")
        return _fallback_dse(model_path, dse_config)


def _run_hw_optimization_dse(dataflow_graph, dse_config):
    """Execute hardware optimization DSE on existing Dataflow Graph."""
    try:
        from ..dse.interface import DSEInterface
        
        logger.info("Starting HW optimization DSE on existing Dataflow Graph")
        
        dse_engine = DSEInterface(dse_config)
        
        # Execute only HW optimization stage
        results = dse_engine.optimize_dataflow_graph(
            dataflow_graph=dataflow_graph,
            stages=['hw_optimization']
        )
        
        logger.info(f"HW optimization completed: {len(results.results) if hasattr(results, 'results') else 0} configurations evaluated")
        return results
        
    except ImportError:
        logger.warning("DSE interface not available, using fallback optimization")
        return _fallback_hw_optimization(dataflow_graph, dse_config)


def _load_dataflow_graph(model_path: str):
    """Load existing Dataflow Graph from ONNX file."""
    try:
        import onnx
        model = onnx.load(model_path)
        logger.info(f"Loaded Dataflow Graph with {len(model.graph.node)} nodes")
        return model
    except ImportError:
        raise RuntimeError("ONNX library not available for loading Dataflow Graph")
    except Exception as e:
        raise ValueError(f"Failed to load Dataflow Graph from {model_path}: {str(e)}")


def _generate_dataflow_core(dataflow_graph, dse_config):
    """Generate complete stitched IP design from Dataflow Graph."""
    try:
        from ..finn import build_accelerator
        
        logger.info("Generating Dataflow Core (stitched IP design)")
        
        # Extract blueprint configuration from DSE config
        blueprint_config = {}
        if hasattr(dse_config, 'blueprint'):
            # DSE config object with blueprint attribute
            blueprint_config = dse_config.blueprint
        elif isinstance(dse_config, dict) and 'blueprint' in dse_config:
            # Dictionary-based config with blueprint
            blueprint_config = dse_config['blueprint']
        
        # Set output directory
        output_dir = dse_config.get('output_dir', './output') if isinstance(dse_config, dict) else './output'
        
        # Use simplified FINN interface for build
        finn_result = build_accelerator(
            model_path=str(dataflow_graph),  # Convert dataflow_graph to path representation
            blueprint_config=blueprint_config,
            output_dir=output_dir
        )
        
        logger.info("Dataflow Core generation completed")
        return finn_result.to_dict() if hasattr(finn_result, 'to_dict') else finn_result
        
    except ImportError:
        logger.warning("FINN orchestration not available, using fallback core generation")
        return _fallback_core_generation(dataflow_graph, dse_config)


def _assemble_results(dataflow_graph, dataflow_core, dse_results):
    """Assemble final results dictionary."""
    
    # Import analysis hooks
    from ..analysis import expose_analysis_data, register_analyzer, get_raw_data
    
    results = {
        'dataflow_graph': {
            'onnx_model': dataflow_graph,
            'metadata': {
                'kernel_mapping': getattr(dse_results, 'kernel_mapping', {}) if hasattr(dse_results, 'kernel_mapping') else {},
                'resource_estimates': getattr(dse_results, 'resource_estimates', {}) if hasattr(dse_results, 'resource_estimates') else {},
                'performance_estimates': getattr(dse_results, 'performance_estimates', {}) if hasattr(dse_results, 'performance_estimates') else {}
            }
        },
        'dataflow_core': dataflow_core,
        'dse_results': {
            'best_configuration': getattr(dse_results, 'best_result', {}) if hasattr(dse_results, 'best_result') else {},
            'pareto_frontier': getattr(dse_results, 'pareto_points', []) if hasattr(dse_results, 'pareto_points') else [],
            'exploration_history': getattr(dse_results, 'results', []) if hasattr(dse_results, 'results') else [],
            'convergence_metrics': getattr(dse_results, 'convergence', {}) if hasattr(dse_results, 'convergence') else {}
        },
        'metrics': _extract_metrics(dse_results),
        'analysis': _generate_analysis(dse_results),
        
        # NEW: Analysis hooks for external tools
        'analysis_data': expose_analysis_data(getattr(dse_results, 'results', [])),
        'analysis_hooks': {
            'register_analyzer': register_analyzer,
            'get_raw_data': lambda: get_raw_data(getattr(dse_results, 'results', [])),
            'available_adapters': ['pandas', 'scipy', 'sklearn']
        }
    }
    
    return results


def _extract_metrics(dse_results):
    """Extract performance and resource metrics from DSE results."""
    if hasattr(dse_results, 'best_result') and dse_results.best_result:
        best_result = dse_results.best_result
        return {
            'performance': {
                'throughput_ops_sec': getattr(best_result, 'throughput', 0.0),
                'latency_ms': getattr(best_result, 'latency', 0.0),
                'frequency_mhz': getattr(best_result, 'frequency', 0.0)
            },
            'resources': {
                'lut_utilization': getattr(best_result, 'lut_util', 0.0),
                'dsp_utilization': getattr(best_result, 'dsp_util', 0.0),
                'bram_utilization': getattr(best_result, 'bram_util', 0.0),
                'power_consumption_w': getattr(best_result, 'power', 0.0)
            }
        }
    else:
        return {
            'performance': {'throughput_ops_sec': 0.0, 'latency_ms': 0.0, 'frequency_mhz': 0.0},
            'resources': {'lut_utilization': 0.0, 'dsp_utilization': 0.0, 'bram_utilization': 0.0, 'power_consumption_w': 0.0}
        }


def _generate_analysis(dse_results):
    """Generate analysis and recommendations from DSE results."""
    num_results = len(getattr(dse_results, 'results', []))
    
    return {
        'design_space_coverage': min(1.0, num_results / 100.0),  # Estimate based on results count
        'optimization_quality': 0.8 if num_results > 10 else 0.5,  # Simple quality metric
        'recommendations': [
            "Consider increasing evaluation budget for better optimization",
            "Review resource constraints for feasibility",
            "Validate blueprint configuration for completeness"
        ],
        'warnings': [] if num_results > 0 else ["No valid design points found - check constraints"]
    }


def _save_forge_results(results: Dict[str, Any], output_dir: str):
    """Save forge results to output directory."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results as JSON
        results_copy = results.copy()
        # Remove ONNX model for JSON serialization
        if 'dataflow_graph' in results_copy and 'onnx_model' in results_copy['dataflow_graph']:
            results_copy['dataflow_graph']['onnx_model'] = str(results_copy['dataflow_graph']['onnx_model'])
        
        with open(output_path / "forge_results.json", 'w') as f:
            json.dump(results_copy, f, indent=2, default=str)
        
        # Save ONNX model separately if available
        if results.get('dataflow_graph', {}).get('onnx_model'):
            try:
                import onnx
                onnx.save(results['dataflow_graph']['onnx_model'], str(output_path / "dataflow_graph.onnx"))
            except ImportError:
                logger.warning("ONNX library not available, skipping ONNX model save")
        
        logger.info(f"Forge results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save forge results: {e}")


# Fallback implementations for when components are not available

def _fallback_dse(model_path: str, dse_config):
    """Fallback DSE when full DSE system not available."""
    logger.warning("Using fallback DSE implementation")
    
    class FallbackResult:
        def __init__(self):
            self.results = []
            self.best_result = {
                'dataflow_graph': None,
                'throughput': 100.0,
                'latency': 10.0,
                'lut_util': 0.5,
                'dsp_util': 0.6
            }
    
    return FallbackResult()


def _fallback_hw_optimization(dataflow_graph, dse_config):
    """Fallback HW optimization when DSE system not available."""
    logger.warning("Using fallback HW optimization")
    
    class FallbackOptimization:
        def __init__(self):
            self.results = []
            self.best_result = {
                'dataflow_graph': dataflow_graph,
                'throughput': 150.0,
                'latency': 8.0,
                'frequency': 200.0
            }
    
    return FallbackOptimization()


def _fallback_core_generation(dataflow_graph, dse_config):
    """Fallback core generation when FINN orchestration not available."""
    logger.warning("Using fallback core generation")
    
    return {
        'ip_files': [],
        'synthesis_results': {'status': 'fallback_mode'},
        'driver_code': {},
        'bitstream': None,
        'fallback': True
    }