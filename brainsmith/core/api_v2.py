"""
Clean Blueprint V2 API - The future of BrainSmith

This module provides the new forge_v2() function with no legacy baggage,
designed for Blueprint V2 design space exploration with real FINN integration.
"""

from typing import Dict, Any, Optional, List
from pathlib import Path
import logging
import time

from .blueprint_v2 import DesignSpaceDefinition, load_blueprint_v2
from .dse_v2.space_explorer import DesignSpaceExplorer, ExplorationConfig
from .finn_v2 import FINNEvaluationBridge

logger = logging.getLogger(__name__)


def forge_v2(
    model_path: str,
    blueprint_path: str,
    objectives: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    target_device: Optional[str] = None,
    output_dir: Optional[str] = None,
    dse_config: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Clean Blueprint V2 API - The future of BrainSmith.
    
    No legacy baggage, no backward compatibility, just the best system we can build.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to Blueprint V2 YAML
        objectives: Optimization objectives (override blueprint defaults)
        constraints: Resource constraints (override blueprint defaults)  
        target_device: Target FPGA device
        output_dir: Output directory for results
        dse_config: DSE strategy configuration
        
    Returns:
        Clean results dictionary with:
        - best_design: Optimal design point found
        - pareto_frontier: Multi-objective optimization results
        - exploration_summary: DSE execution statistics
        - build_artifacts: FINN build outputs (if successful)
    """
    start_time = time.time()
    logger.info(f"Starting forge_v2 with model: {model_path}, blueprint: {blueprint_path}")
    
    try:
        # 1. Load and validate Blueprint V2 (strict validation)
        logger.info("Loading Blueprint V2 with strict validation")
        design_space = _load_blueprint_v2_strict(blueprint_path)
        
        # 2. Override blueprint objectives/constraints if provided
        if objectives or constraints:
            design_space = _apply_overrides(design_space, objectives, constraints, target_device)
        
        # 3. Create DSE configuration
        exploration_config = _create_exploration_config(dse_config, output_dir)
        
        # 4. Create design space explorer with FINN integration
        logger.info("Creating DesignSpaceExplorer with real FINN integration")
        explorer = DesignSpaceExplorer(design_space, exploration_config)
        
        # 5. Execute design space exploration with real FINN evaluations
        logger.info("Starting design space exploration with real FINN builds")
        exploration_results = explorer.explore_design_space(model_path)
        
        # 6. Generate clean, well-structured results
        results = _format_clean_results(exploration_results, start_time)
        
        # 7. Save results if output directory specified
        if output_dir:
            _save_results_v2(results, output_dir)
        
        logger.info(f"forge_v2 completed successfully in {time.time() - start_time:.2f}s")
        return results
        
    except Exception as e:
        error_time = time.time() - start_time
        logger.error(f"forge_v2 failed after {error_time:.2f}s: {e}")
        
        # Return clean error response
        return {
            'success': False,
            'error': str(e),
            'execution_time': error_time,
            'best_design': None,
            'pareto_frontier': [],
            'exploration_summary': {
                'total_evaluations': 0,
                'successful_evaluations': 0,
                'error_message': str(e)
            }
        }


def validate_blueprint_v2(blueprint_path: str) -> tuple[bool, List[str]]:
    """
    Validate Blueprint V2 configuration with strict checking.
    
    Args:
        blueprint_path: Path to Blueprint V2 YAML file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    logger.info(f"Validating Blueprint V2: {blueprint_path}")
    
    try:
        design_space = _load_blueprint_v2_strict(blueprint_path)
        
        # Additional strict validations
        errors = []
        
        # Check for required objectives
        if not design_space.objectives:
            errors.append("Blueprint V2 must define optimization objectives")
        
        # Check for design space completeness
        total_components = (
            len(design_space.nodes.canonical_ops.available) +
            len(design_space.nodes.hw_kernels.available) +
            len(design_space.transforms.model_topology.available)
        )
        
        if total_components == 0:
            errors.append("Blueprint V2 must define at least some design space components")
        
        # Check DSE strategy compatibility
        if design_space.dse_strategies and design_space.dse_strategies.primary_strategy:
            strategy_errors = _validate_strategy_compatibility(design_space)
            errors.extend(strategy_errors)
        
        is_valid = len(errors) == 0
        if is_valid:
            logger.info("Blueprint V2 validation successful")
        else:
            logger.warning(f"Blueprint V2 validation found {len(errors)} issues")
        
        return is_valid, errors
        
    except Exception as e:
        error_msg = f"Blueprint V2 validation failed: {str(e)}"
        logger.error(error_msg)
        return False, [error_msg]


def _load_blueprint_v2_strict(blueprint_path: str) -> DesignSpaceDefinition:
    """Load Blueprint V2 with strict validation - no compromises."""
    
    # Validate file exists and is readable
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    if not blueprint_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError(f"Blueprint must be YAML format, got: {blueprint_path}")
    
    try:
        # Load using Blueprint V2 parser
        design_space = load_blueprint_v2(blueprint_path)
        
        # Strict validation
        _validate_blueprint_v2_strict(design_space)
        
        logger.info(f"Successfully loaded Blueprint V2: {design_space.name}")
        return design_space
        
    except Exception as e:
        raise ValueError(f"Failed to load Blueprint V2 '{blueprint_path}': {str(e)}")


def _validate_blueprint_v2_strict(design_space: DesignSpaceDefinition) -> None:
    """Strict validation with clear error messages."""
    
    errors = []
    
    # Validate basic structure
    if not design_space.name:
        errors.append("Blueprint must have a name")
    
    # Validate design space content
    if not design_space.nodes:
        errors.append("Blueprint must define nodes design space")
    
    if not design_space.transforms:
        errors.append("Blueprint must define transforms design space")
    
    # Validate objectives for DSE
    if not design_space.objectives:
        errors.append("Blueprint V2 must define optimization objectives for DSE")
    
    # Validate DSE strategies
    if design_space.dse_strategies:
        if not design_space.dse_strategies.primary_strategy:
            errors.append("Blueprint must specify primary DSE strategy")
        
        if not design_space.dse_strategies.strategies:
            errors.append("Blueprint must define at least one DSE strategy")
    
    if errors:
        error_msg = "Blueprint V2 validation failed:\n" + "\n".join(f"  - {error}" for error in errors)
        raise ValueError(error_msg)


def _apply_overrides(design_space: DesignSpaceDefinition, 
                    objectives: Optional[Dict[str, Any]], 
                    constraints: Optional[Dict[str, Any]],
                    target_device: Optional[str]) -> DesignSpaceDefinition:
    """Apply runtime overrides to design space."""
    
    # Create a copy to avoid modifying original
    import copy
    modified_space = copy.deepcopy(design_space)
    
    # Override objectives
    if objectives:
        logger.info(f"Applying objective overrides: {objectives}")
        # Convert dict to Objective objects and replace
        from .blueprint_v2 import Objective
        modified_space.objectives = [
            Objective(name=name, **config) for name, config in objectives.items()
        ]
    
    # Override constraints
    if constraints:
        logger.info(f"Applying constraint overrides: {constraints}")
        # Merge with existing constraints
        if not modified_space.constraints:
            modified_space.constraints = []
        
        from .blueprint_v2 import Constraint
        for name, value in constraints.items():
            constraint = Constraint(name=name, value=value)
            modified_space.constraints.append(constraint)
    
    # Override target device
    if target_device:
        logger.info(f"Setting target device: {target_device}")
        # Add as constraint
        from .blueprint_v2 import Constraint
        device_constraint = Constraint(name="target_device", value=target_device)
        if not modified_space.constraints:
            modified_space.constraints = []
        modified_space.constraints.append(device_constraint)
    
    return modified_space


def _create_exploration_config(dse_config: Optional[Dict[str, Any]], 
                             output_dir: Optional[str]) -> ExplorationConfig:
    """Create exploration configuration."""
    
    config_params = {
        'max_evaluations': 50,  # Reasonable default for real FINN builds
        'parallel_evaluations': 1,  # Conservative for FINN resource usage
        'enable_caching': True,
        'early_termination_patience': 10,
        'checkpoint_frequency': 5
    }
    
    # Apply user overrides
    if dse_config:
        config_params.update(dse_config)
    
    # Set cache directory
    if output_dir:
        config_params['cache_directory'] = str(Path(output_dir) / "dse_cache")
    
    logger.info(f"Created exploration config: max_evaluations={config_params['max_evaluations']}")
    return ExplorationConfig(**config_params)


def _validate_strategy_compatibility(design_space: DesignSpaceDefinition) -> List[str]:
    """Validate strategy compatibility with blueprint."""
    
    errors = []
    
    # Check if primary strategy exists
    primary_strategy = design_space.dse_strategies.primary_strategy
    available_strategies = list(design_space.dse_strategies.strategies.keys())
    
    if primary_strategy not in available_strategies:
        errors.append(f"Primary strategy '{primary_strategy}' not found in defined strategies")
    
    # Check strategy-objective compatibility
    for strategy_name, strategy_config in design_space.dse_strategies.strategies.items():
        if hasattr(strategy_config, 'objectives'):
            strategy_objectives = strategy_config.objectives
            blueprint_objectives = [obj.name for obj in design_space.objectives]
            
            for obj_name in strategy_objectives:
                if obj_name not in blueprint_objectives:
                    errors.append(f"Strategy '{strategy_name}' references undefined objective '{obj_name}'")
    
    return errors


def _format_clean_results(exploration_results, start_time: float) -> Dict[str, Any]:
    """Generate clean, well-structured results."""
    
    execution_time = time.time() - start_time
    
    return {
        'success': True,
        'execution_time': execution_time,
        
        # Best design point found
        'best_design': {
            'combination': exploration_results.best_combination.to_dict() if exploration_results.best_combination else None,
            'score': exploration_results.best_score,
            'metrics': _extract_best_metrics(exploration_results)
        },
        
        # Multi-objective optimization results
        'pareto_frontier': [
            {
                'combination': combo.to_dict(),
                'metrics': _extract_combination_metrics(combo, exploration_results.performance_data)
            }
            for combo in exploration_results.pareto_frontier
        ],
        
        # DSE execution statistics
        'exploration_summary': {
            'total_evaluations': len(exploration_results.all_combinations),
            'successful_evaluations': len([r for r in exploration_results.performance_data if r.get('success', False)]),
            'pareto_frontier_size': len(exploration_results.pareto_frontier),
            'execution_time': execution_time,
            'strategy_metadata': exploration_results.strategy_metadata
        },
        
        # Build artifacts (if available)
        'build_artifacts': exploration_results.execution_stats,
        
        # Raw exploration data for advanced analysis
        'raw_data': {
            'all_combinations': [combo.to_dict() for combo in exploration_results.all_combinations],
            'performance_data': exploration_results.performance_data,
            'exploration_summary': exploration_results.exploration_summary
        }
    }


def _extract_best_metrics(exploration_results) -> Dict[str, Any]:
    """Extract metrics for best design."""
    if not exploration_results.performance_data:
        return {}
    
    # Find best performing result
    successful_results = [r for r in exploration_results.performance_data if r.get('success', False)]
    if not successful_results:
        return {}
    
    best_result = max(successful_results, key=lambda x: x.get('primary_metric', 0))
    return best_result.get('metrics', {})


def _extract_combination_metrics(combination, performance_data: List[Dict]) -> Dict[str, Any]:
    """Extract metrics for specific combination."""
    for result in performance_data:
        if result.get('combination') == combination:
            return result.get('metrics', {})
    return {}


def _save_results_v2(results: Dict[str, Any], output_dir: str) -> None:
    """Save forge_v2 results to output directory."""
    
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save main results
        import json
        with open(output_path / "forge_v2_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save summary
        summary = {
            'success': results['success'],
            'execution_time': results['execution_time'],
            'total_evaluations': results['exploration_summary']['total_evaluations'],
            'best_score': results['best_design']['score'],
            'pareto_frontier_size': len(results['pareto_frontier'])
        }
        
        with open(output_path / "forge_v2_summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"forge_v2 results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save forge_v2 results: {e}")