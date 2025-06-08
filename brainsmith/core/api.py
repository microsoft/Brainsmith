"""
Python API using existing components in extensible structure.

This module provides the main Python API functions for Brainsmith,
implementing hierarchical exit points with extensible structure
around existing functionality while maintaining backward compatibility.
"""

from typing import Tuple, Dict, Any, Optional, List
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)

def brainsmith_explore(model_path: str, 
                      blueprint_path: str,
                      exit_point: str = "dataflow_generation",
                      output_dir: Optional[str] = None,
                      **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Main exploration API using existing components only.
    
    Provides hierarchical exit points with extensible structure
    around current functionality while maintaining compatibility
    with existing workflows.
    
    Args:
        model_path: Path to quantized ONNX model
        blueprint_path: Path to blueprint YAML configuration
        exit_point: Analysis exit point ('roofline', 'dataflow_analysis', 'dataflow_generation')
        output_dir: Optional output directory for results
        **kwargs: Additional configuration options
    
    Returns:
        Tuple of (DSE results, comprehensive analysis)
    """
    logger.info(f"Starting brainsmith_explore with exit_point: {exit_point}")
    
    # Validate inputs using existing validation patterns
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not Path(blueprint_path).exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    # Validate exit point
    valid_exit_points = ["roofline", "dataflow_analysis", "dataflow_generation"]
    if exit_point not in valid_exit_points:
        raise ValueError(f"Invalid exit point: {exit_point}. Must be one of {valid_exit_points}")
    
    try:
        # Load blueprint using existing Blueprint class with extensions
        blueprint = _load_and_validate_blueprint(blueprint_path, model_path)
        
        # Create orchestrator using existing components
        from .design_space_orchestrator import DesignSpaceOrchestrator
        orchestrator = DesignSpaceOrchestrator(blueprint)
        
        # Execute exploration with specified exit point
        results = orchestrator.orchestrate_exploration(exit_point)
        
        # Generate analysis using existing analysis tools
        analysis = _generate_exploration_analysis(results, orchestrator, **kwargs)
        
        # Save results if output directory specified
        if output_dir:
            _save_results_existing(results, analysis, output_dir, orchestrator)
        
        logger.info(f"Exploration completed successfully with exit_point: {exit_point}")
        
        return results, analysis
        
    except Exception as e:
        logger.error(f"Exploration failed: {e}")
        # Create fallback results for graceful error handling
        fallback_results, fallback_analysis = _create_fallback_results(
            model_path, blueprint_path, exit_point, str(e)
        )
        
        if output_dir:
            _save_results_existing(fallback_results, fallback_analysis, output_dir, None)
        
        return fallback_results, fallback_analysis

def brainsmith_roofline(model_path: str, blueprint_path: str, 
                       output_dir: Optional[str] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Roofline analysis using existing tools (Exit Point 1).
    
    Performs quick analytical performance bounds estimation
    without hardware generation using existing analysis capabilities.
    
    Args:
        model_path: Path to quantized ONNX model
        blueprint_path: Path to blueprint YAML configuration  
        output_dir: Optional output directory for results
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (DSE results, roofline analysis)
    """
    logger.info("Starting roofline analysis using existing tools")
    return brainsmith_explore(model_path, blueprint_path, "roofline", output_dir, **kwargs)

def brainsmith_dataflow_analysis(model_path: str, blueprint_path: str,
                                output_dir: Optional[str] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Dataflow analysis using existing transforms and estimation (Exit Point 2).
    
    Applies existing transforms and provides dataflow-level performance
    estimation without RTL generation.
    
    Args:
        model_path: Path to quantized ONNX model
        blueprint_path: Path to blueprint YAML configuration
        output_dir: Optional output directory for results
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (DSE results, dataflow analysis)
    """
    logger.info("Starting dataflow analysis using existing transforms")
    return brainsmith_explore(model_path, blueprint_path, "dataflow_analysis", output_dir, **kwargs)

def brainsmith_generate(model_path: str, blueprint_path: str,
                       output_dir: Optional[str] = None, **kwargs) -> Tuple[Any, Dict[str, Any]]:
    """
    Full RTL/HLS generation using existing FINN flow (Exit Point 3).
    
    Performs complete optimization and generation using existing
    DataflowBuildConfig workflow and optimization strategies.
    
    Args:
        model_path: Path to quantized ONNX model
        blueprint_path: Path to blueprint YAML configuration
        output_dir: Optional output directory for results
        **kwargs: Additional configuration options
        
    Returns:
        Tuple of (DSE results, generation results)
    """
    logger.info("Starting full generation using existing FINN flow")
    return brainsmith_explore(model_path, blueprint_path, "dataflow_generation", output_dir, **kwargs)

# Backward compatibility layer maintaining existing functionality
def explore_design_space(model_path: str, blueprint_name: str, **kwargs):
    """
    Backward compatibility wrapper for existing API.
    
    Maintains 100% compatibility with current usage patterns while
    routing to new extensible architecture when appropriate.
    
    Args:
        model_path: Path to model file
        blueprint_name: Blueprint name or path
        **kwargs: Additional arguments matching existing API
        
    Returns:
        Results in existing format for compatibility
    """
    logger.info(f"Legacy API called: explore_design_space({model_path}, {blueprint_name})")
    
    # Warn about legacy usage while maintaining support
    warnings.warn(
        "explore_design_space is legacy API. Consider using brainsmith_explore for enhanced features.",
        DeprecationWarning,
        stacklevel=2
    )
    
    try:
        # Check if blueprint_name is a path or legacy name
        if Path(blueprint_name).exists():
            # Use new API with blueprint path
            results, analysis = brainsmith_explore(
                model_path, 
                blueprint_name, 
                exit_point=kwargs.get('exit_point', 'dataflow_generation'),
                **kwargs
            )
            
            # Convert to legacy format for compatibility
            return _convert_to_legacy_format(results, analysis)
        else:
            # Route to existing legacy system if available
            return _route_to_existing_legacy_system(model_path, blueprint_name, **kwargs)
            
    except Exception as e:
        logger.error(f"Legacy API failed: {e}")
        # Return legacy-compatible error format
        return _create_legacy_error_result(model_path, blueprint_name, str(e))

def validate_blueprint(blueprint_path: str) -> Tuple[bool, List[str]]:
    """
    Validate blueprint configuration for existing components.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    logger.info(f"Validating blueprint: {blueprint_path}")
    
    try:
        blueprint = _load_blueprint_for_validation(blueprint_path)
        return blueprint.validate_library_config()
    except Exception as e:
        logger.error(f"Blueprint validation failed: {e}")
        return False, [f"Validation error: {str(e)}"]

# Internal helper functions

def _load_and_validate_blueprint(blueprint_path: str, model_path: str):
    """Load and validate blueprint with model path assignment."""
    try:
        # Import existing Blueprint class
        from ..blueprints.base import Blueprint
        
        # Load blueprint from YAML file
        blueprint = Blueprint.from_yaml_file(Path(blueprint_path))
        
        # Validate blueprint supports library-driven DSE using existing components
        if hasattr(blueprint, 'validate_library_config'):
            is_valid, errors = blueprint.validate_library_config()
            if not is_valid:
                raise ValueError(f"Blueprint validation failed: {'; '.join(errors)}")
        
        # Store model path for orchestrator access
        blueprint.model_path = model_path
        
        logger.info(f"Blueprint loaded and validated: {blueprint.name if hasattr(blueprint, 'name') else 'unnamed'}")
        return blueprint
        
    except ImportError:
        # Fallback for when Blueprint class is not available
        logger.warning("Blueprint class not available, using mock blueprint")
        return _create_mock_blueprint(blueprint_path, model_path)
    except Exception as e:
        logger.error(f"Failed to load blueprint: {e}")
        raise

def _generate_exploration_analysis(results, orchestrator, **kwargs) -> Dict[str, Any]:
    """Generate comprehensive analysis using existing analysis tools."""
    analysis = {
        'exit_point': results.analysis.get('exit_point', 'unknown'),
        'method': results.analysis.get('method', 'existing_tools'),
        'components_source': 'existing_only',
        'orchestrator_history': orchestrator.get_orchestration_history(),
        'libraries_status': _get_libraries_status(orchestrator),
        'analysis_timestamp': str(Path.cwd()),  # Placeholder for timestamp
        'configuration': kwargs
    }
    
    # Add exit-point specific analysis
    if results.analysis.get('exit_point') == 'roofline':
        analysis['roofline_specific'] = _analyze_roofline_results(results)
    elif results.analysis.get('exit_point') == 'dataflow_analysis':
        analysis['dataflow_specific'] = _analyze_dataflow_results(results)
    elif results.analysis.get('exit_point') == 'dataflow_generation':
        analysis['generation_specific'] = _analyze_generation_results(results)
    
    return analysis

def _get_libraries_status(orchestrator) -> Dict[str, Any]:
    """Get status of all libraries in orchestrator."""
    status = {}
    
    for lib_name, lib in orchestrator.libraries.items():
        status[lib_name] = {
            'available': lib is not None,
            'type': 'existing_component_library',
            'class_name': lib.__class__.__name__ if lib else None
        }
    
    return status

def _analyze_roofline_results(results) -> Dict[str, Any]:
    """Analyze roofline-specific results."""
    return {
        'analysis_type': 'roofline_bounds',
        'roofline_data': results.analysis.get('roofline_results', {}),
        'performance_bounds': 'computed_using_existing_tools',
        'recommendations': ['Use dataflow_analysis for more detailed estimation']
    }

def _analyze_dataflow_results(results) -> Dict[str, Any]:
    """Analyze dataflow-specific results."""
    return {
        'analysis_type': 'dataflow_estimation',
        'transformed_model': results.analysis.get('transformed_model', {}),
        'kernel_mapping': results.analysis.get('kernel_mapping', {}),
        'performance_estimates': results.analysis.get('performance_estimates', {}),
        'recommendations': ['Use dataflow_generation for actual RTL/HLS files']
    }

def _analyze_generation_results(results) -> Dict[str, Any]:
    """Analyze generation-specific results."""
    generation_data = results.analysis.get('generation_results', {})
    return {
        'analysis_type': 'complete_generation',
        'rtl_files_count': len(generation_data.get('rtl_files', [])),
        'hls_files_count': len(generation_data.get('hls_files', [])),
        'synthesis_status': generation_data.get('synthesis_results', {}).get('status', 'unknown'),
        'recommendations': ['Review generated files and synthesis results']
    }

def _save_results_existing(results, analysis: Dict, output_dir: str, orchestrator):
    """Save exploration results using existing save functionality."""
    try:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save raw results using existing result save method
        if hasattr(results, 'save'):
            results.save(output_path / "dse_results.json")
        else:
            # Fallback save method
            import json
            with open(output_path / "dse_results.json", 'w') as f:
                json.dump({
                    'results': str(results),
                    'type': 'fallback_save'
                }, f, indent=2)
        
        # Save analysis using existing export methods
        import json
        with open(output_path / "analysis.json", 'w') as f:
            json.dump(analysis, f, indent=2, default=str)
        
        # Generate report if orchestrator has analysis library
        if orchestrator and 'analysis' in orchestrator.libraries:
            try:
                analyzer = orchestrator.libraries['analysis']
                if hasattr(analyzer, 'generate_report'):
                    report = analyzer.generate_report(results, analysis)
                    with open(output_path / "report.html", 'w') as f:
                        f.write(report)
            except Exception as e:
                logger.warning(f"Could not generate report: {e}")
        
        logger.info(f"Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Failed to save results: {e}")

def _create_fallback_results(model_path: str, blueprint_path: str, exit_point: str, error_msg: str):
    """Create fallback results when exploration fails."""
    # Import result class
    try:
        from ..core.result import DSEResult
        
        fallback_results = DSEResult(
            results=[],
            analysis={
                'exit_point': exit_point,
                'status': 'failed',
                'error': error_msg,
                'fallback': True,
                'model_path': model_path,
                'blueprint_path': blueprint_path
            }
        )
    except ImportError:
        # Mock result if DSEResult not available
        fallback_results = {
            'results': [],
            'analysis': {
                'exit_point': exit_point,
                'status': 'failed',
                'error': error_msg,
                'fallback': True
            }
        }
    
    fallback_analysis = {
        'exit_point': exit_point,
        'method': 'fallback_error_handling',
        'error': error_msg,
        'status': 'failed',
        'components_source': 'none_due_to_error'
    }
    
    return fallback_results, fallback_analysis

def _route_to_existing_legacy_system(model_path: str, blueprint_name: str, **kwargs):
    """Route to existing legacy system when available."""
    try:
        # Try to import and use existing explore_design_space function
        from ..legacy.compatibility import existing_explore_design_space
        return existing_explore_design_space(model_path, blueprint_name, **kwargs)
    except ImportError:
        # Fallback if legacy system not available
        logger.warning("Legacy system not available, creating mock result")
        return _create_legacy_error_result(model_path, blueprint_name, "Legacy system not available")

def _convert_to_legacy_format(results, analysis) -> Dict[str, Any]:
    """Convert new results format to legacy format for compatibility."""
    return {
        'results': results,
        'analysis': analysis,
        'legacy_format': True,
        'converted_from': 'new_api'
    }

def _create_legacy_error_result(model_path: str, blueprint_name: str, error_msg: str) -> Dict[str, Any]:
    """Create legacy-compatible error result."""
    return {
        'error': error_msg,
        'model_path': model_path,
        'blueprint_name': blueprint_name,
        'legacy_format': True,
        'status': 'error'
    }

def _load_blueprint_for_validation(blueprint_path: str):
    """Load blueprint for validation purposes."""
    try:
        from ..blueprints.base import Blueprint
        return Blueprint.from_yaml_file(Path(blueprint_path))
    except ImportError:
        return _create_mock_blueprint(blueprint_path, None)

def _create_mock_blueprint(blueprint_path: str, model_path: Optional[str]):
    """Create mock blueprint when Blueprint class not available."""
    class MockBlueprint:
        def __init__(self, path, model_path):
            self.path = path
            self.model_path = model_path
            self.name = f"mock_blueprint_{Path(path).stem}"
        
        def validate_library_config(self):
            return True, []
        
        def get_finn_legacy_config(self):
            return {}
    
    return MockBlueprint(blueprint_path, model_path)

# Workflow convenience functions
def brainsmith_workflow(model_path: str, blueprint_path: str, workflow_type: str = "standard", **kwargs):
    """
    Execute predefined workflows using existing components.
    
    Args:
        model_path: Path to model file
        blueprint_path: Path to blueprint configuration
        workflow_type: Workflow type ('fast', 'standard', 'comprehensive')
        **kwargs: Additional workflow configuration
        
    Returns:
        Workflow execution results
    """
    logger.info(f"Executing predefined workflow: {workflow_type}")
    
    # Map workflow types to exit points
    workflow_mapping = {
        'fast': 'roofline',
        'standard': 'dataflow_analysis', 
        'comprehensive': 'dataflow_generation'
    }
    
    exit_point = workflow_mapping.get(workflow_type, 'dataflow_analysis')
    
    return brainsmith_explore(model_path, blueprint_path, exit_point, **kwargs)