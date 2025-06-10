"""
Batch Processing Automation

Simple utilities for processing multiple models or configurations in batch.
"""

from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def batch_process(
    model_blueprint_pairs: List[Tuple[str, str]],
    common_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4,
    progress_callback: Optional[callable] = None
) -> List[Dict[str, Any]]:
    """
    Process multiple model/blueprint pairs in batch.
    
    Args:
        model_blueprint_pairs: List of (model_path, blueprint_path) tuples
        common_config: Common configuration for all runs
        max_workers: Number of parallel workers
        progress_callback: Optional progress callback
        
    Returns:
        List of forge() results
        
    Example:
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml"),
            ("model3.onnx", "blueprint3.yaml")
        ])
    """
    from ..core.api import forge
    
    total_pairs = len(model_blueprint_pairs)
    logger.info(f"Starting batch processing of {total_pairs} model/blueprint pairs")
    
    common_config = common_config or {}
    results = []
    
    def process_single_pair(pair: Tuple[str, str], index: int) -> Dict[str, Any]:
        """Process single model/blueprint pair."""
        model_path, blueprint_path = pair
        
        try:
            # Run forge with common configuration
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                **common_config
            )
            
            # Add batch processing metadata
            result['batch_info'] = {
                'model_path': model_path,
                'blueprint_path': blueprint_path,
                'batch_index': index,
                'success': True
            }
            
            if progress_callback:
                progress_callback(index + 1, total_pairs, (model_path, blueprint_path))
            
            return result
            
        except Exception as e:
            logger.error(f"Batch processing failed for {model_path}: {e}")
            return {
                'batch_info': {
                    'model_path': model_path,
                    'blueprint_path': blueprint_path,
                    'batch_index': index,
                    'success': False,
                    'error': str(e)
                }
            }
    
    # Process with parallel execution
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_pair = {
                executor.submit(process_single_pair, pair, i): (pair, i)
                for i, pair in enumerate(model_blueprint_pairs)
            }
            
            for future in as_completed(future_to_pair):
                result = future.result()
                results.append(result)
    else:
        # Sequential processing
        for i, pair in enumerate(model_blueprint_pairs):
            result = process_single_pair(pair, i)
            results.append(result)
    
    # Sort results by batch index
    results.sort(key=lambda x: x.get('batch_info', {}).get('batch_index', 0))
    
    successful_runs = sum(1 for r in results if r.get('batch_info', {}).get('success', False))
    logger.info(f"Batch processing completed: {successful_runs}/{total_pairs} successful")
    
    return results


def multi_objective_runs(
    model_path: str,
    blueprint_path: str,
    objective_sets: List[Dict[str, Any]],
    base_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Run forge() with different objective configurations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        objective_sets: List of objective dictionaries
        base_constraints: Common constraints for all runs
        
    Returns:
        List of forge() results for each objective set
        
    Example:
        results = multi_objective_runs(
            "model.onnx", 
            "blueprint.yaml",
            [
                {'throughput': {'direction': 'maximize'}},
                {'power': {'direction': 'minimize'}},
                {'latency': {'direction': 'minimize'}}
            ]
        )
    """
    from ..core.api import forge
    
    results = []
    base_constraints = base_constraints or {}
    
    for i, objectives in enumerate(objective_sets):
        try:
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=objectives,
                constraints=base_constraints
            )
            
            # Add multi-objective metadata
            result['multi_objective_info'] = {
                'objective_set': objectives,
                'run_index': i,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Multi-objective run {i} failed: {e}")
            results.append({
                'multi_objective_info': {
                    'objective_set': objectives,
                    'run_index': i,
                    'success': False,
                    'error': str(e)
                }
            })
    
    successful_runs = sum(1 for r in results if r.get('multi_objective_info', {}).get('success', False))
    logger.info(f"Multi-objective runs completed: {successful_runs}/{len(objective_sets)} successful")
    
    return results


def configuration_sweep(
    model_path: str,
    blueprint_configs: List[str],
    common_objectives: Optional[Dict[str, Any]] = None,
    common_constraints: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    Sweep across different blueprint configurations.
    
    Args:
        model_path: Path to ONNX model
        blueprint_configs: List of blueprint configuration paths
        common_objectives: Common objectives for all runs
        common_constraints: Common constraints for all runs
        
    Returns:
        List of forge() results for each configuration
        
    Example:
        results = configuration_sweep(
            "model.onnx",
            ["config1.yaml", "config2.yaml", "config3.yaml"]
        )
    """
    from ..core.api import forge
    
    results = []
    common_objectives = common_objectives or {}
    common_constraints = common_constraints or {}
    
    for i, blueprint_path in enumerate(blueprint_configs):
        try:
            result = forge(
                model_path=model_path,
                blueprint_path=blueprint_path,
                objectives=common_objectives,
                constraints=common_constraints
            )
            
            # Add configuration sweep metadata
            result['config_sweep_info'] = {
                'blueprint_path': blueprint_path,
                'config_index': i,
                'success': True
            }
            
            results.append(result)
            
        except Exception as e:
            logger.error(f"Configuration sweep for {blueprint_path} failed: {e}")
            results.append({
                'config_sweep_info': {
                    'blueprint_path': blueprint_path,
                    'config_index': i,
                    'success': False,
                    'error': str(e)
                }
            })
    
    successful_runs = sum(1 for r in results if r.get('config_sweep_info', {}).get('success', False))
    logger.info(f"Configuration sweep completed: {successful_runs}/{len(blueprint_configs)} successful")
    
    return results