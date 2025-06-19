"""
Batch Processing Automation

Simple utilities for processing multiple models or configurations in batch
by running forge() multiple times with different inputs.
"""

from typing import List, Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

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
        common_config: Common configuration for all runs (objectives/constraints)
        max_workers: Number of parallel workers
        progress_callback: Optional progress callback
        
    Returns:
        List of forge() results with batch metadata
        
    Example:
        results = batch_process([
            ("model1.onnx", "blueprint1.yaml"),
            ("model2.onnx", "blueprint2.yaml"),
            ("model3.onnx", "blueprint3.yaml")
        ])
    """
    from ...core.api import forge
    
    total_pairs = len(model_blueprint_pairs)
    logger.info(f"Starting batch processing: {total_pairs} model/blueprint pairs")
    
    common_config = common_config or {}
    
    def process_pair(pair: Tuple[str, str], index: int) -> Dict[str, Any]:
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
                'index': index,
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
                    'index': index,
                    'success': False,
                    'error': str(e)
                }
            }
    
    # Execute with optional parallelization
    results = []
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(process_pair, pair, i): i
                for i, pair in enumerate(model_blueprint_pairs)
            }
            
            for future in as_completed(futures):
                results.append(future.result())
    else:
        for i, pair in enumerate(model_blueprint_pairs):
            results.append(process_pair(pair, i))
    
    # Sort by index
    results.sort(key=lambda x: x.get('batch_info', {}).get('index', 0))
    
    successful = sum(1 for r in results if r.get('batch_info', {}).get('success', False))
    logger.info(f"Batch processing completed: {successful}/{total_pairs} successful")
    
    return results