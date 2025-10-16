# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Transform utilities for Brainsmith operations.
"""
from typing import List, Optional, Any
import logging

logger = logging.getLogger(__name__)


def apply_transforms(model: Any, transform_names: List[str], debug_path: Optional[str] = None) -> Any:
    """Apply a sequence of transforms to a model.
    
    This helper function retrieves and applies transforms in order, following
    the common pattern used throughout Brainsmith steps.
    
    Args:
        model: The model to transform (typically QONNX ModelWrapper)
        transform_names: List of transform names to apply in order
        debug_path: Optional path prefix for saving debug models between transforms
        
    Returns:
        The transformed model
        
    Example:
        model = apply_transforms(model, [
            'RemoveIdentityOps',
            'RemoveUnusedTensors', 
            'SortGraph'
        ])
    """
    # Import inside function to avoid circular imports
    from brainsmith.registry import get_transform
    
    for i, transform_name in enumerate(transform_names):
        logger.debug(f"Applying transform: {transform_name}")
        
        # Get and apply transform
        Transform = get_transform(transform_name)
        model = model.transform(Transform())
        
        # Save debug model if requested
        if debug_path:
            debug_file = f"{debug_path}_step{i:02d}_{transform_name}.onnx"
            from brainsmith.core.explorer.utils import save_debug_model
            save_debug_model(model, debug_file)
    
    return model


def apply_transforms_with_params(model: Any, transforms: List[tuple]) -> Any:
    """Apply transforms with parameters.
    
    Args:
        model: The model to transform
        transforms: List of (transform_name, kwargs) tuples
        
    Returns:
        The transformed model
        
    Example:
        model = apply_transforms_with_params(model, [
            ('FoldConstants', {}),
            ('ConvertQONNXToFINN', {'preserve_qnt_ops': True}),
            ('InferDataLayouts', {'topological': True})
        ])
    """
    # Import inside function to avoid circular imports
    from brainsmith.registry import get_transform
    
    for transform_name, kwargs in transforms:
        logger.debug(f"Applying transform: {transform_name} with {kwargs}")
        
        Transform = get_transform(transform_name)
        model = model.transform(Transform(**kwargs))
    
    return model