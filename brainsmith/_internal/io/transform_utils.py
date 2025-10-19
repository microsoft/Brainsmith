# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Transform utilities for Brainsmith operations.

NOTE: This is now a thin wrapper around brainsmith.transforms.
The registry-based approach has been replaced with direct imports.
"""
from typing import List, Optional, Any
import logging

# Import the new transform functions
from brainsmith.transforms import (
    apply_transforms as _apply_transforms,
    apply_transforms_with_params as _apply_transforms_with_params
)

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
    # Direct delegation to transforms module
    return _apply_transforms(model, transform_names, debug_path)


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
    # Direct delegation to transforms module
    return _apply_transforms_with_params(model, transforms)