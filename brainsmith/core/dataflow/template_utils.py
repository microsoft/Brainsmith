############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Template resolution utilities for tiling specifications.

This module provides utilities for working with tiling templates, including
parameter extraction, validation, and resolution.
"""

from typing import List, Union, Set, Dict, Any, Tuple, Optional


def extract_tiling_parameters(template: List[Union[int, str]]) -> Set[str]:
    """Extract parameter names from a tiling template.
    
    Args:
        template: Tiling template with literals, ":", and parameter names
        
    Returns:
        Set of parameter names (excluding ":")
        
    Example:
        >>> extract_tiling_parameters([1, "PE", ":", "SIMD"])
        {"PE", "SIMD"}
    """
    return {item for item in template if isinstance(item, str) and item != ":"}


def resolve_template_params(
    template: List[Union[int, str]],
    param_getter: callable,
    nodeattr_types: Optional[Dict[str, tuple]] = None
) -> List[Union[int, str]]:
    """Resolve nodeattr references in template to their values.
    
    Args:
        template: Template with nodeattr references
        param_getter: Function to get parameter value by name
        nodeattr_types: Optional dict of nodeattr type definitions
        
    Returns:
        Resolved template with nodeattr values substituted
        
    Raises:
        ValueError: If a required parameter is not found
    """
    resolved = []
    for item in template:
        if isinstance(item, str) and item != ":":
            # This is a nodeattr reference
            try:
                value = param_getter(item)
            except (AttributeError, Exception):
                # Try to get from nodeattr_types default
                if nodeattr_types and item in nodeattr_types:
                    value = nodeattr_types[item][2]  # default value
                else:
                    value = None
                    
            if value is None:
                raise ValueError(f"Nodeattr '{item}' not found")
                
            # Handle FINN's list encoding
            if isinstance(value, list) and len(value) == 1:
                value = value[0]
                
            resolved.append(int(value))
        else:
            # Keep literals and ":" as-is
            resolved.append(item)
            
    return resolved


def validate_tiling_template(
    template: List[Union[int, str]],
    shape_len: int
) -> List[str]:
    """Validate a tiling template for consistency.
    
    Args:
        template: Tiling template to validate
        shape_len: Expected length of shape it will be applied to
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    if not template:
        errors.append("Tiling template cannot be empty")
        return errors
    
    if len(template) > shape_len:
        errors.append(
            f"Template has {len(template)} dimensions "
            f"but shape has only {shape_len}"
        )
    
    for i, item in enumerate(template):
        if not isinstance(item, (int, str)):
            errors.append(
                f"Invalid tiling item {item!r} at position {i} - "
                f"must be int or str"
            )
        elif isinstance(item, int) and item <= 0:
            errors.append(
                f"Invalid tiling value {item} at position {i} - "
                f"must be positive"
            )
        elif isinstance(item, str) and item != ":" and not item.isidentifier():
            errors.append(
                f"Invalid parameter name '{item}' at position {i}"
            )
    
    return errors


def apply_tiling_to_shape(
    resolved_template: List[Union[int, str]],
    shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Apply a resolved tiling template to a concrete shape.
    
    Args:
        resolved_template: Template with all parameters resolved to values
        shape: Shape to tile
        
    Returns:
        Tuple of tiling dimensions
        
    Raises:
        ValueError: If template cannot be applied to shape
    """
    if len(resolved_template) > len(shape):
        raise ValueError(
            f"Template has {len(resolved_template)} dimensions "
            f"but shape has only {len(shape)}"
        )
    
    # Left-pad result with 1s if shape has more dims
    padding = len(shape) - len(resolved_template)
    result = [1] * padding
    
    # Process each template item aligned to the right of shape
    for i, (item, dim_size) in enumerate(zip(resolved_template, shape[padding:])):
        actual_idx = padding + i
        
        if item == ":":
            # Full dimension
            result.append(dim_size)
            
        elif isinstance(item, int):
            if item <= 0:
                raise ValueError(
                    f"Dimension {actual_idx}: Value must be positive, got {item}"
                )
            elif dim_size % item != 0:
                raise ValueError(
                    f"Dimension {actual_idx}: {item} does not evenly divide {dim_size}"
                )
            else:
                result.append(item)
        else:
            # This should never happen after parameter resolution
            raise ValueError(
                f"Invalid template item at {i}: {item} (type: {type(item).__name__})"
            )
    
    # Final validation
    final_result = tuple(result)
    for i, (tile, shape_dim) in enumerate(zip(final_result, shape)):
        if shape_dim % tile != 0:
            raise ValueError(
                f"Dimension {i}: Tiling value {tile} does not evenly divide "
                f"shape dimension {shape_dim}"
            )
    
    return final_result


def merge_tiling_templates(
    block_template: List[Union[int, str]],
    stream_template: List[Union[int, str]]
) -> Dict[str, Set[str]]:
    """Merge parameters from block and stream tiling templates.
    
    Args:
        block_template: Block tiling template
        stream_template: Stream tiling template
        
    Returns:
        Dict with 'block_only', 'stream_only', and 'shared' parameter sets
    """
    block_params = extract_tiling_parameters(block_template)
    stream_params = extract_tiling_parameters(stream_template)
    
    return {
        'block_only': block_params - stream_params,
        'stream_only': stream_params - block_params,
        'shared': block_params & stream_params
    }