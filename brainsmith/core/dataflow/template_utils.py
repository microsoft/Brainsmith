############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Shape resolution utilities for block and stream tiling.

This module provides focused utilities for resolving shape dimensions
in the context of hardware tiling specifications.
"""

from typing import List, Union, Tuple, Dict, Any


def resolve_shape_template(
    template: List[Union[int, str]],
    shape: Tuple[int, ...],
    nodeattrs: Dict[str, Any]
) -> Tuple[int, ...]:
    """
    Resolve a shape template against actual tensor shape and node attributes.

    This is the primary function for resolving block and stream tiling dimensions.
    It handles:
    - ":" for full dimension
    - Integer literals (only 1 allowed as singleton)
    - String parameter names resolved from nodeattrs

    Args:
        template: Shape template with ":", 1, or parameter names
        shape: Reference shape to resolve against
        nodeattrs: Node attributes containing parameter values

    Returns:
        Resolved shape tuple

    Raises:
        ValueError: If resolution fails or constraints are violated
    """
    if not template:
        raise ValueError("Shape template cannot be empty")

    if len(template) > len(shape):
        raise ValueError(
            f"Template has {len(template)} dimensions but shape has only {len(shape)}"
        )

    # Left-pad with 1s if shape has more dimensions
    padding = len(shape) - len(template)
    result = [1] * padding

    # Process each template element
    for i, (tmpl_item, shape_dim) in enumerate(zip(template, shape[padding:])):
        actual_idx = padding + i

        if tmpl_item == ":":
            # Full dimension
            result.append(shape_dim)

        elif tmpl_item == 1:
            # Singleton literal (only 1 allowed)
            result.append(1)

        elif isinstance(tmpl_item, int):
            # No other integer literals allowed
            raise ValueError(
                f"Dimension {actual_idx}: Only singleton literal (1) allowed, "
                f"got {tmpl_item}. Use parameters for other values."
            )

        elif isinstance(tmpl_item, str):
            # Parameter reference - must exist in nodeattrs
            if tmpl_item not in nodeattrs:
                raise ValueError(
                    f"Dimension {actual_idx}: Parameter '{tmpl_item}' not found in nodeattrs"
                )

            value = nodeattrs[tmpl_item]

            # Handle FINN's single-element list encoding
            if isinstance(value, list) and len(value) == 1:
                value = value[0]

            try:
                value = int(value)
            except (ValueError, TypeError):
                raise ValueError(
                    f"Dimension {actual_idx}: Parameter '{tmpl_item}' has non-integer "
                    f"value: {value}"
                )

            if value <= 0:
                raise ValueError(
                    f"Dimension {actual_idx}: Parameter '{tmpl_item}' must be positive, "
                    f"got {value}"
                )

            if shape_dim % value != 0:
                raise ValueError(
                    f"Dimension {actual_idx}: Parameter '{tmpl_item}'={value} does not "
                    f"evenly divide shape dimension {shape_dim}"
                )

            result.append(value)

        else:
            raise ValueError(
                f"Dimension {actual_idx}: Invalid template item {tmpl_item!r} "
                f"(type: {type(tmpl_item).__name__})"
            )

    return tuple(result)


def validate_stream_divides_block(
    stream_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...]
) -> None:
    """
    Validate that stream dimensions evenly divide block dimensions.

    Args:
        stream_shape: Stream tiling dimensions
        block_shape: Block tiling dimensions

    Raises:
        ValueError: If stream doesn't divide block evenly
    """
    if len(stream_shape) != len(block_shape):
        raise ValueError(
            f"Shape mismatch: stream has {len(stream_shape)} dims, "
            f"block has {len(block_shape)} dims"
        )

    for i, (stream_dim, block_dim) in enumerate(zip(stream_shape, block_shape)):
        if block_dim % stream_dim != 0:
            raise ValueError(
                f"Dimension {i}: Stream dim {stream_dim} does not evenly divide "
                f"block dim {block_dim}"
            )


def resolve_template_params(
    template: List[Union[int, str]],
    param_getter: callable,
    nodeattr_types: Dict[str, tuple] = None
) -> List[Union[int, str]]:
    """
    Backward compatibility function for resolving template parameters.

    This function is maintained for compatibility but should be replaced
    with resolve_shape_template in new code.

    Args:
        template: Template with parameter references
        param_getter: Function to get parameter values
        nodeattr_types: Optional nodeattr type definitions

    Returns:
        Resolved template
    """
    resolved = []
    for item in template:
        if isinstance(item, str) and item != ":":
            # Parameter reference
            try:
                value = param_getter(item)
            except (AttributeError, Exception):
                # Try default from nodeattr_types
                if nodeattr_types and item in nodeattr_types:
                    value = nodeattr_types[item][2]  # default value
                else:
                    value = None

            if value is None:
                raise ValueError(f"Parameter '{item}' not found")

            # Handle FINN's list encoding
            if isinstance(value, list) and len(value) == 1:
                value = value[0]

            resolved.append(int(value))
        else:
            # Keep literals and ":" as-is
            resolved.append(item)

    return resolved