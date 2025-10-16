############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Template resolution for shape tiling specifications.

Template Syntax:
  FULL_DIM -> Copy dimension from reference shape (full dimension)
  1        -> Singleton dimension (only literal allowed)
  str      -> Parameter name to look up
  DimensionSource subclasses -> Derive dimension from cross-interface computation
    - DerivedDim -> Copy dimension from another interface
    - ScaledDim -> Scale dimension from another interface
    - SumDims -> Sum dimensions from multiple interfaces
    - MaxDim -> Maximum dimension across interfaces
    - ComputedDim -> Custom computation
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .dimension_sources import DimensionSource
from .types import DerivedDim, ScaledDim, FULL_DIM  # Backward compatibility

logger = logging.getLogger(__name__)


def resolve_template(
    template: List[Union[int, str, DimensionSource]],
    reference_shape: Tuple[int, ...],
    param_getter: Callable[[str], Any],
    context: str = "",
    interfaces: Optional[Dict[str, Any]] = None
) -> Tuple[int, ...]:
    """Resolve template dimensions to concrete shape.

    Args:
        template: Template specification (e.g., [FULL_DIM, "PE", 1, DerivedDim("Q", 1)])
        reference_shape: Reference shape to resolve against
        param_getter: Function to resolve parameter names (e.g., get_nodeattr)
        context: Context string for error messages (e.g., "input.block")
        interfaces: Dict of interface models for DimensionSource resolution (optional)

    Returns:
        Resolved concrete shape

    Raises:
        ValueError: If template is invalid or parameters not found

    Examples:
        >>> resolve_template([FULL_DIM, "PE"], (128, 768), lambda k: {"PE": 64}[k])
        (128, 64)
        >>> resolve_template([1, FULL_DIM], (128, 768), lambda k: {})
        (1, 768)
        >>> resolve_template([DerivedDim("input", 1)], (768,), lambda k: {},
        ...                  interfaces={"input": input_model})
        (768,)  # Copies input[1]
        >>> resolve_template([SumDims((("in0", -1), ("in1", -1)))], (128,), lambda k: {},
        ...                  interfaces={"in0": model0, "in1": model1})
        (sum_of_last_dims,)  # Sums dimensions for concat
    """
    logger.debug(f"{context}: Resolving template {template} against reference {reference_shape}")

    # Pad template to match reference rank (prepend singletons)
    if len(template) < len(reference_shape):
        padding = len(reference_shape) - len(template)
        template = [1] * padding + template
        logger.debug(f"{context}: Auto-padded template with {padding} singletons → {template}")
    elif len(template) > len(reference_shape):
        raise ValueError(
            f"{context}: template length {len(template)} exceeds "
            f"reference rank {len(reference_shape)}"
        )

    resolved = []
    for i, (dim, ref) in enumerate(zip(template, reference_shape)):
        if isinstance(dim, type(FULL_DIM)) or dim is FULL_DIM:
            # Copy from reference (full dimension)
            value = ref
        elif isinstance(dim, str):
            # Parameter lookup
            try:
                value = param_getter(dim)
            except (AttributeError, KeyError):
                raise ValueError(
                    f"{context}[{i}]: parameter '{dim}' not found"
                )

            # Validate divisibility
            if ref % value != 0:
                raise ValueError(
                    f"{context}[{i}]: parameter '{dim}' value {value} "
                    f"does not divide parent dimension {ref}"
                )
        elif isinstance(dim, int):
            if dim == 1:
                value = 1
            else:
                raise ValueError(
                    f"{context}[{i}]: only singleton (1) allowed for literals, "
                    f"got {dim}. Use parameters for other values."
                )
        elif isinstance(dim, DimensionSource):
            # Use extensible DimensionSource resolution system
            if interfaces is None:
                raise ValueError(
                    f"{context}[{i}]: {type(dim).__name__} requires interfaces dict"
                )

            try:
                value = dim.resolve(interfaces, param_getter)
            except ValueError as e:
                raise ValueError(
                    f"{context}[{i}]: {type(dim).__name__} resolution failed: {e}"
                )
        else:
            raise ValueError(
                f"{context}[{i}]: invalid template element '{dim}' (type {type(dim).__name__})"
            )

        resolved.append(value)

    result = tuple(resolved)
    logger.debug(f"{context}: Resolved to {result}")
    return result


__all__ = ['resolve_template']
