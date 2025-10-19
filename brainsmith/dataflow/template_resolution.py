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
from .types import FULL_DIM

logger = logging.getLogger(__name__)


def normalize_template(
    template: List[Union[int, str, DimensionSource]],
    reference_shape: Tuple[int, ...]
) -> List[Union[int, str, DimensionSource]]:
    """Normalize template structure to match reference rank (no value resolution).

    Pads template with singleton (1) dimensions on the left to match the length
    of the reference shape. This is a pure structural transformation that does
    not resolve parameter values or DimensionSource objects.

    This function separates structural normalization from value resolution,
    enabling storage of normalized templates before parameter values are known.

    Args:
        template: Template specification (e.g., ["SIMD"], [1, "PE"], [DerivedDim(...)])
        reference_shape: Reference shape to match rank against

    Returns:
        Normalized template (same element types, left-padded with 1s)

    Raises:
        ValueError: If template length exceeds reference rank

    Examples:
        >>> normalize_template(["SIMD"], (1, 1, 64))
        [1, 1, "SIMD"]
        >>> normalize_template([1, "PE"], (128, 768))
        [1, "PE"]
        >>> normalize_template([FULL_DIM, "PE", 1], (128, 768, 64))
        [FULL_DIM, "PE", 1]
    """
    # Convert to list to allow modification
    template = list(template)

    # Pad template to match reference rank (prepend singletons)
    if len(template) < len(reference_shape):
        padding = len(reference_shape) - len(template)
        template = [1] * padding + template
        logger.debug(f"Auto-padded template with {padding} singletons → {template}")
    elif len(template) > len(reference_shape):
        raise ValueError(
            f"template length {len(template)} exceeds reference rank {len(reference_shape)}"
        )

    return template


def resolve_template(
    template: List[Union[int, str, DimensionSource]],
    reference_shape: Tuple[int, ...],
    param_getter: Callable[[str], Any],
    interfaces: Optional[Dict[str, Any]] = None
) -> Tuple[int, ...]:
    """Resolve template dimensions to concrete shape.

    Args:
        template: Template specification (e.g., [FULL_DIM, "PE", 1, DerivedDim("Q", 1)])
        reference_shape: Reference shape to resolve against
        param_getter: Function to resolve parameter names (e.g., get_nodeattr)
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
    logger.debug(f"Resolving template {template} against reference {reference_shape}")

    # Step 1: Normalize structure (pad to match reference rank)
    template = normalize_template(template, reference_shape)

    # Step 2: Resolve values
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
                raise ValueError(f"parameter '{dim}' not found")

            # Validate divisibility
            if ref % value != 0:
                raise ValueError(
                    f"parameter '{dim}' value {value} does not divide parent dimension {ref}"
                )
        elif isinstance(dim, int):
            if dim == 1:
                value = 1
            else:
                raise ValueError(
                    f"only singleton (1) allowed for literals, got {dim}. Use parameters for other values."
                )
        elif isinstance(dim, DimensionSource):
            # Use extensible DimensionSource resolution system
            if interfaces is None:
                raise ValueError(f"{type(dim).__name__} requires interfaces dict")

            try:
                value = dim.resolve(interfaces, param_getter)
            except ValueError as e:
                raise ValueError(f"{type(dim).__name__} resolution failed: {e}") from e
        else:
            raise ValueError(f"invalid template element '{dim}' (type {type(dim).__name__})")

        resolved.append(value)

    result = tuple(resolved)
    logger.debug(f"Resolved to {result}")
    return result


__all__ = ['resolve_template']
