############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Template resolution for shape tiling specifications.

Template Syntax:
  ":" -> Copy dimension from reference shape
  1   -> Singleton dimension (only literal allowed)
  str -> Parameter name to look up
  DerivedDim -> Copy dimension from another interface
  ScaledDim -> Scale dimension from another interface
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .types import DerivedDim, ScaledDim

logger = logging.getLogger(__name__)


def resolve_template(
    template: List[Union[int, str, DerivedDim, ScaledDim]],
    reference_shape: Tuple[int, ...],
    param_getter: Callable[[str], Any],
    context: str = "",
    interfaces: Optional[Dict[str, Any]] = None
) -> Tuple[int, ...]:
    """Resolve template dimensions to concrete shape.

    Args:
        template: Template specification (e.g., [":", "PE", 1, DerivedDim("Q", 1)])
        reference_shape: Reference shape to resolve against
        param_getter: Function to resolve parameter names (e.g., get_nodeattr)
        context: Context string for error messages (e.g., "input.block")
        interfaces: Dict of interface models for DerivedDim/ScaledDim resolution (optional)

    Returns:
        Resolved concrete shape

    Raises:
        ValueError: If template is invalid or parameters not found

    Examples:
        >>> resolve_template([":", "PE"], (128, 768), lambda k: {"PE": 64}[k])
        (128, 64)
        >>> resolve_template([1, ":"], (128, 768), lambda k: {})
        (1, 768)
        >>> resolve_template([DerivedDim("input", 1)], (768,), lambda k: {},
        ...                  interfaces={"input": input_model})
        (768,)  # Copies input[1]
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
        if isinstance(dim, str):
            if dim == ":":
                # Copy from reference
                value = ref
            else:
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
        elif isinstance(dim, DerivedDim):
            # Copy dimension from another interface
            if interfaces is None:
                raise ValueError(
                    f"{context}[{i}]: DerivedDim requires interfaces dict"
                )
            if dim.source_interface not in interfaces:
                raise ValueError(
                    f"{context}[{i}]: source interface '{dim.source_interface}' not found"
                )

            source_model = interfaces[dim.source_interface]
            try:
                # Get the appropriate shape hierarchy
                source_shape = source_model.get_shape(dim.hierarchy)
                # Support negative indexing (Python convention)
                source_dim = dim.source_dim
                if source_dim < 0:
                    source_dim = len(source_shape) + source_dim
                value = source_shape[source_dim]
            except (IndexError, AttributeError) as e:
                raise ValueError(
                    f"{context}[{i}]: cannot access dimension {dim.source_dim} "
                    f"from interface '{dim.source_interface}' at {dim.hierarchy.value} level: {e}"
                )

            if value is None:
                raise ValueError(
                    f"{context}[{i}]: source dimension '{dim.source_interface}'.{dim.hierarchy.value}[{dim.source_dim}] "
                    f"is not yet resolved"
                )
        elif isinstance(dim, ScaledDim):
            # Scale dimension from another interface
            if interfaces is None:
                raise ValueError(
                    f"{context}[{i}]: ScaledDim requires interfaces dict"
                )
            if dim.source_interface not in interfaces:
                raise ValueError(
                    f"{context}[{i}]: source interface '{dim.source_interface}' not found"
                )

            source_model = interfaces[dim.source_interface]
            try:
                source_shape = source_model.get_shape(dim.hierarchy)
                # Support negative indexing (Python convention)
                source_dim = dim.source_dim
                if source_dim < 0:
                    source_dim = len(source_shape) + source_dim
                source_value = source_shape[source_dim]
            except (IndexError, AttributeError) as e:
                raise ValueError(
                    f"{context}[{i}]: cannot access dimension {dim.source_dim} "
                    f"from interface '{dim.source_interface}' at {dim.hierarchy.value} level: {e}"
                )

            if source_value is None:
                raise ValueError(
                    f"{context}[{i}]: source dimension '{dim.source_interface}'.{dim.hierarchy.value}[{dim.source_dim}] "
                    f"is not yet resolved"
                )

            # Validate scaling produces exact integer
            if dim.scale_factor <= 0:
                raise ValueError(
                    f"{context}[{i}]: scale_factor must be positive, got {dim.scale_factor}"
                )

            # For scale_down, check divisibility upfront
            if dim.scale_factor < 1.0:
                divisor = 1.0 / dim.scale_factor
                # Divisor must be an integer (or very close)
                int_divisor = round(divisor)
                if not math.isclose(divisor, int_divisor, abs_tol=1e-9):
                    raise ValueError(
                        f"{context}[{i}]: scale_factor {dim.scale_factor} is not 1/n for integer n "
                        f"(divisor = {divisor})"
                    )
                # Source must divide evenly
                if source_value % int_divisor != 0:
                    raise ValueError(
                        f"{context}[{i}]: source dimension {source_value} not evenly divisible by {int_divisor} "
                        f"(scale_factor = {dim.scale_factor} = 1/{int_divisor})"
                    )
                value = source_value // int_divisor
            else:
                # For scale_up, compute and verify exactness
                raw_value = source_value * dim.scale_factor
                value = round(raw_value)
                if not math.isclose(raw_value, value, abs_tol=1e-9):
                    raise ValueError(
                        f"{context}[{i}]: scaled dimension {source_value} * {dim.scale_factor} "
                        f"= {raw_value} is not an exact integer"
                    )

            if value <= 0:
                raise ValueError(
                    f"{context}[{i}]: scaled dimension must be positive, got {value}"
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
