############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Template resolution for shape tiling specifications.

Template Syntax (DimSpec union type):
  FULL_DIM                -> Copy dimension from reference shape (full dimension)
  1                       -> Singleton dimension (only literal allowed)
  str                     -> Parameter name to look up
  Callable                -> Custom dimension computation (4-param signature)

TilingSpec Syntax:
  [DimSpec, ...]          -> List of dimension specifications (standard)
  FULL_SHAPE              -> Bare sentinel expands to [FULL_DIM, ...] matching rank
"""

import logging
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .types import FULL_DIM, FULL_SHAPE, ShapeHierarchy
from .spec_helpers import derive_dim

logger = logging.getLogger(__name__)


def normalize_template(
    template: Union[List[Union[int, str, type]], type],
    reference_shape: Tuple[int, ...]
) -> List[Union[int, str, type]]:
    """Normalize template structure to match reference rank (no value resolution).

    Handles both list-based templates and FULL_SHAPE sentinel. Pads templates
    with singleton (1) dimensions on the left to match reference shape length.
    This is a pure structural transformation that does not resolve parameter
    values, tuples, or callables.

    This function separates structural normalization from value resolution,
    enabling storage of normalized templates before parameter values are known.

    Args:
        template: TilingSpec - either list of DimSpecs or FULL_SHAPE sentinel
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
        >>> normalize_template(FULL_SHAPE, (1, 224, 224, 64))
        [FULL_DIM, FULL_DIM, FULL_DIM, FULL_DIM]
    """
    # Handle FULL_SHAPE sentinel - expand to rank-appropriate FULL_DIM list
    if template is FULL_SHAPE:
        result = [FULL_DIM] * len(reference_shape)
        logger.debug(f"Expanded FULL_SHAPE to {result} for rank {len(reference_shape)}")
        return result

    # Standard list-based template processing
    template = list(template)  # Allow modification (input may be tuple)

    # Handle rank 0 tensors (scalars) - no dimensions to tile
    if len(reference_shape) == 0:
        if len(template) > 0:
            raise ValueError(
                f"Cannot apply template with {len(template)} dimensions to rank 0 tensor (scalar). "
                f"Scalars have no dimensions to tile. Template: {template}"
            )
        return []

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


def _resolve_full_dim(ref_dim: int) -> int:
    """Copy full dimension from reference shape.

    Args:
        ref_dim: Reference dimension value

    Returns:
        The reference dimension value unchanged
    """
    return ref_dim


def _resolve_parameter(
    param_name: str,
    ref_dim: int,
    param_getter: Callable[[str], Any]
) -> int:
    """Resolve parameter name to value with divisibility validation.

    Args:
        param_name: Parameter name to look up
        ref_dim: Reference dimension that must be divisible by parameter value
        param_getter: Function to retrieve parameter values

    Returns:
        Parameter value

    Raises:
        ValueError: If parameter not found or doesn't divide reference dimension
    """
    try:
        value = param_getter(param_name)
    except (AttributeError, KeyError):
        raise ValueError(f"parameter '{param_name}' not found")

    if ref_dim % value != 0:
        raise ValueError(
            f"parameter '{param_name}' value {value} does not divide "
            f"parent dimension {ref_dim}"
        )
    return value


def _resolve_literal(value: int) -> int:
    """Validate and return singleton literal.

    Args:
        value: Literal integer value

    Returns:
        The value (must be 1)

    Raises:
        ValueError: If value is not 1
    """
    if value != 1:
        raise ValueError(
            f"Only singleton (1) allowed for literals, got {value}. "
            f"Use parameters for other values."
        )
    return value


def _resolve_callable(
    func: Callable,
    interfaces: Dict[str, Any],
    param_getter: Callable,
    model: Optional[Any],
    tensor_name: Optional[str]
) -> int:
    """Execute custom dimension computation function.

    Args:
        func: Callable dimension function
        interfaces: Dict of interface models for dimension derivation
        param_getter: Function to retrieve parameter values
        model: ModelWrapper for ONNX graph access (optional)
        tensor_name: Tensor name (optional)

    Returns:
        Computed dimension value

    Raises:
        ValueError: If interfaces not provided, computation fails, or result invalid
    """
    if interfaces is None:
        raise ValueError("Callable dimension function requires interfaces dict")

    try:
        value = func(interfaces, param_getter, model, tensor_name)
    except Exception as e:
        raise ValueError(f"Callable dimension resolution failed: {e}") from e

    if not isinstance(value, int) or value < 1:
        raise ValueError(
            f"Callable dimension function returned invalid value {value} "
            f"(must be positive int)"
        )
    return value


def resolve_template(
    template: Union[List[Union[int, str, Callable, type]], type],
    reference_shape: Tuple[int, ...],
    param_getter: Callable[[str], Any],
    interfaces: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    tensor_name: Optional[str] = None
) -> Tuple[int, ...]:
    """Resolve template dimensions to concrete shape.

    Supports both list-based templates and FULL_SHAPE sentinel.
    DimSpec union type supports multiple resolution strategies:
    - int (1 only): Singleton dimension
    - str: Parameter lookup
    - FULL_DIM: Copy from reference

    TilingSpec supports:
    - List[DimSpec]: Standard list of dimension specs
    - FULL_SHAPE: Bare sentinel expands to [FULL_DIM, ...] matching rank

    Args:
        template: TilingSpec (list or FULL_SHAPE sentinel)
        reference_shape: Reference shape to resolve against
        param_getter: Function to resolve parameter names (e.g., get_nodeattr)
        interfaces: Dict of interface models for dimension derivation (optional)
        model: ModelWrapper for ONNX graph access (optional, for custom callbacks)
        tensor_name: Tensor name (optional, for custom callbacks)

    Returns:
        Resolved concrete shape

    Raises:
        ValueError: If template is invalid or parameters not found

    Examples:
        >>> resolve_template([FULL_DIM, "PE"], (128, 768), lambda k: {"PE": 64}[k])
        (128, 64)
        >>> resolve_template([1, FULL_DIM], (128, 768), lambda k: {})
        (1, 768)
        >>> resolve_template(FULL_SHAPE, (1, 224, 224, 64), lambda k: {})
        (1, 224, 224, 64)  # Expands to full reference shape
    """
    logger.debug(f"Resolving template {template} against reference {reference_shape}")

    # Step 1: Normalize structure (pad to match reference rank)
    template = normalize_template(template, reference_shape)

    # Step 2: Resolve values using specialized resolvers
    resolved = []
    for i, (dim, ref) in enumerate(zip(template, reference_shape)):
        if dim is FULL_DIM:
            value = _resolve_full_dim(ref)
        elif isinstance(dim, str):
            value = _resolve_parameter(dim, ref, param_getter)
        elif isinstance(dim, int):
            value = _resolve_literal(dim)
        elif callable(dim):
            value = _resolve_callable(dim, interfaces, param_getter, model, tensor_name)
        else:
            raise ValueError(f"invalid template element '{dim}' (type {type(dim).__name__})")

        resolved.append(value)

    result = tuple(resolved)
    logger.debug(f"Resolved to {result}")
    return result


__all__ = ['resolve_template', 'normalize_template']
