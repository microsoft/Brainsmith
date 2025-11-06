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
  (interface, dim_idx)    -> Tuple shorthand (uses context hierarchy: BLOCK or STREAM)
  (interface, dim_idx, hierarchy) -> Tuple shorthand with explicit hierarchy override
  Callable[[Dict, Callable, Any, Optional[str]], int] -> Custom dimension computation function

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
    template: Union[List[Union[int, str, Tuple[str, int], Callable, type]], type],
    reference_shape: Tuple[int, ...]
) -> List[Union[int, str, Tuple[str, int], Callable, type]]:
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
        >>> normalize_template([("input", -1)], (1, 1, 64))
        [1, 1, ("input", -1)]
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


def resolve_template(
    template: Union[List[Union[int, str, Tuple[str, int], Callable, type]], type],
    reference_shape: Tuple[int, ...],
    param_getter: Callable[[str], Any],
    interfaces: Optional[Dict[str, Any]] = None,
    model: Optional[Any] = None,
    tensor_name: Optional[str] = None,
    hierarchy: ShapeHierarchy = ShapeHierarchy.STREAM
) -> Tuple[int, ...]:
    """Resolve template dimensions to concrete shape.

    Supports both list-based templates and FULL_SHAPE sentinel.
    DimSpec union type supports multiple resolution strategies:
    - int (1 only): Singleton dimension
    - str: Parameter lookup
    - (interface, dim_idx): Tuple shorthand (uses context hierarchy)
    - (interface, dim_idx, hierarchy): Tuple shorthand with explicit hierarchy override
    - Callable: Custom dimension function (unified 4-param signature)
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
        hierarchy: Shape hierarchy for 2-tuple resolution (default: STREAM).
                   3-tuple syntax can override: ("input", -1, BLOCK)

    Returns:
        Resolved concrete shape

    Raises:
        ValueError: If template is invalid or parameters not found

    Examples:
        >>> resolve_template([FULL_DIM, "PE"], (128, 768), lambda k: {"PE": 64}[k])
        (128, 64)
        >>> resolve_template([1, FULL_DIM], (128, 768), lambda k: {})
        (1, 768)
        >>> resolve_template([("input", -1)], (768,), lambda k: {},
        ...                  interfaces={"input": input_model})
        (768,)  # Copies input's last dim via tuple shorthand
        >>> resolve_template([derive_dim("input", BLOCK, 1)], (768,), lambda k: {},
        ...                  interfaces={"input": input_model})
        (768,)  # Copies input[1] via callable
        >>> resolve_template(FULL_SHAPE, (1, 224, 224, 64), lambda k: {})
        (1, 224, 224, 64)  # Expands to full reference shape
    """
    logger.debug(f"Resolving template {template} against reference {reference_shape}")

    # Step 1: Normalize structure (pad to match reference rank)
    template = normalize_template(template, reference_shape)

    # Step 2: Resolve values
    resolved = []
    for i, (dim, ref) in enumerate(zip(template, reference_shape)):
        if dim is FULL_DIM:
            # FULL_DIM: copy entire dimension from reference
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
            if dim != 1:
                raise ValueError(
                    f"Only singleton (1) allowed for literals, got {dim}. "
                    f"Use parameters for other values."
                )
            value = dim
        elif isinstance(dim, tuple):
            # Tuple shorthand: ("input", -1) or ("input", -1, STREAM)
            if interfaces is None:
                raise ValueError(f"Tuple shorthand {dim} requires interfaces dict")

            if len(dim) == 2:
                # 2-tuple: use context hierarchy (from parameter)
                interface_name, dim_idx = dim
                effective_hierarchy = hierarchy
            elif len(dim) == 3:
                # 3-tuple: explicit hierarchy override
                interface_name, dim_idx, explicit_hierarchy = dim
                effective_hierarchy = explicit_hierarchy
            else:
                raise ValueError(f"Tuple shorthand must be 2 or 3 elements, got {len(dim)}")

            try:
                resolver = derive_dim(interface_name, effective_hierarchy, dim_idx)
                value = resolver(interfaces, param_getter, model, tensor_name)
            except ValueError as e:
                raise ValueError(f"Tuple shorthand {dim} resolution failed: {e}") from e
        elif callable(dim):
            # Callable: custom dimension computation function (unified 4-param signature)
            if interfaces is None:
                raise ValueError(f"Callable dimension function requires interfaces dict")

            try:
                value = dim(interfaces, param_getter, model, tensor_name)
            except Exception as e:
                raise ValueError(f"Callable dimension resolution failed: {e}") from e

            if not isinstance(value, int) or value < 1:
                raise ValueError(
                    f"Callable dimension function returned invalid value {value} (must be positive int)"
                )
        else:
            raise ValueError(f"invalid template element '{dim}' (type {type(dim).__name__})")

        resolved.append(value)

    result = tuple(resolved)
    logger.debug(f"Resolved to {result}")
    return result


__all__ = ['resolve_template', 'normalize_template']
