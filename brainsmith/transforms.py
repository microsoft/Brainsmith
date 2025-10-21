# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Direct transform import - no registration required.

Transforms are imported directly from known framework locations using
naming conventions. No pre-registration or manifest needed.

This is a key simplification: transforms aren't "plugins" that need discovery,
they're just imports we can find by convention.
"""

import re
import importlib
import logging
from typing import Type, List

logger = logging.getLogger(__name__)

# Import cache
_transform_cache = {}


def import_transform(name: str) -> Type:
    """Import transform by name from known framework locations.

    Uses naming conventions to locate transforms in:
    - Brainsmith primitives (brainsmith.primitives.transforms.*)
    - FINN transforms (finn.transformation.*)
    - QONNX transforms (qonnx.transformation.*)

    Args:
        name: Transform class name (e.g., 'FoldConstants', 'Streamline')

    Returns:
        Transform class

    Raises:
        ImportError: If transform not found in any known location

    Examples:
        >>> FoldConstants = import_transform('FoldConstants')
        >>> model = model.transform(FoldConstants())

        >>> Streamline = import_transform('Streamline')
        >>> model = model.transform(Streamline())
    """
    if name in _transform_cache:
        return _transform_cache[name]

    logger.debug(f"Importing transform: {name}")

    # Try standard framework locations using conventions
    search_patterns = _get_search_patterns(name)

    for module_pattern in search_patterns:
        try:
            logger.debug(f"Trying: {module_pattern}.{name}")
            module = importlib.import_module(module_pattern)
            transform_cls = getattr(module, name)
            _transform_cache[name] = transform_cls
            logger.debug(f"Found: {module_pattern}.{name}")
            return transform_cls
        except (ImportError, AttributeError):
            continue

    # Not found - provide helpful error
    raise ImportError(
        f"Transform '{name}' not found. Searched:\n" +
        "\n".join(f"  - {p}.{name}" for p in search_patterns) +
        "\n\nIf this is a custom transform, ensure it's in one of the standard locations."
    )


def _get_search_patterns(name: str) -> List[str]:
    """Get module search patterns for a transform name.

    Uses conventions to generate likely module paths:
    - Brainsmith: flat structure under cleanup/kernel_opt/post_proc
    - QONNX: module name is snake_case of class name
    - FINN: module name is snake_case of class name, various subpackages

    Args:
        name: Transform class name

    Returns:
        List of module path patterns to try
    """
    snake_name = _to_snake_case(name)

    patterns = []

    # Brainsmith primitives - organized by category
    # Try all three categories since we don't know which one
    for category in ['cleanup', 'kernel_opt', 'post_proc']:
        patterns.append(f'brainsmith.primitives.transforms.{category}.{snake_name}')

    # QONNX - flat structure under qonnx.transformation
    patterns.append(f'qonnx.transformation.{snake_name}')

    # FINN - various subpackages under finn.transformation
    finn_patterns = [
        f'finn.transformation.{snake_name}',
        f'finn.transformation.streamline.{snake_name}',
        f'finn.transformation.streamline.absorb.{snake_name}',
        f'finn.transformation.streamline.reorder.{snake_name}',
        f'finn.transformation.streamline.collapse_repeated.{snake_name}',
        f'finn.transformation.streamline.round_thresholds.{snake_name}',
        f'finn.transformation.streamline.sign_to_thres.{snake_name}',
        f'finn.transformation.fpgadataflow.{snake_name}',
        f'finn.transformation.fpgadataflow.convert_to_hw_layers.{snake_name}',
        f'finn.transformation.qonnx.{snake_name}',
    ]
    patterns.extend(finn_patterns)

    return patterns


def _to_snake_case(name: str) -> str:
    """Convert CamelCase to snake_case for module names.

    Args:
        name: CamelCase string

    Returns:
        snake_case string

    Examples:
        >>> _to_snake_case('FoldConstants')
        'fold_constants'
        >>> _to_snake_case('ConvertQONNXtoFINN')
        'convert_qonnxto_finn'
    """
    # Insert underscore before uppercase letters (except at start)
    s1 = re.sub(r'(?<!^)(?=[A-Z])', '_', name)
    # Convert to lowercase
    return s1.lower()
