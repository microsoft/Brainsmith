# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared runtime state for component registry.

This module contains mutable global state shared across manifest, discovery,
and lookup modules. Extracted to avoid circular dependencies.

Design Note:
    Separating state from logic enables clean module boundaries while
    maintaining a single source of truth for component metadata.
"""

from typing import Dict
from ._metadata import ComponentMetadata

# Global component index - single source of truth for component metadata
# Maps "source:name" -> ComponentMetadata
# Unified index for all components from all sources (core, user, plugins)
_component_index: Dict[str, ComponentMetadata] = {}

# Discovery state flag - tracks whether component discovery has completed
_components_discovered = False

# Backend inverted index - maps kernel name to list of backend names (lazy-built)
# Example: {'brainsmith:LayerNorm': ['brainsmith:LayerNorm_hls', 'user:LayerNorm_rtl']}
# Built on first call to list_backends_for_kernel() for O(k) instead of O(n) lookup
_backends_by_kernel: Dict[str, list[str]] = {}
_backend_index_built = False


def _build_backend_index() -> None:
    """Build inverted index of backends by target kernel (lazy, called once).

    Scans _component_index and builds _backends_by_kernel for O(k) lookup
    instead of O(n) scanning. Called automatically on first use.

    Side effects:
        - Populates _backends_by_kernel
        - Sets _backend_index_built = True
    """
    global _backend_index_built

    if _backend_index_built:
        return

    import logging
    logger = logging.getLogger(__name__)

    _backends_by_kernel.clear()

    for full_name, meta in _component_index.items():
        if meta.component_type != 'backend':
            continue

        target = meta.backend_target
        if not target:
            logger.warning(
                f"Backend {full_name} missing target_kernel metadata. "
                "This is a discovery bug - backends should have target_kernel set."
            )
            continue

        if target not in _backends_by_kernel:
            _backends_by_kernel[target] = []
        _backends_by_kernel[target].append(full_name)

    _backend_index_built = True
    logger.debug(f"Built backend index: {len(_backends_by_kernel)} kernels indexed")


def _invalidate_backend_index() -> None:
    """Invalidate backend index (force rebuild on next lookup).

    Called when backends are registered dynamically after discovery.
    """
    global _backend_index_built
    _backend_index_built = False
    _backends_by_kernel.clear()
