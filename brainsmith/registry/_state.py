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
# Unified index for all components from all sources (core, project, custom)
_component_index: Dict[str, ComponentMetadata] = {}

# Discovery state flag - tracks whether component discovery has completed
_components_discovered = False
