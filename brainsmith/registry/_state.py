# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared runtime state for component registry.

This module contains mutable global state shared across manifest, discovery,
and lookup modules. Extracted to avoid circular dependencies.

Design Note:
    Separating state from logic enables clean module boundaries while
    maintaining a single source of truth for component metadata.
"""


from ._metadata import ComponentMetadata

# Global component index - single source of truth for component metadata
# Maps "source:name" -> ComponentMetadata
# Unified index for all components from all sources (core, project, custom)
_component_index: dict[str, ComponentMetadata] = {}

# Discovered sources - tracks all active component sources
# Populated during discovery from:
# - Entrypoint names (e.g., 'finn' from brainsmith.plugins entrypoints)
# - component_sources keys (e.g., 'brainsmith', 'project', custom sources)
# Used for source detection in _decorators.py and domain matching in _domain_utils.py
_discovered_sources: set[str] = set()

# Discovery state flag - tracks whether component discovery has completed
_components_discovered = False
