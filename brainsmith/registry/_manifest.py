# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Manifest caching for component registry.

Provides performance optimization through JSON manifest caching of component
metadata. Enables fast startup by avoiding expensive filesystem scans and
imports on subsequent runs.

Functions:
    - _build_manifest_from_index() - Serialize component index to dict
    - _save_manifest() - Write manifest to JSON file
    - _load_manifest() - Read manifest from JSON file
    - _is_manifest_stale() - Check if cached manifest needs refresh
    - _populate_index_from_manifest() - Rebuild index from cached manifest
"""

import os
import json
import importlib.util
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any

from ._state import _component_index
from ._metadata import ComponentMetadata, ImportSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Manifest Caching (Simplified - Direct JSON Operations)
# ============================================================================

def _build_manifest_from_index() -> Dict[str, Any]:
    """Build manifest dict from current component index with full metadata.

    Converts _component_index to simple dict for JSON caching. Includes file
    mtimes for cache invalidation and type-specific metadata for full restoration.

    Returns:
        Manifest dict with version, timestamp, and components
    """
    components = {}

    for full_name, meta in _component_index.items():
        # Resolve module to file path and get mtime for staleness detection
        file_path = None
        mtime = None
        try:
            spec = importlib.util.find_spec(meta.import_spec.module)
            if spec and spec.origin:
                file_path = spec.origin
                mtime = os.path.getmtime(file_path)
        except Exception as e:
            logger.debug(f"Could not get mtime for {meta.import_spec.module}: {e}")

        # Build component data with type-specific metadata
        components[full_name] = {
            'type': meta.component_type,
            'module': meta.import_spec.module,
            'attr': meta.import_spec.attr,
            'metadata': meta.import_spec.extra,
            'file_path': file_path,
            'mtime': mtime,
            # Type-specific metadata (preserved for restoration)
            'kernel_infer': meta.kernel_infer if meta.component_type == 'kernel' else None,
            'kernel_domain': meta.kernel_domain if meta.component_type == 'kernel' else None,
            'kernel_backends': meta.kernel_backends if meta.component_type == 'kernel' else None,
            'backend_target': meta.backend_target if meta.component_type == 'backend' else None,
            'backend_language': meta.backend_language if meta.component_type == 'backend' else None,
        }

    return {
        'version': '1.0',
        'generated_at': datetime.now().isoformat(),
        'components': components
    }


def _save_manifest(manifest: Dict[str, Any], path: Path) -> None:
    """Save manifest dict to JSON file.

    Args:
        manifest: Manifest dict from _build_manifest_from_index()
        path: Path to save manifest (typically .brainsmith/component_manifest.json)
    """
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        json.dump(manifest, f, indent=2)

    logger.debug(f"Saved manifest with {len(manifest['components'])} components to {path}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate manifest from JSON file.

    Args:
        path: Path to manifest file

    Returns:
        Manifest dict with version, timestamp, and components

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest has invalid version or structure
        json.JSONDecodeError: If manifest is corrupted JSON
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Simple version validation
    version = data.get('version')
    if version != '1.0':
        raise ValueError(
            f"Unknown manifest version: {version}. "
            f"Expected '1.0'. Cache will be regenerated."
        )

    # Basic structure validation
    if 'components' not in data:
        raise ValueError("Manifest missing 'components' field. Cache will be regenerated.")

    logger.debug(f"Loaded manifest with {len(data['components'])} components from {path}")
    return data


def _is_manifest_stale(manifest: Dict[str, Any]) -> bool:
    """Check if manifest is stale by comparing file mtimes.

    A manifest is stale if any component file has been modified since the
    manifest was generated. This enables automatic cache invalidation when
    code changes.

    Note: This only detects file modifications, not new files. Users must
    manually refresh (--refresh) when adding new component files.

    Args:
        manifest: Manifest dict from _load_manifest()

    Returns:
        True if manifest is stale and should be regenerated
    """
    for full_name, comp_data in manifest['components'].items():
        file_path = comp_data.get('file_path')
        cached_mtime = comp_data.get('mtime')

        # Skip if we don't have mtime data
        if file_path is None or cached_mtime is None:
            continue

        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > cached_mtime:
                logger.info(
                    f"Cache stale: {full_name} modified "
                    f"(cached: {cached_mtime}, current: {current_mtime})"
                )
                return True
        except OSError as e:
            # File doesn't exist or can't be accessed - treat as stale
            logger.info(f"Cache stale: cannot access {file_path}: {e}")
            return True

    return False


def _populate_index_from_manifest(manifest: Dict[str, Any]) -> None:
    """Populate component index from cached manifest with full metadata.

    Rebuilds _component_index from manifest without importing components.
    Components will be lazy-loaded on first use. Type-specific metadata is
    restored from manifest.

    Args:
        manifest: Manifest dict from _load_manifest()
    """
    for full_name, component in manifest['components'].items():
        # Parse source and name from key (source:name)
        source, name = full_name.split(':', 1)

        # Build ComponentMetadata with type-specific fields
        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=component['type'],
            import_spec=ImportSpec(
                module=component['module'],
                attr=component['attr'],
                extra=component.get('metadata', {})
            ),
            # Restore type-specific metadata from manifest
            kernel_infer=component.get('kernel_infer'),
            kernel_domain=component.get('kernel_domain'),
            kernel_backends=component.get('kernel_backends'),
            backend_target=component.get('backend_target'),
            backend_language=component.get('backend_language'),
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed from manifest: {full_name}")
