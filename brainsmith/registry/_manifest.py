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
from ._metadata import ComponentMetadata, ComponentType, ImportSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Manifest Caching (Simplified - Direct JSON Operations)
# ============================================================================

def _build_manifest_from_index() -> Dict[str, Any]:
    """Build type-stratified manifest from current component index.

    Creates manifest with components separated by type (kernels/backends/steps).
    Uses single generated_at timestamp for staleness detection instead of
    per-component mtimes. Eliminates null fields by storing only relevant
    metadata for each component type.

    Returns:
        Manifest dict with version, timestamp, and type-stratified components
    """
    kernels, backends, steps = {}, {}, {}

    for full_name, meta in _component_index.items():
        # Resolve module to file path (needed for staleness detection)
        file_path = None
        try:
            spec = importlib.util.find_spec(meta.import_spec.module)
            if spec and spec.origin:
                file_path = spec.origin
        except Exception as e:
            logger.debug(f"Could not resolve file path for {meta.import_spec.module}: {e}")

        # Build base component data (common fields, no mtime)
        base = {
            'module': meta.import_spec.module,
            'attr': meta.import_spec.attr,
            'file_path': file_path,
        }

        # Add to appropriate section with type-specific fields (no nulls)
        if meta.component_type == ComponentType.KERNEL:
            kernels[full_name] = {
                **base,
                'infer': meta.kernel_infer,
                'domain': meta.kernel_domain,
                'backends': meta.kernel_backends or []
            }
        elif meta.component_type == ComponentType.BACKEND:
            backends[full_name] = {
                **base,
                'target': meta.backend_target,
                'language': meta.backend_language
            }
        else:  # ComponentType.STEP
            steps[full_name] = base

    return {
        'version': '2.0',
        'generated_at': datetime.now().isoformat(),
        'kernels': kernels,
        'backends': backends,
        'steps': steps
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

    # Calculate total components for logging
    total = len(manifest.get('kernels', {})) + len(manifest.get('backends', {})) + len(manifest.get('steps', {}))
    logger.debug(f"Saved manifest with {total} components to {path}")


def _load_manifest(path: Path) -> Dict[str, Any]:
    """Load and validate manifest from JSON file.

    Args:
        path: Path to manifest file

    Returns:
        Manifest dict with version, timestamp, and type-stratified components

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest has invalid version or structure
        json.JSONDecodeError: If manifest is corrupted JSON
    """
    with open(path, 'r') as f:
        data = json.load(f)

    # Version validation
    version = data.get('version')
    if version == '1.0':
        # Old flat format - trigger regeneration
        logger.info(
            f"Manifest version 1.0 (old format) found at {path}. "
            f"Regenerating with type-stratified format (v2.0)."
        )
        raise ValueError("Old manifest version - will regenerate")
    elif version != '2.0':
        raise ValueError(
            f"Unknown manifest version: {version}. "
            f"Expected '2.0'. Cache will be regenerated."
        )

    # Structure validation for v2.0
    required_fields = ['generated_at', 'kernels', 'backends', 'steps']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Manifest missing '{field}' field. Cache will be regenerated.")

    # Calculate total components for logging
    total = len(data['kernels']) + len(data['backends']) + len(data['steps'])
    logger.debug(
        f"Loaded manifest from {path}: "
        f"{len(data['kernels'])} kernels, "
        f"{len(data['backends'])} backends, "
        f"{len(data['steps'])} steps "
        f"(total: {total})"
    )
    return data


def _is_manifest_stale(manifest: Dict[str, Any]) -> bool:
    """Check if manifest is stale by comparing file mtimes to manifest timestamp.

    A manifest is stale if any component file has been modified after the
    manifest was generated. Uses a single manifest timestamp instead of
    per-component mtimes for efficient staleness detection.

    Note: This only detects file modifications, not new files. Users must
    manually refresh (--refresh) when adding new component files.

    Args:
        manifest: Manifest dict from _load_manifest()

    Returns:
        True if manifest is stale and should be regenerated
    """
    # Parse manifest generation timestamp
    try:
        generated_at_str = manifest.get('generated_at')
        if not generated_at_str:
            logger.info("Cache stale: missing generated_at timestamp")
            return True

        # Parse ISO format timestamp to Unix timestamp for comparison
        generated_at = datetime.fromisoformat(generated_at_str).timestamp()
    except (ValueError, TypeError) as e:
        logger.info(f"Cache stale: invalid generated_at timestamp: {e}")
        return True

    # Check all component types
    for section in ['kernels', 'backends', 'steps']:
        for full_name, comp_data in manifest.get(section, {}).items():
            file_path = comp_data.get('file_path')

            # Skip if we don't have file path
            if file_path is None:
                continue

            try:
                current_mtime = os.path.getmtime(file_path)

                # Compare file mtime against manifest generation time
                if current_mtime > generated_at:
                    logger.info(
                        f"Cache stale: {full_name} modified after manifest generation "
                        f"(file: {current_mtime}, manifest: {generated_at})"
                    )
                    return True
            except OSError as e:
                # File doesn't exist or can't be accessed - treat as stale
                logger.info(f"Cache stale: cannot access {file_path}: {e}")
                return True

    return False


def _populate_index_from_manifest(manifest: Dict[str, Any]) -> None:
    """Populate component index from type-stratified manifest.

    Rebuilds _component_index from manifest without importing components.
    Components will be lazy-loaded on first use. Loads from type-stratified
    sections (kernels/backends/steps) with only relevant metadata per type.

    Args:
        manifest: Manifest dict from _load_manifest()
    """
    # Load kernels
    for full_name, data in manifest.get('kernels', {}).items():
        source, name = full_name.split(':', 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(
                module=data['module'],
                attr=data['attr']
            ),
            # Restore kernel-specific metadata
            kernel_infer=data.get('infer'),
            kernel_domain=data.get('domain'),
            kernel_backends=data.get('backends')
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed kernel from manifest: {full_name}")

    # Load backends
    for full_name, data in manifest.get('backends', {}).items():
        source, name = full_name.split(':', 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.BACKEND,
            import_spec=ImportSpec(
                module=data['module'],
                attr=data['attr']
            ),
            # Restore backend-specific metadata
            backend_target=data.get('target'),
            backend_language=data.get('language')
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed backend from manifest: {full_name}")

    # Load steps
    for full_name, data in manifest.get('steps', {}).items():
        source, name = full_name.split(':', 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.STEP,
            import_spec=ImportSpec(
                module=data['module'],
                attr=data['attr']
            )
            # Steps have no type-specific metadata
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed step from manifest: {full_name}")
