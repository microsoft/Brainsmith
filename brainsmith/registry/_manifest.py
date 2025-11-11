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

import importlib.util
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any

from ._metadata import ComponentMetadata, ComponentType, ImportSpec
from ._state import _component_index

logger = logging.getLogger(__name__)


# ============================================================================
# Manifest Caching (Simplified - Direct JSON Operations)
# ============================================================================


def _build_manifest_from_index() -> dict[str, Any]:
    """Build type-stratified manifest from current component index.

    Creates manifest with components separated by type (kernels/backends/steps).
    Uses single generated_at timestamp for staleness detection instead of
    per-component mtimes. Eliminates null fields by storing only relevant
    metadata for each component type.

    Excludes 'custom' source components (ephemeral, not cached).

    Returns:
        Manifest dict with version, timestamp, and type-stratified components
    """
    kernels, backends, steps = {}, {}, {}

    for full_name, meta in _component_index.items():
        # Skip 'custom' source - these are ephemeral and must be reimported each run
        if meta.source == "custom":
            logger.debug(f"Skipping custom component in manifest: {full_name}")
            continue
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
            "module": meta.import_spec.module,
            "attr": meta.import_spec.attr,
            "file_path": file_path,
        }

        # Add to appropriate section with type-specific fields (no nulls)
        if meta.component_type == ComponentType.KERNEL:
            kernels[full_name] = {
                **base,
                "infer": meta.kernel_infer,
                "backends": meta.kernel_backends or [],
                "is_infrastructure": meta.is_infrastructure,
            }
        elif meta.component_type == ComponentType.BACKEND:
            backends[full_name] = {
                **base,
                "target": meta.backend_target,
                "language": meta.backend_language,
            }
        else:  # ComponentType.STEP
            steps[full_name] = base

    return {
        "version": "2.0",
        "generated_at": datetime.now().isoformat(),
        "kernels": kernels,
        "backends": backends,
        "steps": steps,
    }


def _save_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Save manifest dict to JSON file.

    Args:
        manifest: Manifest dict from _build_manifest_from_index()
        path: Path to save manifest (typically .brainsmith/component_manifest.json)
    """
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(manifest, f, indent=2)

    # Calculate total components for logging
    total = (
        len(manifest.get("kernels", {}))
        + len(manifest.get("backends", {}))
        + len(manifest.get("steps", {}))
    )
    logger.debug(f"Saved manifest with {total} components to {path}")


def _load_manifest(path: Path) -> dict[str, Any]:
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
    with open(path) as f:
        data = json.load(f)

    # Version validation
    version = data.get("version")
    if version == "1.0":
        # Old flat format - trigger regeneration
        logger.info(
            f"Manifest version 1.0 (old format) found at {path}. "
            f"Regenerating with type-stratified format (v2.0)."
        )
        raise ValueError("Old manifest version - will regenerate")
    elif version != "2.0":
        raise ValueError(
            f"Unknown manifest version: {version}. " f"Expected '2.0'. Cache will be regenerated."
        )

    # Structure validation for v2.0
    required_fields = ["generated_at", "kernels", "backends", "steps"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Manifest missing '{field}' field. Cache will be regenerated.")

    # Calculate total components for logging
    total = len(data["kernels"]) + len(data["backends"]) + len(data["steps"])
    logger.debug(
        f"Loaded manifest from {path}: "
        f"{len(data['kernels'])} kernels, "
        f"{len(data['backends'])} backends, "
        f"{len(data['steps'])} steps "
        f"(total: {total})"
    )
    return data


def _is_manifest_stale(manifest: dict[str, Any]) -> bool:
    """Check if manifest is stale by comparing file mtimes to manifest timestamp.

    A manifest is stale if any component file OR package __init__.py has been
    modified after the manifest was generated. Uses a single manifest timestamp
    instead of per-component mtimes for efficient staleness detection.

    Checks two types of files:
    1. Package __init__.py files (control which components are discovered)
    2. Component implementation files (actual kernel/backend/step code)

    Note: This only detects file modifications, not new files. Users must
    manually refresh (--refresh) when adding new component files.

    Args:
        manifest: Manifest dict from _load_manifest()

    Returns:
        True if manifest is stale and should be regenerated
    """
    # Parse manifest generation timestamp
    try:
        generated_at_str = manifest.get("generated_at")
        if not generated_at_str:
            logger.debug("Cache stale: missing generated_at timestamp")
            return True

        # Parse ISO format timestamp to Unix timestamp for comparison
        generated_at = datetime.fromisoformat(generated_at_str).timestamp()
    except (ValueError, TypeError) as e:
        logger.debug(f"Cache stale: invalid generated_at timestamp: {e}")
        return True

    # Helper function for DRY mtime checking
    def _check_file(file_path: Path | str, description: str) -> bool:
        """Check if file is newer than manifest. Returns True if stale."""
        try:
            current_mtime = os.path.getmtime(file_path)
            if current_mtime > generated_at:
                logger.debug(
                    f"Cache stale: {description} modified after manifest generation "
                    f"(file: {current_mtime}, manifest: {generated_at})"
                )
                return True
        except OSError as e:
            logger.debug(f"Cache stale: cannot access {description}: {e}")
            return True
        return False

    # Check package __init__.py files that control component discovery
    try:
        from brainsmith.registry.constants import SOURCE_PROJECT
        from brainsmith.settings import get_config

        config = get_config()

        # Core brainsmith packages (always checked)
        for package_name in ["kernels", "steps"]:
            init_file = config.bsmith_dir / "brainsmith" / package_name / "__init__.py"
            if init_file.exists() and _check_file(
                init_file, f"brainsmith.{package_name}.__init__.py"
            ):
                return True

        # Project source packages (structured layout)
        project_path = config.component_sources.get(SOURCE_PROJECT)
        if project_path and project_path.exists():
            # Check structured layout: project_dir/kernels/__init__.py, project_dir/steps/__init__.py
            for package_name in ["kernels", "steps"]:
                init_file = project_path / package_name / "__init__.py"
                if init_file.exists() and _check_file(
                    init_file, f"project.{package_name}.__init__.py"
                ):
                    return True

    except Exception as e:
        # If we can't check __init__.py files, treat as stale to be safe
        logger.warning(f"Could not check __init__.py files, treating cache as stale: {e}")
        return True

    # Check all component implementation files
    for section in ["kernels", "backends", "steps"]:
        for full_name, comp_data in manifest.get(section, {}).items():
            file_path = comp_data.get("file_path")
            if file_path and _check_file(file_path, full_name):
                return True

    return False


def _populate_index_from_manifest(manifest: dict[str, Any]) -> None:
    """Populate component index from type-stratified manifest.

    Rebuilds _component_index from manifest without importing components.
    Components will be lazy-loaded on first use. Loads from type-stratified
    sections (kernels/backends/steps) with only relevant metadata per type.

    Args:
        manifest: Manifest dict from _load_manifest()
    """
    # Load kernels
    for full_name, data in manifest.get("kernels", {}).items():
        source, name = full_name.split(":", 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.KERNEL,
            import_spec=ImportSpec(module=data["module"], attr=data["attr"]),
            # Restore kernel-specific metadata
            kernel_infer=data.get("infer"),
            kernel_backends=data.get("backends"),
            is_infrastructure=data.get("is_infrastructure", False),
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed kernel from manifest: {full_name}")

    # Load backends
    for full_name, data in manifest.get("backends", {}).items():
        source, name = full_name.split(":", 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.BACKEND,
            import_spec=ImportSpec(module=data["module"], attr=data["attr"]),
            # Restore backend-specific metadata
            backend_target=data.get("target"),
            backend_language=data.get("language"),
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed backend from manifest: {full_name}")

    # Load steps
    for full_name, data in manifest.get("steps", {}).items():
        source, name = full_name.split(":", 1)

        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=ComponentType.STEP,
            import_spec=ImportSpec(module=data["module"], attr=data["attr"]),
            # Steps have no type-specific metadata
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed step from manifest: {full_name}")
