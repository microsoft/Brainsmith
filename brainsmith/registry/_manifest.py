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
import datetime
import importlib.util
import logging
from pathlib import Path
from typing import Literal, Union, Dict, Any, Optional
from datetime import datetime as dt
from pydantic import BaseModel, Field, field_validator, ConfigDict

from ._state import _component_index
from ._metadata import ComponentMetadata, ImportSpec

logger = logging.getLogger(__name__)


# ============================================================================
# Pydantic Schema for Manifest Validation (Issue #8)
# ============================================================================

class ManifestComponentV1(BaseModel):
    """Component entry in manifest v1.

    Includes type-specific metadata fields to fix Issue #9 (metadata loss).
    Supports backwards compatibility with old manifests using 'class_name'.
    """
    type: Literal['step', 'kernel', 'backend']
    module: str
    class_name: str = Field(..., description="Component class/function name")
    attr: Optional[str] = None  # New format uses 'attr', backwards compat with 'class_name'
    metadata: Dict[str, Any] = Field(default_factory=dict)  # Deprecated, kept for backwards compat
    file_path: Optional[str] = None
    mtime: Optional[float] = None

    # Type-specific metadata (NEW - fixes issue #9)
    kernel_infer: Optional[Union[str, Dict[str, str]]] = None
    kernel_domain: Optional[str] = None
    backend_target: Optional[str] = None
    backend_language: Optional[str] = None

    @field_validator('attr', mode='before')
    @classmethod
    def attr_from_class_name(cls, v, info):
        """Backwards compat: use class_name if attr not provided."""
        if v is None and 'class_name' in info.data:
            return info.data['class_name']
        return v

    model_config = ConfigDict(extra='ignore')  # Ignore unknown fields for forward compat


class ManifestV1(BaseModel):
    """Manifest schema v1 with Pydantic validation (Issue #8)."""
    version: Literal["1.0"]
    generated_at: dt
    components: Dict[str, ManifestComponentV1]

    model_config = ConfigDict(
        json_encoders={dt: lambda v: v.isoformat()}
    )


# ============================================================================
# Manifest Caching (Arete: Eager Discovery + Optional Performance Cache)
# ============================================================================

def _build_manifest_from_index() -> ManifestV1:
    """Build validated manifest from current component index with full metadata.

    Converts _component_index to Pydantic-validated manifest for caching.
    Includes file mtimes for cache invalidation and type-specific metadata
    to fix Issue #9 (metadata loss).

    Returns:
        Validated ManifestV1 with complete component metadata
    """
    components = {}

    for full_name, meta in _component_index.items():
        # Resolve module to file path and get mtime
        file_path = None
        mtime = None
        try:
            spec = importlib.util.find_spec(meta.import_spec.module)
            if spec and spec.origin:
                file_path = spec.origin
                mtime = os.path.getmtime(file_path)
        except Exception as e:
            logger.debug(f"Could not get mtime for {meta.import_spec.module}: {e}")

        # Build component data with type-specific metadata (fixes #9!)
        component = ManifestComponentV1(
            type=meta.component_type,
            module=meta.import_spec.module,
            class_name=meta.import_spec.attr,
            attr=meta.import_spec.attr,
            metadata=meta.import_spec.extra,
            file_path=file_path,
            mtime=mtime,
            # Type-specific metadata (preserved now!)
            kernel_infer=meta.kernel_infer if meta.component_type == 'kernel' else None,
            kernel_domain=meta.kernel_domain if meta.component_type == 'kernel' else None,
            backend_target=meta.backend_target if meta.component_type == 'backend' else None,
            backend_language=meta.backend_language if meta.component_type == 'backend' else None,
        )

        components[full_name] = component

    return ManifestV1(
        version="1.0",
        generated_at=dt.now(),
        components=components
    )


def _save_manifest(manifest: ManifestV1, path: Path) -> None:
    """Save validated manifest to JSON file with Pydantic serialization.

    Args:
        manifest: Validated ManifestV1 from _build_manifest_from_index()
        path: Path to save manifest (typically .brainsmith/component_manifest.json)
    """
    # Create parent directory if needed
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, 'w') as f:
        f.write(manifest.model_dump_json(indent=2))

    logger.debug(f"Saved manifest with {len(manifest.components)} components to {path}")


def _load_manifest(path: Path) -> ManifestV1:
    """Load and validate manifest from JSON file with Pydantic validation.

    Args:
        path: Path to manifest file

    Returns:
        Validated ManifestV1

    Raises:
        FileNotFoundError: If manifest doesn't exist
        ValueError: If manifest fails Pydantic validation
        json.JSONDecodeError: If manifest is corrupted JSON
    """
    with open(path, 'r') as f:
        data = json.load(f)

    try:
        manifest = ManifestV1.model_validate(data)
        logger.debug(f"Loaded manifest with {len(manifest.components)} components from {path}")
        return manifest
    except Exception as e:
        logger.warning(f"Manifest validation failed: {e}")
        logger.info("Invalidating stale manifest, will regenerate")
        raise ValueError(f"Invalid manifest schema: {e}") from e


def _is_manifest_stale(manifest: ManifestV1) -> bool:
    """Check if manifest is stale by comparing file mtimes.

    A manifest is stale if any component file has been modified since the
    manifest was generated. This enables automatic cache invalidation when
    code changes.

    Note: This only detects file modifications, not new files. Users must
    manually refresh (--refresh) when adding new component files.

    Args:
        manifest: Validated ManifestV1 from _load_manifest()

    Returns:
        True if manifest is stale and should be regenerated
    """
    for full_name, comp_data in manifest.components.items():
        file_path = comp_data.file_path
        cached_mtime = comp_data.mtime

        # Skip if we don't have mtime data (old manifest format)
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


def _populate_index_from_manifest(manifest: ManifestV1) -> None:
    """Populate component index from cached manifest with full metadata.

    Rebuilds _component_index from manifest without importing components.
    Components will be lazy-loaded on first use. Type-specific metadata is
    restored from manifest, fixing Issue #9.

    Args:
        manifest: Validated ManifestV1 from _load_manifest()
    """
    for full_name, component in manifest.components.items():
        # Parse source and name from key (source:name)
        source, name = full_name.split(':', 1)

        # Build ComponentMetadata with type-specific fields (fixes #9!)
        meta = ComponentMetadata(
            name=name,
            source=source,
            component_type=component.type,
            import_spec=ImportSpec(
                module=component.module,
                attr=component.attr or component.class_name,  # Backwards compat
                extra=component.metadata
            ),
            # Restore type-specific metadata (preserved from manifest now!)
            kernel_infer=component.kernel_infer,
            kernel_domain=component.kernel_domain,
            backend_target=component.backend_target,
            backend_language=component.backend_language,
        )

        _component_index[full_name] = meta
        logger.debug(f"Indexed from manifest: {full_name}")
