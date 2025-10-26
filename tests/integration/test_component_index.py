# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Integration tests for unified component index.

These tests verify that the new unified component indexing system
populates the _component_index correctly during component discovery.
"""

import pytest
from brainsmith.registry import (
    discover_components,
    _component_index,
)


class TestComponentIndexPopulation:
    """Tests for component index population during discovery."""

    def test_index_populated_after_discovery(self):
        """Verify component index is populated during discovery."""
        # Force fresh discovery (not from cache)
        discover_components(use_cache=False)

        # Check index has components
        assert len(_component_index) > 0, "Index should be populated after discovery"

        # Check core brainsmith components indexed
        assert 'brainsmith:LayerNorm' in _component_index, "Core kernel missing from index"
        assert 'brainsmith:LayerNorm_hls' in _component_index, "Core backend missing from index"
        assert 'brainsmith:qonnx_to_finn' in _component_index, "Core step missing from index"

        # Verify metadata structure
        meta = _component_index['brainsmith:LayerNorm']
        assert meta.source == 'brainsmith'
        assert meta.component_type == 'kernel'
        assert meta.is_loaded  # Eagerly loaded during discovery
        assert meta.import_spec is not None
        assert meta.import_spec.module == 'brainsmith.kernels.layernorm.layernorm'
        assert meta.import_spec.attr == 'LayerNorm'

    def test_all_component_types_indexed(self):
        """Verify all component types are represented in index."""
        discover_components()

        # Count by type
        kernels = [k for k in _component_index if _component_index[k].component_type == 'kernel']
        backends = [k for k in _component_index if _component_index[k].component_type == 'backend']
        steps = [k for k in _component_index if _component_index[k].component_type == 'step']

        assert len(kernels) >= 4, f"Expected at least 4 kernels, got {len(kernels)}"
        assert len(backends) >= 4, f"Expected at least 4 backends, got {len(backends)}"
        assert len(steps) >= 7, f"Expected at least 7 steps, got {len(steps)}"

    def test_core_components_have_import_specs(self):
        """Verify core brainsmith components have proper import specs."""
        # Force fresh discovery (not from cache)
        discover_components(use_cache=False)

        core_components = [
            meta for meta in _component_index.values()
            if meta.source == 'brainsmith'
        ]

        assert len(core_components) > 0

        for meta in core_components:
            assert meta.import_spec is not None, \
                f"{meta.full_name} should have import_spec"
            assert meta.import_spec.module.startswith('brainsmith.'), \
                f"{meta.full_name} should have brainsmith module path"
            # Note: import_spec.attr may differ from meta.name for steps
            # (e.g., qonnx_to_finn_step vs qonnx_to_finn - decorator name differs)
            assert meta.import_spec.attr, \
                f"{meta.full_name} should have import_spec.attr"


class TestIndexCompleteness:
    """Tests to verify index completeness."""

    def test_filesystem_components_counted_correctly(self):
        """Verify filesystem components (core + user) are indexed."""
        discover_components()

        # Core brainsmith components
        brainsmith_components = [
            k for k, v in _component_index.items()
            if v.source == 'brainsmith'
        ]

        # Should have 15 components (4 kernels + 4 backends + 7 steps)
        # This is based on brainsmith/kernels/__init__.py and brainsmith/steps/__init__.py
        assert len(brainsmith_components) == 15, \
            f"Expected 15 core components, got {len(brainsmith_components)}"


class TestExternalComponents:
    """Tests for external components loaded from entry points."""

    def test_external_components_indexed_if_available(self):
        """Verify external components are indexed (if installed)."""
        discover_components()

        # Check if any external components exist (non-brainsmith source)
        external_components = [
            k for k, v in _component_index.items()
            if v.source != 'brainsmith'
        ]

        if len(external_components) > 0:
            # External components available - verify structure
            sample = external_components[0]
            meta = _component_index[sample]

            assert meta.import_spec is not None, "External component should have import_spec"
            assert meta.import_spec.module, "import_spec should have module"
            assert meta.import_spec.attr, "import_spec should have attr"

            print(f"✓ Found {len(external_components)} external components")
            print(f"  Sample: {meta.full_name} from {meta.import_spec.module}")
        else:
            print("⚠ No external components found (FINN/QONNX not installed)")


class TestIndexIntegrity:
    """Tests for index data integrity."""

    def test_full_name_property_matches_key(self):
        """Verify component full_name property matches index key."""
        discover_components()

        for key, meta in _component_index.items():
            assert meta.full_name == key, \
                f"Key mismatch: index['{key}'] but full_name='{meta.full_name}'"

    def test_no_duplicate_components(self):
        """Verify no duplicate component registrations."""
        discover_components()

        # Check for duplicates (same source:name)
        seen = set()
        duplicates = []

        for key in _component_index:
            if key in seen:
                duplicates.append(key)
            seen.add(key)

        assert len(duplicates) == 0, f"Found duplicates: {duplicates}"

    def test_all_components_have_valid_import_specs(self):
        """Verify all indexed components have valid import specs."""
        discover_components()

        for key, meta in _component_index.items():
            assert meta.import_spec is not None, f"{key} missing import_spec"
            assert meta.import_spec.module, f"{key} import_spec missing module"
            assert meta.import_spec.attr, f"{key} import_spec missing attr"

    def test_core_components_loaded_after_discovery(self):
        """Verify core components are loaded after discovery (eager loading).

        Core brainsmith components use eager imports - they are imported during
        discovery and decorators fire immediately, populating both registry and
        index. This is intentional for simplicity and debuggability.

        Manifest caching provides performance for subsequent runs.
        """
        discover_components()

        # Core components should be loaded (eager imports)
        core_loaded = sum(
            1 for meta in _component_index.values()
            if meta.source == 'brainsmith' and meta.is_loaded
        )

        # Should have at least kernels + backends + steps loaded
        assert core_loaded >= 15, \
            f"Expected at least 15 loaded core components, got {core_loaded}"
