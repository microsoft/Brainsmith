# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for unified component metadata structures.

Focused on testing actual logic, not Pydantic dataclass creation.
"""

import pytest
from brainsmith.registry import (
    ComponentMetadata,
    ImportSpec,
)


class TestComponentMetadata:
    """Tests for ComponentMetadata logic (not trivial dataclass creation)."""

    def test_metadata_loaded_obj_can_be_set(self):
        """Test that component loaded_obj state management works correctly."""
        spec = ImportSpec(
            module='brainsmith.kernels.layernorm.layernorm',
            attr='LayerNorm'
        )

        meta = ComponentMetadata(
            name='LayerNorm',
            source='brainsmith',
            component_type='kernel',
            import_spec=spec
        )

        # Initially unloaded
        assert not meta.is_loaded
        assert meta.loaded_obj is None

        # Simulate loading
        mock_obj = object()
        meta.loaded_obj = mock_obj

        # State should update
        assert meta.is_loaded
        assert meta.loaded_obj is mock_obj

    def test_full_name_property(self):
        """Test full_name property generates correct source:name format."""
        spec = ImportSpec(
            module='custom_source.kernels.test',
            attr='TestKernel'
        )

        meta = ComponentMetadata(
            name='TestKernel',
            source='custom_source',
            component_type='kernel',
            import_spec=spec
        )

        # Test string formatting logic
        assert meta.full_name == 'custom_source:TestKernel'

        # Test with different source
        meta2 = ComponentMetadata(
            name='MyStep',
            source='brainsmith',
            component_type='step',
            import_spec=ImportSpec(module='brainsmith.steps.core', attr='MyStep')
        )
        assert meta2.full_name == 'brainsmith:MyStep'
