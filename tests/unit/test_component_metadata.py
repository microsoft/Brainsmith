# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Unit tests for unified component metadata structures."""

import pytest
from brainsmith.loader import (
    ComponentMetadata,
    ImportSpec,
    LoadState,
)


class TestImportSpec:
    """Tests for ImportSpec dataclass."""

    def test_basic_import_spec(self):
        """Test creating a basic import spec."""
        spec = ImportSpec(
            module='finn.custom_op.fc',
            attr='StreamingFCLayer'
        )

        assert spec.module == 'finn.custom_op.fc'
        assert spec.attr == 'StreamingFCLayer'
        assert spec.extra == {}

    def test_import_spec_with_extra_metadata(self):
        """Test import spec with extra metadata (infer_transform)."""
        spec = ImportSpec(
            module='finn.custom_op.fc',
            attr='StreamingFCLayer',
            extra={
                'infer_transform': {
                    'module': 'finn.custom_op.fc',
                    'class_name': 'InferStreamingFCLayer'
                }
            }
        )

        assert spec.module == 'finn.custom_op.fc'
        assert spec.attr == 'StreamingFCLayer'
        assert 'infer_transform' in spec.extra
        assert spec.extra['infer_transform']['module'] == 'finn.custom_op.fc'


class TestComponentMetadata:
    """Tests for ComponentMetadata dataclass."""

    def test_core_component_metadata_creation(self):
        """Test metadata for core brainsmith component."""
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

        assert meta.name == 'LayerNorm'
        assert meta.source == 'brainsmith'
        assert meta.component_type == 'kernel'
        assert meta.full_name == 'brainsmith:LayerNorm'
        assert meta.state == LoadState.DISCOVERED
        assert meta.import_spec is spec
        assert meta.loaded_obj is None

    def test_plugin_component_metadata_creation(self):
        """Test metadata for plugin component (FINN)."""
        spec = ImportSpec(
            module='finn.custom_op.fc',
            attr='StreamingFCLayer'
        )

        meta = ComponentMetadata(
            name='StreamingFCLayer',
            source='finn',
            component_type='kernel',
            import_spec=spec
        )

        assert meta.name == 'StreamingFCLayer'
        assert meta.source == 'finn'
        assert meta.component_type == 'kernel'
        assert meta.full_name == 'finn:StreamingFCLayer'
        assert meta.state == LoadState.DISCOVERED
        assert meta.import_spec is spec
        assert meta.loaded_obj is None

    def test_user_component_metadata_creation(self):
        """Test metadata for user directory component."""
        spec = ImportSpec(
            module='user.kernels.custom_kernel',
            attr='CustomKernel'
        )

        meta = ComponentMetadata(
            name='CustomKernel',
            source='user',
            component_type='kernel',
            import_spec=spec
        )

        assert meta.full_name == 'user:CustomKernel'
        assert meta.source == 'user'
        assert meta.import_spec is spec

    def test_backend_component_metadata(self):
        """Test metadata for backend component."""
        spec = ImportSpec(
            module='brainsmith.kernels.layernorm.layernorm_hls',
            attr='LayerNorm_hls',
            extra={
                'target_kernel': 'brainsmith:LayerNorm',
                'language': 'hls'
            }
        )

        meta = ComponentMetadata(
            name='LayerNorm_hls',
            source='brainsmith',
            component_type='backend',
            import_spec=spec
        )

        assert meta.component_type == 'backend'
        assert meta.full_name == 'brainsmith:LayerNorm_hls'
        assert meta.import_spec.extra['target_kernel'] == 'brainsmith:LayerNorm'

    def test_step_component_metadata(self):
        """Test metadata for step component."""
        spec = ImportSpec(
            module='brainsmith.steps.core_steps',
            attr='streamline'
        )

        meta = ComponentMetadata(
            name='streamline',
            source='brainsmith',
            component_type='step',
            import_spec=spec
        )

        assert meta.component_type == 'step'
        assert meta.full_name == 'brainsmith:streamline'

    def test_metadata_validation_fails_with_no_import_spec(self):
        """Ensure validation catches missing import_spec."""
        with pytest.raises(ValueError, match="missing import_spec"):
            ComponentMetadata(
                name='Test',
                source='test',
                component_type='kernel'
            )

    def test_metadata_state_can_be_updated(self):
        """Test that component state can be updated after creation."""
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

        assert meta.state == LoadState.DISCOVERED
        assert meta.loaded_obj is None

        # Simulate loading
        mock_obj = object()
        meta.state = LoadState.LOADED
        meta.loaded_obj = mock_obj

        assert meta.state == LoadState.LOADED
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

        assert meta.full_name == 'custom_source:TestKernel'

    def test_plugin_component_with_complex_extra(self):
        """Test plugin component with complex extra metadata."""
        spec = ImportSpec(
            module='finn.custom_op.fc',
            attr='StreamingFCLayer',
            extra={
                'infer_transform': {
                    'module': 'finn.custom_op.fc',
                    'class_name': 'InferStreamingFCLayer'
                },
                'target_kernel': 'finn:StreamingFCLayer',
                'language': 'hls'
            }
        )

        meta = ComponentMetadata(
            name='StreamingFCLayer_hls',
            source='finn',
            component_type='backend',
            import_spec=spec
        )

        assert meta.import_spec.extra['target_kernel'] == 'finn:StreamingFCLayer'
        assert meta.import_spec.extra['language'] == 'hls'


class TestLoadState:
    """Tests for LoadState enum."""

    def test_load_states_exist(self):
        """Test that load states are defined."""
        assert hasattr(LoadState, 'DISCOVERED')
        assert hasattr(LoadState, 'LOADED')

    def test_load_states_are_distinct(self):
        """Test that load states are distinct values."""
        assert LoadState.DISCOVERED != LoadState.LOADED
