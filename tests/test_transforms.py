# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tests for the transform import system.

Tests cover:
- Convention-based transform import
- Search pattern generation
- Caching behavior
- Error handling
- apply_transforms helper
"""

import pytest
from unittest.mock import Mock, MagicMock, patch

from brainsmith import transforms


class TestTransformImport:
    """Test the import_transform function."""

    def test_import_brainsmith_transform(self):
        """Test importing a Brainsmith transform."""
        # ExpandNorms is in brainsmith.primitives.transforms.cleanup
        ExpandNorms = transforms.import_transform('ExpandNorms')

        assert ExpandNorms is not None
        assert ExpandNorms.__name__ == 'ExpandNorms'

    def test_import_qonnx_transform(self):
        """Test importing a QONNX transform."""
        # FoldConstants is in qonnx.transformation.fold_constants
        FoldConstants = transforms.import_transform('FoldConstants')

        assert FoldConstants is not None
        assert FoldConstants.__name__ == 'FoldConstants'

    def test_import_finn_transform(self):
        """Test importing a FINN transform."""
        # Streamline is in finn.transformation.streamline
        Streamline = transforms.import_transform('Streamline')

        assert Streamline is not None
        assert 'Streamline' in Streamline.__name__

    def test_import_caching(self):
        """Test that transform imports are cached."""
        # Clear cache
        transforms._transform_cache.clear()

        # Import twice
        transform1 = transforms.import_transform('FoldConstants')
        transform2 = transforms.import_transform('FoldConstants')

        # Should be same object (cached)
        assert transform1 is transform2

        # Should be in cache
        assert 'FoldConstants' in transforms._transform_cache

    def test_import_nonexistent_transform(self):
        """Test error for nonexistent transform."""
        with pytest.raises(ImportError) as exc_info:
            transforms.import_transform('ThisTransformDefinitelyDoesNotExist')

        error_msg = str(exc_info.value)
        assert 'not found' in error_msg
        # Should show searched paths
        assert 'Searched:' in error_msg


class TestSearchPatterns:
    """Test the search pattern generation."""

    def test_to_snake_case(self):
        """Test CamelCase to snake_case conversion."""
        assert transforms._to_snake_case('FoldConstants') == 'fold_constants'
        assert transforms._to_snake_case('InferShapes') == 'infer_shapes'
        assert transforms._to_snake_case('GiveUniqueNodeNames') == 'give_unique_node_names'
        assert transforms._to_snake_case('ConvertQONNXtoFINN') == 'convert_qonnxto_finn'

    def test_get_search_patterns(self):
        """Test search pattern generation."""
        patterns = transforms._get_search_patterns('FoldConstants')

        # Should include Brainsmith patterns
        assert any('brainsmith.primitives.transforms.cleanup' in p for p in patterns)
        assert any('brainsmith.primitives.transforms.kernel_opt' in p for p in patterns)
        assert any('brainsmith.primitives.transforms.post_proc' in p for p in patterns)

        # Should include QONNX pattern
        assert any('qonnx.transformation.fold_constants' in p for p in patterns)

        # Should include FINN patterns
        assert any('finn.transformation.fold_constants' in p for p in patterns)
        assert any('finn.transformation.streamline' in p for p in patterns)
        assert any('finn.transformation.fpgadataflow' in p for p in patterns)

    def test_search_patterns_special_cases(self):
        """Test search patterns for transforms in special locations."""
        # ConvertQONNXtoFINN is in FINN's qonnx subpackage
        patterns = transforms._get_search_patterns('ConvertQONNXtoFINN')

        # Should include finn.transformation.qonnx
        assert any('finn.transformation.qonnx' in p for p in patterns)


class TestApplyTransforms:
    """Test the apply_transforms helper function."""

    def test_apply_transforms_single(self, mocker):
        """Test applying a single transform."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Apply single transform
        result = transforms.apply_transforms(mock_model, ['FoldConstants'])

        # Should have called transform once
        assert mock_model.transform.called
        assert result is mock_model

    def test_apply_transforms_multiple(self, mocker):
        """Test applying multiple transforms in sequence."""
        # Mock model
        mock_model = MagicMock()
        call_count = 0

        def transform_side_effect(transform_instance):
            nonlocal call_count
            call_count += 1
            return mock_model

        mock_model.transform = MagicMock(side_effect=transform_side_effect)

        # Apply multiple transforms
        transform_names = ['FoldConstants', 'InferShapes', 'RemoveIdentityOps']
        result = transforms.apply_transforms(mock_model, transform_names)

        # Should have called transform 3 times
        assert call_count == 3
        assert result is mock_model

    def test_apply_transforms_preserves_order(self, mocker):
        """Test that transforms are applied in the correct order."""
        # Mock model
        mock_model = MagicMock()
        applied_transforms = []

        def transform_side_effect(transform_instance):
            applied_transforms.append(transform_instance.__class__.__name__)
            return mock_model

        mock_model.transform = MagicMock(side_effect=transform_side_effect)

        # Apply transforms
        transform_names = ['FoldConstants', 'InferShapes', 'SortGraph']
        transforms.apply_transforms(mock_model, transform_names)

        # Should have been applied in order
        assert applied_transforms == ['FoldConstants', 'InferShapes', 'SortGraph']


class TestApplyTransformsWithParams:
    """Test the apply_transforms_with_params helper function."""

    def test_apply_with_params(self, mocker):
        """Test applying transforms with constructor parameters."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Apply transform with parameters
        transform_specs = [
            ('FoldConstants', {}),
            ('InferDataTypes', {'allow_scaledint_dtypes': False}),
        ]

        result = transforms.apply_transforms_with_params(mock_model, transform_specs)

        # Should have called transform twice
        assert mock_model.transform.call_count == 2
        assert result is mock_model

    def test_apply_with_params_empty_kwargs(self, mocker):
        """Test applying transform with empty kwargs."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Apply transform with empty kwargs
        transform_specs = [('FoldConstants', {})]
        result = transforms.apply_transforms_with_params(mock_model, transform_specs)

        # Should work fine
        assert mock_model.transform.called
        assert result is mock_model


class TestDebugMode:
    """Test debug model saving functionality."""

    def test_apply_transforms_with_debug_path(self, mocker, tmp_path):
        """Test that debug models are saved when debug_path provided."""
        # Mock model with save method
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)
        mock_model.save = MagicMock()

        # Apply with debug path
        debug_path = str(tmp_path / "debug")
        transforms.apply_transforms(mock_model, ['FoldConstants'], debug_path=debug_path)

        # Should have called save
        assert mock_model.save.called
        save_call = mock_model.save.call_args[0][0]
        assert 'debug' in save_call
        assert 'FoldConstants' in save_call

    def test_apply_transforms_without_debug_path(self, mocker):
        """Test that no debug models are saved without debug_path."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)
        mock_model.save = MagicMock()

        # Apply without debug path
        transforms.apply_transforms(mock_model, ['FoldConstants'])

        # Should not have called save
        assert not mock_model.save.called


class TestConventionMatching:
    """Test that conventions correctly find transforms."""

    @pytest.mark.parametrize("transform_name,expected_module_substring", [
        ('FoldConstants', 'qonnx.transformation'),
        ('InferShapes', 'qonnx.transformation'),
        ('RemoveIdentityOps', 'qonnx.transformation'),
        ('Streamline', 'finn.transformation'),
        ('ExpandNorms', 'brainsmith.primitives.transforms'),
    ])
    def test_convention_finds_transform(self, transform_name, expected_module_substring):
        """Test that conventions successfully find various transforms."""
        # This test actually imports the transforms
        transform_cls = transforms.import_transform(transform_name)

        assert transform_cls is not None
        assert transform_name in transform_cls.__name__
        # Check it came from expected location
        assert expected_module_substring in transform_cls.__module__


class TestTransformCompatibility:
    """Test compatibility with old registry-based code patterns."""

    def test_direct_import_still_works(self):
        """Test that transforms can still be imported and used directly."""
        # Import transform
        FoldConstants = transforms.import_transform('FoldConstants')

        # Should be instantiable
        transform_instance = FoldConstants()
        assert transform_instance is not None

    def test_mixed_transform_sources(self, mocker):
        """Test applying transforms from different sources in one call."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Mix of Brainsmith, FINN, and QONNX transforms
        transform_names = [
            'ExpandNorms',      # Brainsmith
            'FoldConstants',    # QONNX
            'Streamline',       # FINN
        ]

        # Should work without error
        result = transforms.apply_transforms(mock_model, transform_names)
        assert result is mock_model


class TestErrorRecovery:
    """Test error handling and recovery."""

    def test_import_error_shows_search_paths(self):
        """Test that import errors show all searched paths."""
        with pytest.raises(ImportError) as exc_info:
            transforms.import_transform('CompletelyNonexistentTransform')

        error_msg = str(exc_info.value)
        # Should show it searched multiple locations
        assert 'Searched:' in error_msg
        assert 'brainsmith.primitives.transforms' in error_msg
        assert 'qonnx.transformation' in error_msg
        assert 'finn.transformation' in error_msg

    def test_partial_module_import_failure(self):
        """Test behavior when module exists but class doesn't."""
        # Try to import nonexistent class from existing module
        with pytest.raises(ImportError) as exc_info:
            # We know qonnx.transformation exists, but this class doesn't
            transforms.import_transform('ThisClassDefinitelyDoesntExistInQonnx')

        # Should still raise ImportError with helpful message
        assert isinstance(exc_info.value, ImportError)


class TestPerformance:
    """Test performance characteristics."""

    def test_cache_avoids_reimport(self, mocker):
        """Test that caching avoids re-importing modules."""
        # Mock importlib to track import calls
        import_spy = mocker.spy(transforms.importlib, 'import_module')

        # Clear cache
        transforms._transform_cache.clear()

        # First import
        transforms.import_transform('FoldConstants')
        initial_call_count = import_spy.call_count

        # Second import
        transforms.import_transform('FoldConstants')
        final_call_count = import_spy.call_count

        # Should not have imported again (cached)
        assert final_call_count == initial_call_count

    def test_multiple_transforms_cache(self):
        """Test that multiple transforms are all cached."""
        # Clear cache
        transforms._transform_cache.clear()

        # Import several transforms
        transform_names = ['FoldConstants', 'InferShapes', 'RemoveIdentityOps']
        for name in transform_names:
            transforms.import_transform(name)

        # All should be in cache
        for name in transform_names:
            assert name in transforms._transform_cache
