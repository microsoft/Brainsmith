# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Integration tests for the plugin system.

Tests real-world usage patterns:
- DSE pipeline integration
- Step execution
- Kernel inference workflow
- End-to-end plugin loading
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from brainsmith import (
    get_step, get_kernel, get_kernel_infer, get_backend,
    list_steps, list_kernels, list_backends,
    import_transform, apply_transforms
)


class TestPluginSystemIntegration:
    """Test that the plugin system works end-to-end."""

    def test_complete_step_workflow(self):
        """Test complete workflow: discover, get, execute step."""
        # 1. Check step exists
        from brainsmith.loader import has_step
        assert has_step('qonnx_to_finn')

        # 2. Get the step
        step_fn = get_step('qonnx_to_finn')
        assert callable(step_fn)

        # 3. Step should be importable
        assert step_fn.__name__ == 'qonnx_to_finn_step'

    def test_complete_kernel_workflow(self):
        """Test complete workflow: kernel, infer, backend."""
        # 1. Get kernel class
        LayerNorm = get_kernel('LayerNorm')
        assert LayerNorm is not None

        # 2. Get InferTransform from kernel metadata
        InferLayerNorm = get_kernel_infer('LayerNorm')
        assert InferLayerNorm is not None

        # 3. Get backend
        LayerNormHLS = get_backend('LayerNorm', 'hls')
        assert LayerNormHLS is not None

        # 4. All should be different classes
        assert LayerNorm is not InferLayerNorm
        assert LayerNorm is not LayerNormHLS

    def test_complete_transform_workflow(self):
        """Test complete workflow: import and apply transforms."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # 1. Import transform
        FoldConstants = import_transform('FoldConstants')
        assert FoldConstants is not None

        # 2. Can instantiate and use
        transform = FoldConstants()
        assert transform is not None

        # 3. Can apply via helper
        result = apply_transforms(mock_model, ['FoldConstants'])
        assert result is mock_model


class TestDSEIntegration:
    """Test DSE-specific integration points."""

    def test_step_validation_workflow(self):
        """Test how DSE validates steps in blueprints."""
        from brainsmith.loader import has_step

        # Valid steps should pass
        valid_steps = ['qonnx_to_finn', 'specialize_layers', 'infer_kernels']
        for step in valid_steps:
            assert has_step(step), f"Step {step} should be valid"

        # Invalid steps should fail
        assert not has_step('this_is_not_a_real_step')

    def test_kernel_backend_resolution(self):
        """Test how DSE resolves kernel backends."""
        # Get available backends for a kernel
        backends = list_backends('LayerNorm')
        assert 'hls' in backends

        # Should be able to get each backend
        for lang in backends:
            backend_cls = get_backend('LayerNorm', lang)
            assert backend_cls is not None

    def test_step_execution_pattern(self):
        """Test typical step execution pattern from DSE runner."""
        # This is how DSE runner uses steps
        step_name = 'qonnx_to_finn'

        # 1. Get step function
        step_fn = get_step(step_name)

        # 2. Should be callable with (model, cfg) signature
        import inspect
        sig = inspect.signature(step_fn)
        params = list(sig.parameters.keys())
        assert 'model' in params
        assert 'cfg' in params


class TestKernelInferenceIntegration:
    """Test kernel inference step integration."""

    def test_infer_kernels_step_workflow(self):
        """Test how infer_kernels step uses kernel metadata."""
        # Get the infer_kernels step
        infer_step = get_step('infer_kernels')

        # Mock cfg with kernel selections
        mock_cfg = Mock()
        mock_cfg.kernel_selections = [('LayerNorm', 'hls')]

        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Should execute without error
        result = infer_step(mock_model, mock_cfg)
        assert result is mock_model

        # Should have called transform (to apply InferLayerNorm)
        assert mock_model.transform.called

    def test_kernel_metadata_completeness(self):
        """Test that kernel metadata is complete for all kernels."""
        kernels = list_kernels()

        for kernel_name in kernels:
            # Should have kernel class
            kernel_cls = get_kernel(kernel_name)
            assert kernel_cls is not None

            # Should have at least one backend
            backends = list_backends(kernel_name)
            assert len(backends) > 0, f"Kernel {kernel_name} has no backends"

            # Each backend should be gettable
            for lang in backends:
                backend_cls = get_backend(kernel_name, lang)
                assert backend_cls is not None


class TestTransformIntegration:
    """Test transform system integration with steps."""

    def test_step_using_apply_transforms(self):
        """Test steps that use apply_transforms internally."""
        # Get a step that uses apply_transforms
        step_fn = get_step('qonnx_to_finn')

        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Mock cfg
        mock_cfg = Mock()

        # Should execute
        result = step_fn(mock_model, mock_cfg)
        assert result is mock_model

    def test_step_using_direct_transform_import(self):
        """Test steps that import transforms directly."""
        # Get a step that imports transforms with parameters
        step_fn = get_step('specialize_layers')

        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Mock cfg
        mock_cfg = Mock()
        mock_cfg.specialize_layers_config_file = None
        mock_cfg._resolve_fpga_part = Mock(return_value='xczu9eg-ffvb1156-2-e')

        # Should execute
        result = step_fn(mock_model, mock_cfg)
        assert result is mock_model


class TestCrossCuttingConcerns:
    """Test cross-cutting concerns like performance and compatibility."""

    def test_no_duplicate_imports(self):
        """Test that repeated lookups don't cause duplicate imports."""
        from brainsmith import loader, transforms

        # Clear caches
        loader._import_cache.clear()
        transforms._transform_cache.clear()

        # Get same step multiple times
        step1 = get_step('qonnx_to_finn')
        step2 = get_step('qonnx_to_finn')
        step3 = get_step('qonnx_to_finn')

        # Should be same object (cached)
        assert step1 is step2 is step3

        # Get same transform multiple times
        t1 = import_transform('FoldConstants')
        t2 = import_transform('FoldConstants')
        t3 = import_transform('FoldConstants')

        # Should be same object (cached)
        assert t1 is t2 is t3

    def test_listing_functions_consistency(self):
        """Test that listing functions return consistent results."""
        # List steps twice
        steps1 = list_steps()
        steps2 = list_steps()

        # Should be identical
        assert steps1 == steps2

        # List kernels twice
        kernels1 = list_kernels()
        kernels2 = list_kernels()

        # Should be identical
        assert kernels1 == kernels2

    def test_brainsmith_finn_qonnx_coexistence(self):
        """Test that Brainsmith, FINN, and QONNX components coexist."""
        # Get Brainsmith component
        brainsmith_kernel = get_kernel('LayerNorm')
        assert 'brainsmith' in brainsmith_kernel.__module__

        # Get FINN component
        finn_kernel = get_kernel('MVAU')
        assert 'finn' in finn_kernel.__module__

        # Get QONNX transform
        qonnx_transform = import_transform('FoldConstants')
        assert 'qonnx' in qonnx_transform.__module__

        # All should work together
        assert brainsmith_kernel is not None
        assert finn_kernel is not None
        assert qonnx_transform is not None


class TestErrorHandlingIntegration:
    """Test error handling across the plugin system."""

    def test_missing_step_error_flow(self):
        """Test error handling for missing steps."""
        from brainsmith.loader import has_step

        # Check should return False
        assert not has_step('nonexistent_step')

        # Get should raise clear error
        with pytest.raises(KeyError) as exc_info:
            get_step('nonexistent_step')

        assert 'not found' in str(exc_info.value)

    def test_missing_transform_error_flow(self):
        """Test error handling for missing transforms."""
        with pytest.raises(ImportError) as exc_info:
            import_transform('NonexistentTransform')

        # Should show searched paths
        assert 'Searched:' in str(exc_info.value)

    def test_missing_kernel_backend_error_flow(self):
        """Test error handling for missing backends."""
        # Kernel exists
        assert get_kernel('LayerNorm')

        # But requesting invalid backend should fail
        with pytest.raises(KeyError) as exc_info:
            get_backend('LayerNorm', 'invalid_language')

        assert 'not available' in str(exc_info.value)


class TestBackwardCompatibility:
    """Test backward compatibility patterns."""

    def test_import_from_brainsmith_package(self):
        """Test that imports from main package still work."""
        # These should all work from top-level import
        from brainsmith import (
            get_step, get_kernel,
            list_steps, list_kernels,
            import_transform, apply_transforms
        )

        # All should be callable/usable
        assert callable(get_step)
        assert callable(get_kernel)
        assert callable(list_steps)
        assert callable(list_kernels)
        assert callable(import_transform)
        assert callable(apply_transforms)

    def test_transform_utils_still_works(self):
        """Test that transform_utils works."""
        from brainsmith.primitives.utils import apply_transforms as legacy_apply

        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Should still work
        result = legacy_apply(mock_model, ['FoldConstants'])
        assert result is mock_model


class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    def test_blueprint_step_resolution(self):
        """Test resolving steps from a blueprint-like list."""
        # Typical blueprint steps
        blueprint_steps = [
            'qonnx_to_finn',
            'streamline',
            'specialize_layers',
            'infer_kernels',
        ]

        # All should be valid
        from brainsmith.loader import has_step
        for step in blueprint_steps:
            assert has_step(step), f"Blueprint step {step} should exist"

        # All should be retrievable
        for step in blueprint_steps:
            step_fn = get_step(step)
            assert callable(step_fn)

    def test_kernel_backend_selection(self):
        """Test typical kernel/backend selection workflow."""
        # User selects kernels in blueprint
        kernel_selections = [
            ('LayerNorm', 'hls'),
            ('MVAU', 'rtl'),
            ('Thresholding', 'hls'),
        ]

        for kernel_name, backend_lang in kernel_selections:
            # Should be able to get InferTransform
            infer_transform = get_kernel_infer(kernel_name)
            assert infer_transform is not None

            # Should be able to get backend
            backend = get_backend(kernel_name, backend_lang)
            assert backend is not None

    def test_mixed_transform_pipeline(self):
        """Test a realistic transform pipeline mixing sources."""
        # Mock model
        mock_model = MagicMock()
        mock_model.transform = MagicMock(return_value=mock_model)

        # Realistic pipeline: cleanup → FINN → QONNX → custom
        pipeline = [
            'ExpandNorms',          # Brainsmith cleanup
            'FoldConstants',        # QONNX
            'ConvertQONNXtoFINN',  # FINN
            'InferShapes',          # QONNX
            'RemoveIdentityOps',    # QONNX
        ]

        # Should apply all without error
        result = apply_transforms(mock_model, pipeline)
        assert result is mock_model

        # Should have called transform for each
        assert mock_model.transform.call_count == len(pipeline)
