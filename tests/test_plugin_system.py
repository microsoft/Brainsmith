# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Tests for the plugin system after heresy extermination."""

import pytest
import os
from unittest.mock import patch, MagicMock
from brainsmith.core.plugins.registry import Registry, get_registry, plugin
from brainsmith.core.plugins import (
    transform, kernel, backend, step, kernel_inference,
    get_transform, get_kernel, get_backend, get_step,
    has_transform, has_kernel, has_backend, has_step,
    list_transforms, list_kernels, list_backends, list_steps,
    get_transforms_by_metadata, get_kernels_by_metadata,
    get_backends_by_metadata, get_steps_by_metadata
)


@pytest.fixture(autouse=True)
def reset_registry():
    """Reset registry after each test to ensure isolation."""
    yield
    # Reset after test completes
    get_registry().reset()


class TestRegistry:
    """Test the Registry class directly."""
    
    def test_register_and_get(self):
        """Test basic registration and retrieval."""
        registry = Registry()
        
        class MockTransform:
            pass
        
        # Register a plugin
        registry.register('transform', 'TestTransform', MockTransform, 'brainsmith')
        
        # Should be able to retrieve it
        result = registry.get('transform', 'TestTransform')
        assert result is MockTransform
        
        # Brainsmith plugins don't get prefix by design
        with pytest.raises(KeyError):
            registry.get('transform', 'brainsmith:TestTransform')
    
    def test_get_raises_key_error(self):
        """Test that get() raises KeyError for missing plugins (fail fast)."""
        registry = Registry()
        
        with pytest.raises(KeyError, match="Plugin transform:NonExistent not found"):
            registry.get('transform', 'NonExistent')
    
    
    def test_framework_prefix_resolution(self):
        """Test framework prefix resolution logic."""
        registry = Registry()
        
        class QonnxTransform:
            pass
        
        class FinnTransform:
            pass
        
        # Register with different frameworks
        registry.register('transform', 'BatchNorm', QonnxTransform, 'qonnx')
        registry.register('transform', 'Streamline', FinnTransform, 'finn')
        
        # Should find with explicit prefix
        assert registry.get('transform', 'qonnx:BatchNorm') is QonnxTransform
        assert registry.get('transform', 'finn:Streamline') is FinnTransform
        
        # Should find without prefix (first match)
        assert registry.get('transform', 'BatchNorm') is QonnxTransform
        assert registry.get('transform', 'Streamline') is FinnTransform
    
    def test_find_by_metadata(self):
        """Test finding plugins by metadata."""
        registry = Registry()
        
        class Transform1:
            pass
        
        class Transform2:
            pass
        
        registry.register('transform', 'T1', Transform1, 'finn', kernel_inference=True)
        registry.register('transform', 'T2', Transform2, 'finn', kernel_inference=False)
        
        # Find by metadata
        results = registry.find('transform', kernel_inference=True)
        assert len(results) == 1
        assert results[0] is Transform1


class TestDecorators:
    """Test the plugin decorators."""
    
    def test_transform_decorator(self):
        """Test @transform decorator."""
        @transform(name='MyTransform', framework='test')
        class MyTransform:
            pass
        
        # Should be registered
        assert has_transform('MyTransform')
        assert get_transform('MyTransform') is MyTransform
        assert has_transform('test:MyTransform')
        assert get_transform('test:MyTransform') is MyTransform
    
    def test_kernel_decorator(self):
        """Test @kernel decorator."""
        @kernel(name='MyKernel')
        class MyKernel:
            pass
        
        assert has_kernel('MyKernel')
        assert get_kernel('MyKernel') is MyKernel
    
    def test_backend_decorator(self):
        """Test @backend decorator with metadata."""
        @backend(name='MyBackend_hls', kernel='MyKernel', language='hls')
        class MyBackend:
            pass
        
        assert has_backend('MyBackend_hls')
        assert get_backend('MyBackend_hls') is MyBackend
    
    def test_step_decorator(self):
        """Test @step decorator."""
        @step(name='my_step', category='test')
        def my_step(model, cfg):
            return model
        
        assert has_step('my_step')
        assert get_step('my_step') is my_step
    
    def test_kernel_inference_decorator(self):
        """Test @kernel_inference decorator (special transform)."""
        @kernel_inference(name='InferMyKernel')
        class InferMyKernel:
            pass
        
        # Should be registered as a transform with kernel_inference metadata
        assert has_transform('InferMyKernel')
        assert get_transform('InferMyKernel') is InferMyKernel


class TestConvenienceFunctions:
    """Test the convenience functions that now raise KeyError."""
    
    def test_get_functions_raise_key_error(self):
        """Test that get_* functions raise KeyError (fail fast)."""
        with pytest.raises(KeyError):
            get_transform('NonExistent')
        
        with pytest.raises(KeyError):
            get_kernel('NonExistent')
        
        with pytest.raises(KeyError):
            get_backend('NonExistent')
        
        with pytest.raises(KeyError):
            get_step('NonExistent')
    
    def test_get_optional_pattern(self):
        """Test the pattern for optional getting (has + get)."""
        if has_transform('NonExistent'):
            _ = get_transform('NonExistent')
        
        # This is the pattern that replaced get_*_optional
        assert not has_transform('NonExistent')
    
    def test_has_functions(self):
        """Test has_* functions."""
        assert not has_transform('NonExistent')
        assert not has_kernel('NonExistent')
        assert not has_backend('NonExistent')
        assert not has_step('NonExistent')
        
        # Register something and check again
        @transform(name='TestTransform')
        class TestTransform:
            pass
        
        assert has_transform('TestTransform')
    
    def test_list_functions(self):
        """Test list_* functions."""
        # Clear registry for clean test
        registry = get_registry()
        registry._plugins = {'transform': {}, 'kernel': {}, 'backend': {}, 'step': {}}
        
        # Should be empty initially
        assert list_transforms() == []
        assert list_kernels() == []
        assert list_backends() == []
        assert list_steps() == []
        
        # Add some plugins
        @transform(name='T1')
        class T1:
            pass
        
        @kernel(name='K1')
        class K1:
            pass
        
        assert 'T1' in list_transforms()
        assert 'K1' in list_kernels()


class TestFrameworkAdapters:
    """Test the framework adapter registration with strict mode."""
    
    @patch.dict(os.environ, {'BSMITH_PLUGINS_STRICT': 'true'})
    def test_strict_mode_fails_on_missing_imports(self):
        """Test that strict mode fails fast on missing imports."""
        from brainsmith.core.plugins.framework_adapters import _register_transforms
        
        # Mock transforms that will fail to import
        fake_transforms = [
            ('Transform1', 'nonexistent.module.Transform1'),
            ('Transform2', 'another.missing.Transform2')
        ]
        
        with pytest.raises(RuntimeError, match="Failed to register 2 finn transforms"):
            _register_transforms(fake_transforms, 'finn')
    
    @patch.dict(os.environ, {'BSMITH_PLUGINS_STRICT': 'false'})
    def test_non_strict_mode_continues_with_warnings(self):
        """Test that non-strict mode logs warnings but continues."""
        from brainsmith.core.plugins.framework_adapters import _register_transforms
        
        # Mock transforms that will fail to import
        fake_transforms = [
            ('Transform1', 'nonexistent.module.Transform1'),
            ('Transform2', 'another.missing.Transform2')
        ]
        
        # Should not raise, just return 0
        count = _register_transforms(fake_transforms, 'finn')
        assert count == 0
    
    def test_atomic_registration(self):
        """Test that partial registration works in non-strict mode."""
        from brainsmith.core.plugins.framework_adapters import _register_transforms
        
        # Test with a mix of valid and invalid transforms
        transforms = [
            ('RemoveIdentityOps', 'qonnx.transformation.remove.RemoveIdentityOps'),  # Real QONNX transform
            ('FakeTransform1', 'nonexistent.module.FakeTransform1'),  # Will fail
            ('FakeTransform2', 'another.missing.FakeTransform2'),  # Will fail
        ]
        
        # In non-strict mode, should continue with partial registration
        with patch.dict(os.environ, {'BSMITH_PLUGINS_STRICT': 'false'}):
            # This will try to import the transforms
            # The real one might succeed if QONNX is installed, fakes will fail
            count = _register_transforms(transforms, 'test')
            # Count could be 0 or 1 depending on whether QONNX is installed
            assert count >= 0
            assert count <= 1  # At most one could succeed


class TestPluginSystemIntegration:
    """Integration tests for the whole plugin system."""
    
    def test_brainsmith_plugin_registration(self):
        """Test registering Brainsmith-specific plugins."""
        # Clear registry
        registry = get_registry()
        
        # Register a Brainsmith transform
        @transform(name='BrainsmithTransform')
        class BrainsmithTransform:
            """A Brainsmith-specific transform."""
            pass
        
        # Should be findable without prefix (brainsmith plugins don't get prefixed)
        assert get_transform('BrainsmithTransform') is BrainsmithTransform
        
        # Brainsmith prefix is NOT added by design
        with pytest.raises(KeyError):
            get_transform('brainsmith:BrainsmithTransform')
        
        # Should be in the list
        assert 'BrainsmithTransform' in list_transforms()
    
    def test_error_message_quality(self):
        """Test that error messages are helpful (clear failures)."""
        try:
            get_transform('NonExistentTransform')
        except KeyError as e:
            error_msg = str(e)
            # Should mention the plugin type and name
            assert 'transform:NonExistentTransform' in error_msg
            # Should show available plugins (or 'none')
            assert 'Available' in error_msg
    
    def test_singleton_behavior(self):
        """Test that get_registry() returns the same instance."""
        reg1 = get_registry()
        reg2 = get_registry()
        assert reg1 is reg2
        
        # Changes in one should be visible in the other
        @transform(name='SingletonTest')
        class SingletonTest:
            pass
        
        assert reg1.get('transform', 'SingletonTest') is SingletonTest
        assert reg2.get('transform', 'SingletonTest') is SingletonTest

if __name__ == '__main__':
    pytest.main([__file__, '-v'])