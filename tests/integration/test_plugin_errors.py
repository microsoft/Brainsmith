"""Tests for plugin system error handling and edge cases."""

import pytest
from brainsmith.core.plugins import get_registry, transform, kernel, backend, step


class TestPluginErrors:
    """Test error handling in the plugin system."""
    
    def test_access_nonexistent_plugin(self):
        """Test that accessing non-existent plugins provides helpful errors."""
        registry = get_registry()
        
        # Test non-existent transform
        with pytest.raises(KeyError) as exc_info:
            registry.get("transform", "this_transform_does_not_exist")
        
        error_msg = str(exc_info.value)
        assert "Plugin transform:this_transform_does_not_exist not found" in error_msg
        assert "Available" in error_msg  # Should list available plugins
        
        # Test non-existent kernel
        with pytest.raises(KeyError) as exc_info:
            registry.get("kernel", "NonExistentKernel")
        
        error_msg = str(exc_info.value)
        assert "Plugin kernel:NonExistentKernel not found" in error_msg
        
        # Test non-existent step
        with pytest.raises(KeyError) as exc_info:
            registry.get("step", "missing_step")
        
        assert "Plugin step:missing_step not found" in str(exc_info.value)
    
    def test_duplicate_plugin_registration(self):
        """Test behavior when registering duplicate plugins."""
        registry = get_registry()
        
        # Register a plugin
        @transform(name="duplicate_test_transform")
        class FirstTransform:
            def apply(self, model):
                return model
        
        # Try to register another with the same name
        @transform(name="duplicate_test_transform")
        class SecondTransform:
            def apply(self, model):
                return model
        
        # The last registration should win (current behavior)
        retrieved = registry.get("transform", "duplicate_test_transform")
        assert retrieved == SecondTransform
        
        # Test with different plugin types but same name
        @kernel(name="duplicate_name")
        class TestKernel:
            pass
        
        @transform(name="duplicate_name")
        class TestTransform:
            pass
        
        # Should be able to get both since they're different types
        kernel_cls = registry.get("kernel", "duplicate_name")
        transform_cls = registry.get("transform", "duplicate_name")
        assert kernel_cls == TestKernel
        assert transform_cls == TestTransform
    
    def test_plugin_initialization_failures(self):
        """Test handling of plugins that fail during initialization."""
        
        @transform(name="failing_init_transform")
        class FailingInitTransform:
            def __init__(self):
                raise RuntimeError("Initialization failed!")
            
            def apply(self, model):
                return model
        
        registry = get_registry()
        transform_cls = registry.get("transform", "failing_init_transform")
        
        # Getting the class should work
        assert transform_cls == FailingInitTransform
        
        # But instantiation should fail with clear error
        with pytest.raises(RuntimeError) as exc_info:
            transform_instance = transform_cls()
        
        assert "Initialization failed!" in str(exc_info.value)
    
    def test_transform_execution_failures(self):
        """Test handling of transforms that fail during execution."""
        
        @transform(name="failing_execution_transform")
        class FailingExecutionTransform:
            def apply(self, model):
                raise ValueError("Transform execution failed!")
        
        registry = get_registry()
        transform_cls = registry.get("transform", "failing_execution_transform")
        transform_instance = transform_cls()
        
        # Create a mock model
        class MockModel:
            pass
        
        model = MockModel()
        
        # Execution should fail with the original error
        with pytest.raises(ValueError) as exc_info:
            transform_instance.apply(model)
        
        assert "Transform execution failed!" in str(exc_info.value)
    
    def test_invalid_plugin_type(self):
        """Test accessing invalid plugin types."""
        registry = get_registry()
        
        # The registry has fixed plugin types
        with pytest.raises(KeyError):
            registry.get("invalid_type", "some_plugin")
    
    def test_framework_prefix_resolution(self):
        """Test error handling in framework prefix resolution."""
        registry = get_registry()
        
        # Test non-existent plugin with framework prefix
        with pytest.raises(KeyError) as exc_info:
            registry.get("transform", "finn:NonExistentTransform")
        
        assert "finn:NonExistentTransform not found" in str(exc_info.value)
        
        # Test ambiguous name that could match multiple frameworks
        # This should fail if the name exists in multiple frameworks
        # Current implementation returns first match, but we test the error case
        with pytest.raises(KeyError) as exc_info:
            registry.get("transform", "ambiguous_name_that_does_not_exist")
        
        assert "ambiguous_name_that_does_not_exist not found" in str(exc_info.value)