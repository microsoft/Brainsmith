"""
Test Decorator Auto-Registration

Tests for unified decorators with decoration-time registration.
"""

import pytest
from unittest.mock import patch

from brainsmith.core.plugins.decorators import (
    plugin, transform, kernel, backend, step, kernel_inference,
    get_plugin_metadata, is_plugin_class, list_plugin_classes
)
from brainsmith.core.plugins.registry import get_registry, reset_registry


class TestPluginDecorator:
    def setup_method(self):
        reset_registry()
    
    def teardown_method(self):
        reset_registry()
    
    def test_transform_decoration_and_registration(self):
        """Test transform decorator with auto-registration."""
        @plugin(type="transform", name="TestTransform", stage="cleanup", framework="brainsmith")
        class TestTransform:
            def apply(self, model):
                return model, False
        
        # Test metadata stored on class
        assert hasattr(TestTransform, '_plugin_metadata')
        metadata = TestTransform._plugin_metadata
        assert metadata['name'] == "TestTransform"
        assert metadata['type'] == "transform"
        assert metadata['stage'] == "cleanup"
        assert metadata['framework'] == "brainsmith"
        
        # Test auto-registration in registry
        registry = get_registry()
        assert "TestTransform" in registry.transforms
        assert registry.transforms["TestTransform"] == TestTransform
        
        # Test indexing
        assert "TestTransform" in registry.transforms_by_stage["cleanup"]
        assert "TestTransform" in registry.framework_transforms["brainsmith"]
    
    def test_kernel_decoration_and_registration(self):
        """Test kernel decorator with auto-registration."""
        @plugin(type="kernel", name="TestKernel", op_type="TestOp")
        class TestKernel:
            pass
        
        # Test metadata
        metadata = TestKernel._plugin_metadata
        assert metadata['name'] == "TestKernel"
        assert metadata['type'] == "kernel"
        assert metadata['op_type'] == "TestOp"
        assert metadata['framework'] == "brainsmith"  # default
        
        # Test auto-registration
        registry = get_registry()
        assert "TestKernel" in registry.kernels
        assert registry.kernels["TestKernel"] == TestKernel
    
    def test_backend_decoration_and_registration(self):
        """Test backend decorator with auto-registration."""
        @plugin(type="backend", name="TestBackend", kernel="TestKernel", backend_type="hls")
        class TestBackend:
            pass
        
        # Test metadata
        metadata = TestBackend._plugin_metadata
        assert metadata['name'] == "TestBackend"
        assert metadata['type'] == "backend"
        assert metadata['kernel'] == "TestKernel"
        assert metadata['backend_type'] == "hls"
        
        # Test auto-registration
        registry = get_registry()
        backend_key = "TestKernel_hls"
        assert backend_key in registry.backends
        assert registry.backends[backend_key] == TestBackend
        
        # Test kernel indexing
        assert "TestKernel" in registry.backends_by_kernel
        assert "hls" in registry.backends_by_kernel["TestKernel"]
    
    def test_step_decoration_and_registration(self):
        """Test step decorator with auto-registration."""
        @plugin(type="step", name="TestStep", category="metadata")
        class TestStep:
            def execute(self, model, config):
                return model
        
        # Test metadata
        metadata = TestStep._plugin_metadata
        assert metadata['name'] == "TestStep"
        assert metadata['type'] == "step"
        assert metadata['category'] == "metadata"
        
        # Test registration as transform (steps are special transforms)
        registry = get_registry()
        assert "TestStep" in registry.transforms
        # Should be indexed by category as stage
        assert "TestStep" in registry.transforms_by_stage["metadata"]
    
    def test_kernel_inference_decoration(self):
        """Test kernel_inference decorator."""
        @plugin(type="kernel_inference", name="TestInference", kernel="TestKernel")
        class TestInference:
            def apply(self, model):
                return model, False
        
        # Test metadata
        metadata = TestInference._plugin_metadata
        assert metadata['name'] == "TestInference"
        assert metadata['type'] == "kernel_inference"
        assert metadata['kernel'] == "TestKernel"
        
        # Test registration as transform
        registry = get_registry()
        assert "TestInference" in registry.transforms
    
    def test_name_defaults_to_class_name(self):
        """Test that name defaults to class name when not specified."""
        @plugin(type="transform", stage="cleanup")
        class MyDefaultNameTransform:
            pass
        
        metadata = MyDefaultNameTransform._plugin_metadata
        assert metadata['name'] == "MyDefaultNameTransform"
        
        registry = get_registry()
        assert "MyDefaultNameTransform" in registry.transforms
    
    def test_validation_errors(self):
        """Test validation of plugin metadata."""
        # This should generate warnings but not fail
        @plugin(type="transform", name="InvalidTransform")  # Missing stage or kernel
        class InvalidTransform:
            pass
        
        # Should still register despite warnings
        registry = get_registry()
        assert "InvalidTransform" in registry.transforms
        
        # Backend missing required fields
        @plugin(type="backend", name="InvalidBackend")  # Missing kernel and backend_type
        class InvalidBackend:
            pass
        
        # Should still register
        assert "InvalidBackend" in registry.backends


class TestConvenienceDecorators:
    def setup_method(self):
        reset_registry()
    
    def teardown_method(self):
        reset_registry()
    
    def test_transform_convenience_decorator(self):
        """Test transform convenience decorator."""
        @transform(name="ConvenienceTransform", stage="cleanup")
        class ConvenienceTransform:
            pass
        
        metadata = ConvenienceTransform._plugin_metadata
        assert metadata['type'] == "transform"
        assert metadata['name'] == "ConvenienceTransform"
        assert metadata['stage'] == "cleanup"
        assert metadata['framework'] == "brainsmith"
        
        registry = get_registry()
        assert "ConvenienceTransform" in registry.transforms
    
    def test_kernel_convenience_decorator(self):
        """Test kernel convenience decorator."""
        @kernel(name="ConvenienceKernel", op_type="TestOp")
        class ConvenienceKernel:
            pass
        
        metadata = ConvenienceKernel._plugin_metadata
        assert metadata['type'] == "kernel"
        assert metadata['name'] == "ConvenienceKernel"
        assert metadata['op_type'] == "TestOp"
        
        registry = get_registry()
        assert "ConvenienceKernel" in registry.kernels
    
    def test_backend_convenience_decorator(self):
        """Test backend convenience decorator."""
        @backend(name="ConvenienceBackend", kernel="TestKernel", backend_type="rtl")
        class ConvenienceBackend:
            pass
        
        metadata = ConvenienceBackend._plugin_metadata
        assert metadata['type'] == "backend"
        assert metadata['kernel'] == "TestKernel"
        assert metadata['backend_type'] == "rtl"
        
        registry = get_registry()
        assert "TestKernel_rtl" in registry.backends
    
    def test_step_convenience_decorator(self):
        """Test step convenience decorator."""
        @step(name="ConvenienceStep", category="preprocessing")
        class ConvenienceStep:
            pass
        
        metadata = ConvenienceStep._plugin_metadata
        assert metadata['type'] == "step"
        assert metadata['category'] == "preprocessing"
        
        registry = get_registry()
        assert "ConvenienceStep" in registry.transforms
    
    def test_kernel_inference_convenience_decorator(self):
        """Test kernel_inference convenience decorator."""
        @kernel_inference(name="ConvenienceInference", kernel="TestKernel")
        class ConvenienceInference:
            pass
        
        metadata = ConvenienceInference._plugin_metadata
        assert metadata['type'] == "kernel_inference"
        assert metadata['kernel'] == "TestKernel"
        
        registry = get_registry()
        assert "ConvenienceInference" in registry.transforms


class TestUtilityFunctions:
    def setup_method(self):
        reset_registry()
    
    def teardown_method(self):
        reset_registry()
    
    def test_get_plugin_metadata(self):
        """Test get_plugin_metadata utility function."""
        @plugin(type="transform", name="TestTransform", stage="cleanup")
        class TestTransform:
            pass
        
        # Test getting metadata from plugin class
        metadata = get_plugin_metadata(TestTransform)
        assert metadata is not None
        assert metadata['name'] == "TestTransform"
        assert metadata['type'] == "transform"
        
        # Test non-plugin class
        class NonPlugin:
            pass
        
        metadata = get_plugin_metadata(NonPlugin)
        assert metadata is None
    
    def test_is_plugin_class(self):
        """Test is_plugin_class utility function."""
        @plugin(type="kernel", name="TestKernel")
        class TestKernel:
            pass
        
        class NonPlugin:
            pass
        
        assert is_plugin_class(TestKernel) is True
        assert is_plugin_class(NonPlugin) is False
    
    def test_list_plugin_classes(self):
        """Test list_plugin_classes utility function."""
        # Create a mock module with some classes
        class MockModule:
            pass
        
        # Add plugin classes
        @plugin(type="transform", name="Transform1", stage="cleanup")
        class Transform1:
            pass
        
        @plugin(type="kernel", name="Kernel1")
        class Kernel1:
            pass
        
        class NonPlugin:
            pass
        
        # Add to mock module
        MockModule.Transform1 = Transform1
        MockModule.Kernel1 = Kernel1
        MockModule.NonPlugin = NonPlugin
        MockModule.some_function = lambda: None
        
        # Test listing
        plugin_classes = list_plugin_classes(MockModule)
        
        assert len(plugin_classes) == 2
        assert Transform1 in plugin_classes
        assert Kernel1 in plugin_classes
        assert NonPlugin not in plugin_classes


class TestFrameworkIntegration:
    def setup_method(self):
        reset_registry()
    
    def teardown_method(self):
        reset_registry()
    
    def test_framework_specification(self):
        """Test framework specification in decorators."""
        @plugin(type="transform", name="QONNXTransform", stage="cleanup", framework="qonnx")
        class QONNXTransform:
            pass
        
        @plugin(type="transform", name="FINNTransform", stage="streamlining", framework="finn")
        class FINNTransform:
            pass
        
        registry = get_registry()
        
        # Test framework indexing
        assert "QONNXTransform" in registry.framework_transforms["qonnx"]
        assert "FINNTransform" in registry.framework_transforms["finn"]
        
        # Test metadata
        qonnx_metadata = registry.get_plugin_metadata("QONNXTransform")
        assert qonnx_metadata['framework'] == "qonnx"
        
        finn_metadata = registry.get_plugin_metadata("FINNTransform")
        assert finn_metadata['framework'] == "finn"