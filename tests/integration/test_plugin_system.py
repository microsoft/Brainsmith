"""Integration tests for the plugin system."""

import pytest

from brainsmith.core.plugins import get_registry
from brainsmith.core.plugins.registry import list_backends_by_kernel, get_default_backend
from tests.fixtures.model_utils import create_simple_model
from tests.utils.plugin_assertions import (
    PluginAssertions,
    EXPECTED_TEST_PLUGINS,
    EXPECTED_KERNEL_BACKENDS,
    MIN_FRAMEWORK_PLUGINS
)


class TestTransformPlugins:
    """Test suite for transform plugins."""
    
    def test_register_transform_plugin(self):
        """Test registering and retrieving a transform plugin."""
        registry = get_registry()
        # Use helper to verify all expected test plugins are available
        PluginAssertions.assert_test_plugins_available(
            registry,
            EXPECTED_TEST_PLUGINS
        )
        
        # Verify specific transform execution capability
        plugin_cls = registry.get("transform", "test_transform")
        assert plugin_cls is not None
        assert plugin_cls.__name__ == "TestTransformPlugin"
        
        # Create instance and test execution
        transform = plugin_cls()
        model = create_simple_model()
        
        # Apply transform
        modified_model = transform.apply(model)
        
        # Verify model was modified
        assert len(modified_model.graph.node) > 0
        node = modified_model.graph.node[0]
        
        # Check that our custom attribute was added
        attr_names = [attr.name for attr in node.attribute]
        assert "test_transform_applied" in attr_names
        
        # Find the attribute and verify its value
        for attr in node.attribute:
            if attr.name == "test_transform_applied":
                assert attr.i == 1
                break
    
    def test_framework_transform_namespacing(self):
        """Test accessing transforms with framework namespacing."""
        registry = get_registry()
        # First ensure plugins are loaded
        all_transforms = registry.all("transform")
        
        # Test FINN transform access - should always be available
        # Try with explicit prefix
        finn_transform = registry.get("transform", "finn:Streamline")
        assert finn_transform is not None
        
        # Try without prefix (should auto-resolve)
        same_transform = registry.get("transform", "Streamline")
        assert same_transform == finn_transform
        
        # Test QONNX transform access - should always be available
        # Try with explicit prefix
        qonnx_transform = registry.get("transform", "qonnx:InferDataTypes")
        assert qonnx_transform is not None
        
        # Try without prefix (should auto-resolve)
        same_transform = registry.get("transform", "InferDataTypes") 
        assert same_transform == qonnx_transform
        
        # Use helper to verify framework transform availability with flexible thresholds
        PluginAssertions.assert_framework_transforms_available(
            registry, 
            MIN_FRAMEWORK_PLUGINS
        )
    
    def test_transform_metadata(self):
        """Test transform with metadata."""
        registry = get_registry()
        # Use helper to verify plugin metadata structure
        PluginAssertions.assert_plugin_execution_capability(
            registry,
            "transform",
            "test_transform_with_metadata"
        )
        
        # Query by metadata
        transforms = registry.find("transform", test_metadata="value")
        assert len(transforms) > 0
        
        # Verify the plugin is in the metadata query results
        plugin_cls = registry.get("transform", "test_transform_with_metadata")
        assert plugin_cls in transforms


class TestKernelPlugins:
    """Test suite for kernel plugins."""
    
    def test_register_kernel_with_backends(self):
        """Test registering kernels with multiple backends."""
        registry = get_registry()
        # Use helper to verify kernel-backend associations
        PluginAssertions.assert_kernel_backend_associations(
            registry,
            EXPECTED_KERNEL_BACKENDS
        )
        
        # Verify kernel execution capability
        PluginAssertions.assert_plugin_execution_capability(
            registry,
            "kernel",
            "TestKernel"
        )
        
        # Verify kernel class details
        kernel_cls = registry.get("kernel", "TestKernel")
        assert kernel_cls.__name__ == "TestKernelPlugin"
        
        # Get backends
        hls_backend = registry.get("backend", "TestKernel_hls")
        rtl_backend = registry.get("backend", "TestKernel_rtl")
        
        assert hls_backend is not None
        assert rtl_backend is not None
        
        # Query backends by kernel name
        backends = list_backends_by_kernel("TestKernel")
        assert len(backends) == 2
        assert "TestKernel_hls" in backends
        assert "TestKernel_rtl" in backends
        
        # Verify backend metadata
        hls_info = registry.find("backend", kernel="TestKernel", language="hls")
        assert len(hls_info) == 1
        assert hls_info[0] == hls_backend
        
        rtl_info = registry.find("backend", kernel="TestKernel", language="rtl")
        assert len(rtl_info) == 1
        assert rtl_info[0] == rtl_backend
    
    def test_kernel_inference_decorator(self):
        """Test kernel inference decorator."""
        registry = get_registry()
        # Get inference transform
        inference_cls = registry.get("transform", "InferTestKernel")
        assert inference_cls is not None
        
        # Verify kernel association in metadata
        inference_transforms = registry.find("transform", kernel_inference=True)
        assert len(inference_transforms) > 0
        assert inference_cls in inference_transforms
        
        # Also verify the kernel parameter was passed
        _, metadata = next((item for item in registry._plugins["transform"].items() if item[1][0] == inference_cls), (None, None))
        assert metadata is not None
        assert metadata[1].get("kernel") == "TestKernel"
        
        # Test execution
        model = create_simple_model()
        inference = inference_cls()
        result = inference.apply(model)
        
        # Verify inference was applied
        if result.graph.node:
            node = result.graph.node[0]
            attr_names = [attr.name for attr in node.attribute]
            assert "kernel_inferred" in attr_names
    
    def test_default_backend_selection(self):
        """Test default backend selection."""
        registry = get_registry()
        # Get default backend (should be first registered)
        default_backend = get_default_backend("TestKernel")
        assert default_backend == "TestKernel_hls"  # HLS was registered first
        
        # Test with kernel that has only one backend
        kernel_cls = registry.get("kernel", "TestKernelWithBackends")
        assert kernel_cls is not None
        
        default = get_default_backend("TestKernelWithBackends")
        assert default == "TestKernelWithBackends_hls"


class TestStepPlugins:
    """Test suite for step plugins."""
    
    def test_register_step_plugin(self):
        """Test registering and executing step plugins."""
        registry = get_registry()
        # Get step
        step_func = registry.get("step", "test_step")
        assert step_func is not None
        assert callable(step_func)
        
        # Execute step
        blueprint = {"name": "test", "clock_ns": 5.0}
        context = {}
        
        step_func(blueprint, context)
        
        # Verify execution
        assert "executed_steps" in context
        assert "test_step" in context["executed_steps"]
    
    # Removed test_step_with_kernel_backends - tested mock step execution


# TestPluginDiscovery has been removed - tested internal implementation details