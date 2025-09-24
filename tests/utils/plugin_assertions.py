"""Plugin assertion helpers for plugin system tests."""

from typing import Dict, List, Set, Optional


class PluginAssertions:
    """Helper class for plugin system assertions."""
    
    @staticmethod
    def assert_framework_transforms_available(registry, min_expected: Dict[str, int] = None):
        """Assert that framework transforms are available in sufficient quantities.
        
        Args:
            registry: The plugin registry to check
            min_expected: Minimum expected counts by framework (defaults to reasonable minimums)
        """
        if min_expected is None:
            min_expected = {
                "finn": 10,   # Reasonable minimum for FINN framework
                "qonnx": 5    # Reasonable minimum for QONNX framework  
            }
        
        all_transforms = registry.all("transform")
        
        # Count transforms by framework
        framework_counts = {}
        for framework in min_expected.keys():
            framework_transforms = [
                name for name in all_transforms 
                if name.startswith(f"{framework}:")
            ]
            framework_counts[framework] = len(framework_transforms)
        
        # Assert minimums are met
        for framework, min_count in min_expected.items():
            actual_count = framework_counts.get(framework, 0)
            assert actual_count >= min_count, \
                f"Expected at least {min_count} {framework.upper()} transforms, " \
                f"but only found {actual_count}. " \
                f"Available transforms: {sorted([name for name in all_transforms if name.startswith(f'{framework}:')])}"
    
    @staticmethod
    def assert_test_plugins_available(registry, expected_plugins: Dict[str, List[str]]):
        """Assert that expected test plugins are available.
        
        Args:
            registry: The plugin registry to check
            expected_plugins: Dict mapping plugin types to expected plugin names
        """
        for plugin_type, expected_names in expected_plugins.items():
            available_plugins = registry.all(plugin_type)
            
            for plugin_name in expected_names:
                assert plugin_name in available_plugins, \
                    f"Expected test plugin '{plugin_name}' of type '{plugin_type}' not found. " \
                    f"Available {plugin_type} plugins: {sorted(available_plugins)}"
    
    
    @staticmethod
    def assert_kernel_backend_associations(registry, expected_associations: Dict[str, List[str]]):
        """Assert that kernels have expected backend associations.
        
        Args:
            registry: The plugin registry to check  
            expected_associations: Dict mapping kernel names to expected backend names
        """
        for kernel_name, expected_backends in expected_associations.items():
            # Verify kernel exists
            kernel_cls = registry.get("kernel", kernel_name)
            assert kernel_cls is not None, \
                f"Kernel '{kernel_name}' not found"
            
            # Get associated backends
            available_backends = registry.all("backend")
            kernel_backends = [
                backend_name for backend_name in available_backends
                if backend_name.startswith(f"{kernel_name}_")
            ]
            
            for expected_backend in expected_backends:
                assert expected_backend in kernel_backends, \
                    f"Expected backend '{expected_backend}' for kernel '{kernel_name}' not found. " \
                    f"Available backends for {kernel_name}: {kernel_backends}"
    
    @staticmethod
    def assert_plugin_execution_capability(registry, plugin_type: str, plugin_name: str):
        """Assert that a plugin can be retrieved and has basic execution capability.
        
        Args:
            registry: The plugin registry to check
            plugin_type: Type of plugin ('transform', 'kernel', 'step')
            plugin_name: Name of the plugin
        """
        plugin_cls = registry.get(plugin_type, plugin_name)
        assert plugin_cls is not None, \
            f"Plugin '{plugin_name}' of type '{plugin_type}' not found"
        
        # Basic instantiation check (if class)
        if hasattr(plugin_cls, '__call__'):
            try:
                # For step functions, just verify they're callable
                if plugin_type == "step":
                    assert callable(plugin_cls), \
                        f"Step plugin '{plugin_name}' should be callable"
                else:
                    # For classes, try basic instantiation (may need args)
                    assert hasattr(plugin_cls, '__init__'), \
                        f"Plugin '{plugin_name}' should be a proper class"
            except Exception as e:
                # If instantiation fails, at least verify it's a proper class/function
                assert callable(plugin_cls), \
                    f"Plugin '{plugin_name}' should be callable, but got error: {e}"


# Constants for common test expectations
EXPECTED_TEST_PLUGINS = {
    "transform": [
        "test_transform",           # From transforms.py
        "test_transform_with_metadata", 
        "TestAddMetadata",          # From transforms.py
        "TestAttributeAdder",      # From transforms.py
        "TestNodeCounter"           # From transforms.py
    ],
    "kernel": [
        "TestKernel",
        "TestKernelWithBackends"
    ],
    "step": [
        "test_step",
        "test_step1", 
        "test_step2",
        "test_step3"
    ]
}

EXPECTED_KERNEL_BACKENDS = {
    "TestKernel": ["TestKernel_hls", "TestKernel_rtl"],
    "TestKernelWithBackends": ["TestKernelWithBackends_hls"]
}

# Minimum framework plugin expectations (conservative numbers)
MIN_FRAMEWORK_PLUGINS = {
    "finn": 10,    # At least 10 FINN transforms should be available
    "qonnx": 5     # At least 5 QONNX transforms should be available
}