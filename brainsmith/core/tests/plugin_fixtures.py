"""
Real Plugin Fixtures for Testing

This module provides pytest fixtures that use real plugins instead of mocks.
The fixtures ensure proper plugin isolation and cleanup between tests.
"""

import pytest
from brainsmith.core.plugins import get_registry, reset_plugin_system


@pytest.fixture
def real_test_registry():
    """
    Create a clean registry with all test plugins registered.
    
    This fixture replaces the mock_registry fixture and provides
    a real registry with comprehensive test plugins.
    """
    # Reset the plugin system to ensure clean state
    reset_plugin_system()
    
    # Force re-import of test plugins to trigger re-registration
    import sys
    module_name = 'brainsmith.core.tests.test_plugins'
    if module_name in sys.modules:
        del sys.modules[module_name]
    
    # Import test plugins to trigger registration
    import brainsmith.core.tests.test_plugins
    
    # Get the real registry with test plugins
    registry = get_registry()
    
    yield registry
    
    # Cleanup after test
    reset_plugin_system()


@pytest.fixture
def minimal_test_registry():
    """
    Create a registry with just a few plugins for simple tests.
    
    This is useful for tests that don't need the full plugin set
    and want faster execution.
    """
    reset_plugin_system()
    
    # Register only minimal plugins
    from brainsmith.core.plugins.decorators import transform, kernel, backend
    
    @kernel(name="MinimalKernel")
    class MinimalKernel:
        def compile(self, node):
            return {"type": "minimal", "node": node}
    
    @backend(name="MinimalBackend", kernel="MinimalKernel", language="hls")
    class MinimalBackend:
        def generate(self, kernel_instance):
            return "// Minimal backend"
    
    @transform(name="MinimalTransform", stage="cleanup")
    class MinimalTransform:
        def apply(self, model):
            return model, False
    
    registry = get_registry()
    yield registry
    reset_plugin_system()


@pytest.fixture
def error_test_registry():
    """
    Create a registry with plugins that cause errors for testing error scenarios.
    
    This fixture is used for testing error handling and validation.
    """
    reset_plugin_system()
    
    # Import test plugins including error ones
    from . import test_plugins
    
    registry = get_registry()
    yield registry
    reset_plugin_system()


@pytest.fixture
def isolated_registry():
    """
    Create a completely empty registry for testing missing plugin scenarios.
    
    This fixture provides a clean registry with no plugins for testing
    error conditions when plugins are missing.
    """
    reset_plugin_system()
    
    # Don't import any plugins - empty registry
    registry = get_registry()
    yield registry
    reset_plugin_system()


@pytest.fixture
def parser_with_real_registry(real_test_registry):
    """
    Create a parser with real registry instead of mock.
    
    This replaces the parser_with_mock_registry fixture.
    """
    from brainsmith.core.phase1.parser import BlueprintParser
    parser = BlueprintParser()
    parser.plugin_registry = real_test_registry
    return parser


@pytest.fixture
def validator_with_real_registry(real_test_registry):
    """
    Create a validator with real registry instead of mock.
    
    This replaces the validator_with_mock_registry fixture.
    """
    from brainsmith.core.phase1.validator import DesignSpaceValidator
    validator = DesignSpaceValidator()
    validator.plugin_registry = real_test_registry
    return validator


@pytest.fixture
def subset_test_registry():
    """
    Create a registry with specific plugins for subset testing.
    
    This fixture provides a controlled set of plugins for testing
    plugin loading optimization and subset creation.
    """
    reset_plugin_system()
    
    # Import specific test plugins
    from brainsmith.core.plugins.decorators import transform, kernel, backend
    
    # Register 2 kernels with 2 backends each
    @kernel(name="SubsetKernel1")
    class SubsetKernel1:
        def compile(self, node):
            return {"type": "subset1", "node": node}
    
    @kernel(name="SubsetKernel2")
    class SubsetKernel2:
        def compile(self, node):
            return {"type": "subset2", "node": node}
    
    @backend(name="SubsetBackend1HLS", kernel="SubsetKernel1", language="hls")
    class SubsetBackend1HLS:
        def generate(self, kernel_instance):
            return "// Subset 1 HLS"
    
    @backend(name="SubsetBackend1RTL", kernel="SubsetKernel1", language="rtl")
    class SubsetBackend1RTL:
        def generate(self, kernel_instance):
            return "// Subset 1 RTL"
    
    @backend(name="SubsetBackend2HLS", kernel="SubsetKernel2", language="hls")
    class SubsetBackend2HLS:
        def generate(self, kernel_instance):
            return "// Subset 2 HLS"
    
    @backend(name="SubsetBackend2RTL", kernel="SubsetKernel2", language="rtl")
    class SubsetBackend2RTL:
        def generate(self, kernel_instance):
            return "// Subset 2 RTL"
    
    # Register 2 transforms
    @transform(name="SubsetTransform1", stage="cleanup")
    class SubsetTransform1:
        def apply(self, model):
            return model, False
    
    @transform(name="SubsetTransform2", stage="topology_opt")
    class SubsetTransform2:
        def apply(self, model):
            return model, False
    
    registry = get_registry()
    yield registry
    reset_plugin_system()


@pytest.fixture
def performance_test_registry():
    """
    Create a registry with many plugins for performance testing.
    
    This fixture provides a large number of plugins to test
    performance characteristics and optimization benefits.
    """
    reset_plugin_system()
    
    from brainsmith.core.plugins.decorators import transform, kernel, backend
    
    # Register 20 kernels for performance testing
    for i in range(20):
        kernel_class = type(
            f"PerfKernel{i}",
            (object,),
            {"compile": lambda self, node: {"type": f"perf{i}", "node": node}}
        )
        kernel(name=f"PerfKernel{i}")(kernel_class)
        
        # 3 backends per kernel
        for lang in ["hls", "rtl", "dsp"]:
            backend_class = type(
                f"PerfBackend{i}{lang.upper()}",
                (object,),
                {"generate": lambda self, ki: f"// Perf {i} {lang}"}
            )
            backend(name=f"PerfBackend{i}{lang.upper()}", kernel=f"PerfKernel{i}", language=lang)(backend_class)
    
    # Register 30 transforms for performance testing
    stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"]
    for i in range(30):
        stage = stages[i % len(stages)]
        transform_class = type(
            f"PerfTransform{i}",
            (object,),
            {"apply": lambda self, model: (model, False)}
        )
        transform(name=f"PerfTransform{i}", stage=stage)(transform_class)
    
    registry = get_registry()
    yield registry
    reset_plugin_system()


# Helper functions for test data

def get_available_kernels(registry):
    """Get list of available kernel names from registry."""
    return list(registry.kernels.keys())


def get_available_transforms(registry):
    """Get list of available transform names from registry."""
    return list(registry.transforms.keys())


def get_available_backends(registry):
    """Get list of available backend names from registry."""
    return list(registry.backends.keys())


def get_backends_for_kernel(registry, kernel_name):
    """Get list of backend names for a specific kernel."""
    return registry.list_backends_by_kernel(kernel_name)


def get_transforms_for_stage(registry, stage):
    """Get list of transform names for a specific stage."""
    return registry.list_transforms_by_stage(stage)


def validate_test_plugin_setup(registry):
    """
    Validate that test plugins are properly registered.
    
    This function can be used in tests to ensure the registry
    is in the expected state.
    """
    # Check that we have the expected number of plugins
    kernels = get_available_kernels(registry)
    transforms = get_available_transforms(registry)
    backends = get_available_backends(registry)
    
    return {
        "kernels_count": len(kernels),
        "transforms_count": len(transforms),
        "backends_count": len(backends),
        "kernels": kernels,
        "transforms": transforms,
        "backends": backends
    }