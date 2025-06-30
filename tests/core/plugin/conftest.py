"""Shared fixtures for plugin tests."""

import pytest
from typing import Dict, Type, Any


@pytest.fixture(scope="session", autouse=True)
def ensure_plugins_imported():
    """Ensure all plugins are imported at test session start."""
    # Import all transforms
    import brainsmith.transforms.topology_opt.expand_norms
    import brainsmith.transforms.kernel_opt.set_pumped_compute
    import brainsmith.transforms.graph_cleanup.remove_identity
    
    # Import all kernels
    import brainsmith.kernels.layernorm.layernorm
    import brainsmith.kernels.matmul.matmul
    import brainsmith.kernels.softmax.hwsoftmax
    import brainsmith.kernels.crop.crop
    import brainsmith.kernels.shuffle.shuffle
    
    # Import all backends
    import brainsmith.kernels.layernorm.layernorm_hls
    import brainsmith.kernels.layernorm.layernorm_rtl
    import brainsmith.kernels.matmul.matmul_hls
    import brainsmith.kernels.matmul.matmul_rtl
    import brainsmith.kernels.softmax.hwsoftmax_hls
    import brainsmith.kernels.crop.crop_hls
    import brainsmith.kernels.shuffle.shuffle_hls
    
    # Import kernel inference transforms
    import brainsmith.kernels.layernorm.infer_layernorm
    import brainsmith.kernels.softmax.infer_hwsoftmax
    import brainsmith.kernels.shuffle.infer_shuffle
    import brainsmith.kernels.crop.infer_crop_from_gather


@pytest.fixture
def clean_registry():
    """Provide the registry for each test."""
    from brainsmith.plugin.core import get_registry
    
    # Don't clear registry - plugins register at import time
    # and Python caches imports, so they won't re-register
    registry = get_registry()
    yield registry


@pytest.fixture
def sample_plugins():
    """Create sample plugin classes for testing."""
    from brainsmith.plugin.core import transform, kernel, backend
    
    # Clear registry first
    registry = get_registry()
    registry.clear()
    
    # Define test transform
    @transform(name="SampleTransform", stage="topology_opt", description="Test transform")
    class SampleTransform:
        def apply(self, model):
            return model, False
    
    # Define test kernel
    @kernel(name="SampleKernel", op_type="Sample", description="Test kernel")
    class SampleKernel:
        pass
    
    # Define test backend
    @backend(name="SampleBackend", kernel="SampleKernel", backend_type="hls", description="Test backend")
    class SampleBackend:
        pass
    
    # Define test kernel inference
    @transform(name="SampleInference", kernel="SampleKernel", stage=None, description="Test inference")
    class SampleInference:
        def apply(self, model):
            return model, False
    
    return {
        "transform": SampleTransform,
        "kernel": SampleKernel,
        "backend": SampleBackend,
        "inference": SampleInference
    }


@pytest.fixture
def expected_plugin_counts():
    """Expected minimum plugin counts for validation."""
    return {
        "transforms": 5,  # Minimum expected transforms
        "kernels": 4,     # LayerNorm, MatMul, Softmax, Crop
        "backends": 6,    # Various HLS/RTL implementations
    }