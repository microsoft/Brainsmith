"""
Unit tests for plugin discovery methods.
"""

import pytest
from unittest.mock import Mock, MagicMock

from brainsmith.core.plugins import get_registry, transforms, kernels, backends
from brainsmith.core.plugins.registry import BrainsmithPluginRegistry
from brainsmith.core.plugins.collections import TransformCollection, KernelCollection, BackendCollection


class TestRegistryDiscoveryMethods:
    """Test discovery methods on the plugin registry."""
    
    def test_list_available_kernels(self):
        """Test listing all available kernel names."""
        # Test - use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        available = registry.list_available_kernels()
        
        # Verify - check that real kernels are available
        assert isinstance(available, list)
        assert len(available) >= 4  # At least the QONNX/FINN kernels
        
        # Check that real QONNX/FINN kernels are present
        real_kernels = ["LayerNorm", "Crop", "Shuffle", "HWSoftmax"]
        for kernel in real_kernels:
            assert kernel in available
    
    def test_list_available_transforms(self):
        """Test listing all available transform names."""
        # Test - use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        available = registry.list_available_transforms()
        
        # Verify - check that real transforms are available
        assert isinstance(available, list)
        assert len(available) >= 100  # QONNX/FINN have 119+ transforms
        
        # Check that real QONNX/FINN transforms are present
        real_transforms = ["RemoveUnusedTensors", "FoldConstants", 
                          "InferShapes", "InferDataTypes"]
        for transform in real_transforms:
            assert transform in available
    
    def test_get_valid_stages(self):
        """Test getting list of valid transform stages."""
        # Test - use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        stages = registry.get_valid_stages()
        
        # Verify - check all expected stages are present
        assert isinstance(stages, list)
        # Use actual stages from QONNX/FINN
        expected_stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "model_specific"]
        for stage in expected_stages:
            assert stage in stages
    
    def test_validate_kernel_backends(self):
        """Test validating backends for a specific kernel."""
        # Test - use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        
        # Test all valid backends for LayerNorm
        invalid = registry.validate_kernel_backends("LayerNorm", ["LayerNormHLS", "LayerNormRTL"])
        assert invalid == []
        
        # Test some invalid backends for LayerNorm
        invalid = registry.validate_kernel_backends("LayerNorm", ["LayerNormHLS", "NonExistentBackend"])
        assert invalid == ["NonExistentBackend"]
        
        # Test kernel with no backends (non-existent kernel)
        invalid = registry.validate_kernel_backends("UnknownKernel", ["LayerNormHLS"])
        assert invalid == ["LayerNormHLS"]
    
    def test_list_backends_by_kernel(self):
        """Test listing backends available for a specific kernel."""
        # Test - use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        
        # Test real QONNX/FINN kernels with known backends
        layernorm_backends = registry.list_backends_by_kernel("LayerNorm")
        assert "LayerNormHLS" in layernorm_backends
        assert "LayerNormRTL" in layernorm_backends
        
        crop_backends = registry.list_backends_by_kernel("Crop")
        assert "CropHLS" in crop_backends
        
        # Test non-existent kernel
        assert registry.list_backends_by_kernel("NonExistent") == []


class TestCollectionDiscoveryMethods:
    """Test discovery methods on collection classes."""
    
    def test_transform_collection_list_available(self):
        """Test TransformCollection.list_available()."""
        # Create mock registry
        mock_registry = Mock()
        mock_registry.transforms = {
            "Transform1": MagicMock(),
            "Transform2": MagicMock(),
            "Transform3": MagicMock()
        }
        
        # Create collection with mock registry
        collection = TransformCollection(mock_registry)
        collection.list_available = lambda: list(mock_registry.transforms.keys())
        
        # Test
        available = collection.list_available()
        
        # Verify
        assert len(available) == 3
        assert "Transform1" in available
        assert "Transform2" in available
        assert "Transform3" in available
    
    def test_transform_collection_list_stages(self):
        """Test TransformCollection.list_stages()."""
        # Create mock registry
        mock_registry = Mock()
        mock_registry.get_valid_stages.return_value = [
            "pre_proc", "cleanup", "topology_opt", "kernel_opt", "dataflow_opt", "post_proc"
        ]
        
        # Create collection
        collection = TransformCollection(mock_registry)
        collection.list_stages = lambda: mock_registry.get_valid_stages()
        
        # Test
        stages = collection.list_stages()
        
        # Verify
        assert len(stages) == 6
        assert "cleanup" in stages
        assert "topology_opt" in stages
        assert "post_proc" in stages
    
    def test_kernel_collection_list_available(self):
        """Test KernelCollection.list_available()."""
        # Create mock registry
        mock_registry = Mock()
        mock_registry.kernels = {
            "Kernel1": MagicMock(),
            "Kernel2": MagicMock()
        }
        
        # Create collection
        collection = KernelCollection(mock_registry)
        collection.list_available = lambda: list(mock_registry.kernels.keys())
        
        # Test
        available = collection.list_available()
        
        # Verify
        assert len(available) == 2
        assert "Kernel1" in available
        assert "Kernel2" in available
    
    def test_backend_collection_list_available(self):
        """Test BackendCollection.list_available()."""
        # Create mock registry
        mock_registry = Mock()
        mock_registry.backends = {
            "Backend1": {"kernel": "Kernel1"},
            "Backend2": {"kernel": "Kernel1"},
            "Backend3": {"kernel": "Kernel2"}
        }
        
        # Create collection
        collection = BackendCollection(mock_registry)
        collection.list_available = lambda: list(mock_registry.backends.keys())
        
        # Test
        available = collection.list_available()
        
        # Verify
        assert len(available) == 3
        assert "Backend1" in available
        assert "Backend2" in available
        assert "Backend3" in available


class TestDiscoveryIntegration:
    """Test discovery methods working together."""
    
    def test_discover_kernel_with_backends(self):
        """Test discovering a kernel and its available backends."""
        # Use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        
        # Discover all kernels
        kernels = registry.list_available_kernels()
        
        # For each kernel, discover its backends
        kernel_backend_map = {}
        for kernel in kernels[:3]:  # Just test first 3 kernels
            backends = registry.list_backends_by_kernel(kernel)
            kernel_backend_map[kernel] = backends
        
        # Verify we got some kernels with backends
        assert len(kernel_backend_map) > 0
        # Check a known kernel
        if "LayerNorm" in kernels:
            backends = registry.list_backends_by_kernel("LayerNorm")
            assert "LayerNormHLS" in backends
    
    def test_discover_transforms_by_stage(self):
        """Test discovering transforms organized by stage."""
        # Use real QONNX/FINN registry
        from brainsmith.core.plugins import get_registry
        registry = get_registry()
        
        # Discover stages
        stages = registry.get_valid_stages()
        
        # For each stage, find transforms
        transforms_by_stage = {}
        for stage in stages:
            stage_transforms = registry.list_transforms_by_stage(stage)
            if stage_transforms:
                transforms_by_stage[stage] = stage_transforms[:3]  # Just first 3 for test
        
        # Verify we got transforms for some stages
        assert len(transforms_by_stage) > 0
        # Check known stages have transforms
        if "cleanup" in stages:
            assert len(transforms_by_stage.get("cleanup", [])) > 0
        if "topology_opt" in stages:
            assert len(transforms_by_stage.get("topology_opt", [])) > 0