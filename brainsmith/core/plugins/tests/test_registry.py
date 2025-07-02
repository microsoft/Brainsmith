"""
Test Core Registry Functionality

Tests for the high-performance registry with optimized data structures.
"""

import pytest
from unittest.mock import Mock

from brainsmith.core.plugins.registry import BrainsmithPluginRegistry, get_registry, reset_registry


class TestBrainsmithPluginRegistry:
    def setup_method(self):
        self.registry = BrainsmithPluginRegistry()
    
    def test_transform_registration_and_indexing(self):
        """Test transform registration with automatic stage and framework indexing."""
        # Create mock transform class
        class TestTransform:
            def apply(self, model):
                return model, False
        
        # Register transform
        self.registry.register_transform(
            "TestTransform", 
            TestTransform, 
            stage="cleanup", 
            framework="brainsmith"
        )
        
        # Test main registry
        assert "TestTransform" in self.registry.transforms
        assert self.registry.transforms["TestTransform"] == TestTransform
        
        # Test stage indexing
        assert "cleanup" in self.registry.transforms_by_stage
        assert "TestTransform" in self.registry.transforms_by_stage["cleanup"]
        assert self.registry.transforms_by_stage["cleanup"]["TestTransform"] == TestTransform
        
        # Test framework indexing
        assert "brainsmith" in self.registry.framework_transforms
        assert "TestTransform" in self.registry.framework_transforms["brainsmith"]
        assert self.registry.framework_transforms["brainsmith"]["TestTransform"] == TestTransform
        
        # Test metadata storage
        metadata = self.registry.get_plugin_metadata("TestTransform")
        assert metadata['type'] == 'transform'
        assert metadata['stage'] == 'cleanup'
        assert metadata['framework'] == 'brainsmith'
    
    def test_kernel_registration(self):
        """Test kernel registration."""
        class TestKernel:
            pass
        
        self.registry.register_kernel("TestKernel", TestKernel, op_type="TestOp")
        
        # Test main registry
        assert "TestKernel" in self.registry.kernels
        assert self.registry.kernels["TestKernel"] == TestKernel
        
        # Test metadata
        metadata = self.registry.get_plugin_metadata("TestKernel")
        assert metadata['type'] == 'kernel'
        assert metadata['op_type'] == 'TestOp'
    
    def test_backend_registration_and_indexing(self):
        """Test backend registration with automatic kernel indexing."""
        class TestKernelHLS:
            pass
        
        # Register backend
        self.registry.register_backend(
            "TestKernelHLS", 
            TestKernelHLS, 
            kernel="TestKernel", 
            backend_type="hls"
        )
        
        # Test main registry
        backend_key = "TestKernel_hls"
        assert backend_key in self.registry.backends
        assert self.registry.backends[backend_key] == TestKernelHLS
        
        # Test kernel indexing
        assert "TestKernel" in self.registry.backends_by_kernel
        assert "hls" in self.registry.backends_by_kernel["TestKernel"]
        assert self.registry.backends_by_kernel["TestKernel"]["hls"] == TestKernelHLS
        
        # Test metadata
        metadata = self.registry.get_plugin_metadata(backend_key)
        assert metadata['type'] == 'backend'
        assert metadata['kernel'] == 'TestKernel'
        assert metadata['backend_type'] == 'hls'
    
    def test_fast_lookups(self):
        """Test optimized lookup methods."""
        # Setup test data
        class CleanupTransform:
            pass
        class OptTransform:
            pass
        class TestKernel:
            pass
        class TestKernelHLS:
            pass
        
        self.registry.register_transform("CleanupTransform", CleanupTransform, stage="cleanup", framework="brainsmith")
        self.registry.register_transform("OptTransform", OptTransform, stage="topology_opt", framework="qonnx")
        self.registry.register_kernel("TestKernel", TestKernel)
        self.registry.register_backend("TestKernelHLS", TestKernelHLS, kernel="TestKernel", backend_type="hls")
        
        # Test transform lookups
        assert self.registry.get_transform("CleanupTransform") == CleanupTransform
        assert self.registry.get_transform("CleanupTransform", stage="cleanup") == CleanupTransform
        assert self.registry.get_transform("OptTransform", framework="qonnx") == OptTransform
        assert self.registry.get_transform("OptTransform", stage="topology_opt") == OptTransform
        
        # Test kernel lookup
        assert self.registry.get_kernel("TestKernel") == TestKernel
        
        # Test backend lookup
        assert self.registry.get_backend("TestKernel", "hls") == TestKernelHLS
    
    def test_blueprint_optimization_queries(self):
        """Test blueprint optimization methods."""
        # Setup test data
        class CleanupTransform:
            pass
        class OptTransform:
            pass
        class TestKernel:
            pass
        class TestKernelHLS:
            pass
        class TestKernelRTL:
            pass
        
        self.registry.register_transform("CleanupTransform", CleanupTransform, stage="cleanup")
        self.registry.register_transform("OptTransform", OptTransform, stage="topology_opt")
        self.registry.register_kernel("TestKernel", TestKernel)
        self.registry.register_backend("TestKernelHLS", TestKernelHLS, kernel="TestKernel", backend_type="hls")
        self.registry.register_backend("TestKernelRTL", TestKernelRTL, kernel="TestKernel", backend_type="rtl")
        
        # Test stage queries
        cleanup_transforms = self.registry.list_transforms_by_stage("cleanup")
        assert "CleanupTransform" in cleanup_transforms
        assert "OptTransform" not in cleanup_transforms
        
        opt_transforms = self.registry.list_transforms_by_stage("topology_opt")
        assert "OptTransform" in opt_transforms
        assert "CleanupTransform" not in opt_transforms
        
        # Test backend queries
        backends = self.registry.list_backends_by_kernel("TestKernel")
        assert "hls" in backends
        assert "rtl" in backends
        assert len(backends) == 2
        
        # Test framework queries
        brainsmith_transforms = self.registry.get_framework_transforms("brainsmith")
        assert "CleanupTransform" in brainsmith_transforms
        assert "OptTransform" in brainsmith_transforms
    
    def test_stats(self):
        """Test registry statistics."""
        # Register some test plugins
        class TestTransform:
            pass
        class TestKernel:
            pass
        class TestBackend:
            pass
        
        self.registry.register_transform("TestTransform", TestTransform, stage="cleanup")
        self.registry.register_kernel("TestKernel", TestKernel)
        self.registry.register_backend("TestBackend", TestBackend, kernel="TestKernel", backend_type="hls")
        
        stats = self.registry.get_stats()
        
        assert stats['total_plugins'] == 3
        assert stats['transforms'] == 1
        assert stats['kernels'] == 1
        assert stats['backends'] == 1
        assert 'cleanup' in stats['stages']
        assert 'brainsmith' in stats['frameworks']
        assert stats['indexed_backends'] == 1
    
    def test_create_subset(self):
        """Test subset registry creation for blueprint optimization."""
        # Setup full registry
        class Transform1:
            pass
        class Transform2:
            pass
        class Kernel1:
            pass
        class Backend1:
            pass
        
        self.registry.register_transform("Transform1", Transform1, stage="cleanup")
        self.registry.register_transform("Transform2", Transform2, stage="topology_opt")
        self.registry.register_kernel("Kernel1", Kernel1)
        self.registry.register_backend("Backend1", Backend1, kernel="Kernel1", backend_type="hls")
        
        # Create subset with only some plugins
        requirements = {
            'transforms': ['Transform1'],
            'kernels': ['Kernel1'],
            'backends': ['Kernel1_hls']
        }
        
        subset = self.registry.create_subset(requirements)
        
        # Test subset contains only required plugins
        assert len(subset.transforms) == 1
        assert "Transform1" in subset.transforms
        assert "Transform2" not in subset.transforms
        
        assert len(subset.kernels) == 1
        assert "Kernel1" in subset.kernels
        
        assert len(subset.backends) == 1
        assert "Kernel1_hls" in subset.backends
    
    def test_clear(self):
        """Test registry clearing."""
        # Add some plugins
        class TestTransform:
            pass
        
        self.registry.register_transform("TestTransform", TestTransform, stage="cleanup")
        
        # Verify they exist
        assert len(self.registry.transforms) > 0
        assert len(self.registry.transforms_by_stage) > 0
        assert len(self.registry.plugin_metadata) > 0
        
        # Clear registry
        self.registry.clear()
        
        # Verify everything is cleared
        assert len(self.registry.transforms) == 0
        assert len(self.registry.kernels) == 0
        assert len(self.registry.backends) == 0
        assert len(self.registry.transforms_by_stage) == 0
        assert len(self.registry.backends_by_kernel) == 0
        assert len(self.registry.framework_transforms) == 0
        assert len(self.registry.plugin_metadata) == 0


class TestGlobalRegistry:
    def setup_method(self):
        reset_registry()
    
    def teardown_method(self):
        reset_registry()
    
    def test_global_registry_singleton(self):
        """Test global registry singleton pattern."""
        registry1 = get_registry()
        registry2 = get_registry()
        
        # Should be the same instance
        assert registry1 is registry2
        
        # Should be a BrainsmithPluginRegistry
        assert isinstance(registry1, BrainsmithPluginRegistry)
    
    def test_reset_registry(self):
        """Test registry reset functionality."""
        # Get initial registry and register something
        registry1 = get_registry()
        
        class TestTransform:
            pass
        
        registry1.register_transform("TestTransform", TestTransform)
        assert "TestTransform" in registry1.transforms
        
        # Reset registry
        reset_registry()
        
        # Get new registry - should be different instance and empty
        registry2 = get_registry()
        assert registry2 is not registry1
        assert len(registry2.transforms) == 0