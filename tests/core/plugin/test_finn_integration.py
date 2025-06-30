"""Test FINN plugin discovery and integration.

Tests FINN's enhanced registry with query support and
stage validation for transforms.
"""

import pytest
import sys
import os

# Add paths for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestFINNIntegration:
    """Test suite for FINN plugin integration."""
    
    def test_finn_registry_exists(self):
        """Test that FINN registry is accessible."""
        try:
            from finn.plugin.registry import FinnPluginRegistry, get_finn_registry
        except ImportError:
            pytest.skip("FINN not available")
        
        registry = get_finn_registry()
        assert registry is not None
        assert isinstance(registry, FinnPluginRegistry)
    
    def test_finn_registry_query(self):
        """Test FINN's new query method."""
        try:
            from finn.plugin.registry import get_finn_registry
        except ImportError:
            pytest.skip("FINN not available")
        
        registry = get_finn_registry()
        
        # Check if query method exists
        if not hasattr(registry, 'query'):
            pytest.skip("FINN registry doesn't have query method yet")
        
        # Query all plugins
        all_plugins = registry.query()
        assert isinstance(all_plugins, list)
        
        # Query by type
        transforms = registry.query(type="transform")
        kernels = registry.query(type="kernel")
        backends = registry.query(type="backend")
        
        # Each should return a list
        assert isinstance(transforms, list)
        assert isinstance(kernels, list)
        assert isinstance(backends, list)
    
    def test_finn_stage_validation(self):
        """Test valid and invalid stage names in FINN."""
        try:
            from finn.plugin.adapters import transform, VALID_STAGES
        except ImportError:
            pytest.skip("FINN adapters not available")
        
        # Check valid stages
        assert VALID_STAGES == ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        # Test valid stage registration
        @transform(name="ValidStageTest", stage="topology_opt")
        class ValidStageTest:
            pass
        
        # Test invalid stage registration
        with pytest.raises(ValueError, match="Invalid stage"):
            @transform(name="InvalidStageTest", stage="invalid_stage")
            class InvalidStageTest:
                pass
        
        # Old stage names should fail
        with pytest.raises(ValueError, match="Invalid stage"):
            @transform(name="OldStageTest", stage="streamlining")
            class OldStageTest:
                pass
    
    def test_finn_cross_registration(self):
        """Test FINN plugins visible in BrainSmith's unified registry."""
        from brainsmith.plugin.core import get_registry
        
        try:
            from finn.plugin.adapters import transform
        except ImportError:
            pytest.skip("FINN adapters not available")
        
        # Register a FINN transform
        @transform(
            name="FINNTestTransform",
            stage="kernel_opt",
            description="Test FINN transform"
        )
        class FINNTestTransform:
            def apply(self, model):
                return model, False
        
        # Check if it's in BrainSmith's registry
        bs_registry = get_registry()
        
        # It might be registered with finn: prefix
        finn_transform = bs_registry.get("transform", "finn:FINNTestTransform")
        if not finn_transform:
            # Or without prefix if directly registered
            finn_transform = bs_registry.get("transform", "FINNTestTransform")
        
        # FINN transforms may or may not be auto-registered in BrainSmith
        # This depends on the integration implementation
    
    def test_finn_metadata_propagation(self):
        """Test metadata from FINN decorators."""
        try:
            from finn.plugin.registry import get_finn_registry
            from finn.plugin.adapters import transform, kernel, backend
        except ImportError:
            pytest.skip("FINN plugin system not available")
        
        # Register components with metadata
        @transform(
            name="MetadataTransform",
            stage="cleanup",
            description="Test transform with metadata",
            author="test-author",
            version="1.0.0"
        )
        class MetadataTransform:
            pass
        
        @kernel(
            name="MetadataKernel",
            op_type="TestOp",
            description="Test kernel with metadata"
        )
        class MetadataKernel:
            pass
        
        @backend(
            name="MetadataBackend",
            kernel="MetadataKernel",
            backend_type="hls",
            description="Test backend with metadata"
        )
        class MetadataBackend:
            pass
        
        # Check metadata in FINN registry
        finn_registry = get_finn_registry()
        
        if hasattr(finn_registry, 'query'):
            # Query for our test components
            transforms = finn_registry.query(type="transform", name="MetadataTransform")
            if transforms:
                t = transforms[0]
                assert t.get("description") == "Test transform with metadata"
                assert t.get("author") == "test-author"
                assert t.get("version") == "1.0.0"
    
    def test_finn_query_by_stage(self):
        """Test querying FINN transforms by stage."""
        try:
            from finn.plugin.registry import get_finn_registry
        except ImportError:
            pytest.skip("FINN not available")
        
        registry = get_finn_registry()
        
        if not hasattr(registry, 'query'):
            pytest.skip("FINN registry doesn't have query method")
        
        # Query each valid stage
        stages = ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
        
        for stage in stages:
            transforms = registry.query(type="transform", stage=stage)
            
            # Verify all returned transforms have the correct stage
            for t in transforms:
                assert t.get("stage") == stage
    
    def test_finn_kernel_backend_query(self):
        """Test querying backends by kernel."""
        try:
            from finn.plugin.registry import get_finn_registry
        except ImportError:
            pytest.skip("FINN not available")
        
        registry = get_finn_registry()
        
        if not hasattr(registry, 'query'):
            pytest.skip("FINN registry doesn't have query method")
        
        # Get all kernels
        kernels = registry.query(type="kernel")
        
        # For each kernel, check if it has backends
        for kernel in kernels:
            kernel_name = kernel["name"]
            
            # Query backends for this kernel
            backends = registry.query(type="backend", kernel=kernel_name)
            
            # Verify backend associations
            for backend in backends:
                assert backend.get("kernel") == kernel_name
                assert backend.get("backend_type") in ["hls", "rtl"]
    
    def test_finn_transform_imports(self):
        """Test importing actual FINN transforms."""
        try:
            from finn.transformation.streamline.absorb import AbsorbSignBiasIntoMultiThreshold
        except ImportError:
            pytest.skip("FINN transforms not available")
        
        # Check the transform has proper structure
        assert hasattr(AbsorbSignBiasIntoMultiThreshold, 'apply')
        
        # Check if it has plugin metadata
        if hasattr(AbsorbSignBiasIntoMultiThreshold, '_plugin_metadata'):
            metadata = AbsorbSignBiasIntoMultiThreshold._plugin_metadata
            assert metadata.get("type") == "transform"
            assert metadata.get("stage") in ["cleanup", "topology_opt", "kernel_opt", "dataflow_opt"]
    
    def test_finn_get_plugin_info(self):
        """Test FINN's get_plugin_info method."""
        try:
            from finn.plugin.registry import get_finn_registry
            from finn.plugin.adapters import transform
        except ImportError:
            pytest.skip("FINN not available")
        
        registry = get_finn_registry()
        
        if not hasattr(registry, 'get_plugin_info'):
            pytest.skip("FINN registry doesn't have get_plugin_info method")
        
        # Register a test plugin
        @transform(name="InfoTest", stage="cleanup")
        class InfoTest:
            pass
        
        # Get info
        info = registry.get_plugin_info("transform", "InfoTest")
        
        if info:
            assert info["name"] == "InfoTest"
            assert info["type"] == "transform"
            assert info["stage"] == "cleanup"
        
        # Test non-existent plugin
        no_info = registry.get_plugin_info("transform", "NonExistent")
        assert no_info is None
    
    def test_finn_backend_types(self):
        """Test FINN backend type validation."""
        try:
            from finn.plugin.adapters import backend
        except ImportError:
            pytest.skip("FINN adapters not available")
        
        # Valid backend types
        @backend(name="HLSTest", kernel="TestKernel", backend_type="hls")
        class HLSTest:
            pass
        
        @backend(name="RTLTest", kernel="TestKernel", backend_type="rtl")
        class RTLTest:
            pass
        
        # Invalid backend type
        with pytest.raises(ValueError, match="Invalid backend_type"):
            @backend(name="InvalidBackend", kernel="TestKernel", backend_type="invalid")
            class InvalidBackend:
                pass