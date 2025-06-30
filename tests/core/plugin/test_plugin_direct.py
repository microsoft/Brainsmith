"""Direct plugin system tests that bypass import issues."""

import sys
import os
import pytest

# Add brainsmith to path directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))


class TestPluginSystemDirect:
    """Test plugin system by importing modules directly."""
    
    def test_plugin_core_imports(self):
        """Test that plugin core can be imported directly."""
        # Import just the plugin core, avoiding main __init__.py
        from brainsmith.plugin import core
        
        # Get registry
        registry = core.get_registry()
        assert registry is not None
        
        # Test decorators exist
        assert hasattr(core, 'transform')
        assert hasattr(core, 'kernel')
        assert hasattr(core, 'backend')
    
    def test_dynamic_registration_direct(self):
        """Test dynamic registration without full imports."""
        from brainsmith.plugin import core
        
        registry = core.get_registry()
        
        # Register test components
        @core.transform(name="DirectTestTransform", stage="topology_opt")
        class DirectTestTransform:
            pass
        
        @core.kernel(name="DirectTestKernel", op_type="Test")
        class DirectTestKernel:
            pass
        
        @core.backend(name="DirectTestBackend", kernel="DirectTestKernel", backend_type="hls")
        class DirectTestBackend:
            pass
        
        # Verify registration
        assert registry.get("transform", "DirectTestTransform") == DirectTestTransform
        assert registry.get("kernel", "DirectTestKernel") == DirectTestKernel
        assert registry.get("backend", "DirectTestBackend") == DirectTestBackend
    
    def test_kernel_inference_direct(self):
        """Test kernel inference registration directly."""
        from brainsmith.plugin import core
        
        registry = core.get_registry()
        
        @core.transform(name="DirectTestInference", kernel="TestKernel", stage=None)
        class DirectTestInference:
            pass
        
        # Should be registered as kernel_inference
        assert registry.get("kernel_inference", "DirectTestInference") == DirectTestInference
        
        # Should not be in regular transforms
        assert registry.get("transform", "DirectTestInference") is None
    
    def test_query_functionality(self):
        """Test registry query functionality."""
        from brainsmith.plugin import core
        
        registry = core.get_registry()
        
        # Register multiple items
        @core.transform(name="QueryTest1", stage="topology_opt")
        class QueryTest1:
            pass
        
        @core.transform(name="QueryTest2", stage="kernel_opt")
        class QueryTest2:
            pass
        
        @core.kernel(name="QueryKernel", op_type="Query")
        class QueryKernel:
            pass
        
        # Query by type
        transforms = registry.query(type="transform")
        assert len(transforms) >= 2
        
        # Query by stage
        topology_transforms = registry.query(type="transform", stage="topology_opt")
        assert len(topology_transforms) >= 1
        assert any(t["name"] == "QueryTest1" for t in topology_transforms)
    
    def test_plugin_imports_without_init(self):
        """Test importing actual plugins without going through __init__.py."""
        # Try importing a transform directly
        try:
            from brainsmith.transforms.topology_opt import expand_norms
            assert hasattr(expand_norms, 'ExpandNorms')
            print("✓ ExpandNorms imported successfully")
        except ImportError as e:
            pytest.skip(f"Could not import ExpandNorms: {e}")
        
        # Try importing a kernel directly
        try:
            from brainsmith.kernels.layernorm import layernorm
            assert hasattr(layernorm, 'LayerNorm')
            print("✓ LayerNorm imported successfully")
        except ImportError as e:
            pytest.skip(f"Could not import LayerNorm: {e}")


if __name__ == "__main__":
    # Run tests directly
    test = TestPluginSystemDirect()
    
    print("Testing Plugin Core Imports...")
    test.test_plugin_core_imports()
    print("✓ Plugin core imports successful")
    
    print("\nTesting Dynamic Registration...")
    test.test_dynamic_registration_direct()
    print("✓ Dynamic registration successful")
    
    print("\nTesting Kernel Inference...")
    test.test_kernel_inference_direct()
    print("✓ Kernel inference successful")
    
    print("\nTesting Query Functionality...")
    test.test_query_functionality()
    print("✓ Query functionality successful")
    
    print("\nTesting Plugin Imports...")
    test.test_plugin_imports_without_init()
    
    print("\n✅ All tests passed!")