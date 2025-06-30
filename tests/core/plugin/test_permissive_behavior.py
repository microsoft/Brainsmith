"""Test the permissive behavior of the plugin system.

Following PD-1 (Break Fearlessly), the plugin system is now more
permissive to reduce complexity. These tests document the new behavior.
"""

import pytest


class TestPermissiveBehavior:
    """Test suite for permissive plugin behavior."""
    
    def test_duplicate_registration_overwrites(self, clean_registry):
        """Test that duplicate registration overwrites instead of erroring."""
        from brainsmith.plugin.core import transform, get_registry
        
        # Register first version
        @transform(name="Overwrite", stage="cleanup")
        class Version1:
            version = 1
        
        # Register second version - should overwrite
        @transform(name="Overwrite", stage="cleanup")
        class Version2:
            version = 2
        
        # Should get the second version
        registry = get_registry()
        cls = registry.get("transform", "Overwrite")
        assert cls.version == 2
    
    def test_invalid_plugin_type_allowed(self, clean_registry):
        """Test that invalid plugin types are allowed (no validation)."""
        # Direct registration with any type works
        clean_registry.register("custom_type", "TestPlugin", object)
        
        # Can retrieve it
        assert clean_registry.get("custom_type", "TestPlugin") == object
    
    def test_special_characters_allowed(self, clean_registry):
        """Test that special characters in names are allowed."""
        from brainsmith.plugin.core import transform
        
        # These all work now
        @transform(name="Transform:WithColon", stage="cleanup")
        class ColonTransform:
            pass
        
        @transform(name="Transform With Spaces", stage="cleanup")
        class SpaceTransform:
            pass
        
        # Can retrieve them
        assert clean_registry.get("transform", "Transform:WithColon") == ColonTransform
        assert clean_registry.get("transform", "Transform With Spaces") == SpaceTransform
    
    def test_empty_metadata_allowed(self, clean_registry):
        """Test that plugins can be registered with minimal metadata."""
        # Register with just type and name
        clean_registry.register("minimal", "Test", object)
        
        # Works fine
        entry = clean_registry.query(type="minimal", name="Test")[0]
        assert entry["class"] == object
    
    def test_framework_metadata_default(self, clean_registry):
        """Test that framework metadata defaults to brainsmith."""
        from brainsmith.plugin.core import transform, kernel, backend
        
        @transform(name="DefaultFramework", stage="cleanup")
        class TestTransform:
            pass
        
        @kernel(name="DefaultKernel", op_type="Test")
        class TestKernel:
            pass
        
        @backend(name="DefaultBackend", kernel="DefaultKernel", backend_type="hls")
        class TestBackend:
            pass
        
        # All should have brainsmith framework
        transforms = clean_registry.query(type="transform", name="DefaultFramework")
        assert transforms[0]["framework"] == "brainsmith"
        
        kernels = clean_registry.query(type="kernel", name="DefaultKernel")
        assert kernels[0]["framework"] == "brainsmith"
        
        backends = clean_registry.query(type="backend", name="DefaultBackend")
        assert backends[0]["framework"] == "brainsmith"
    
    def test_stage_warnings_only(self, clean_registry):
        """Test that invalid stages only warn, don't error."""
        from brainsmith.plugin.core import transform
        import logging
        
        # Should work but warn
        with pytest.warns(None) as warning_list:
            @transform(name="CustomStage", stage="my_custom_stage")
            class CustomStageTransform:
                pass
        
        # Plugin is registered despite non-standard stage
        assert clean_registry.get("transform", "CustomStage") == CustomStageTransform
    
    def test_query_with_any_operators(self, clean_registry):
        """Test that query accepts any operators (no validation)."""
        from brainsmith.plugin.core import transform
        
        @transform(name="QueryTest", stage="cleanup", custom_field="value")
        class QueryTestTransform:
            pass
        
        # These all return empty results but don't error
        results = clean_registry.query(name__invalid_op="test")
        assert results == []
        
        results = clean_registry.query(custom_field__fake="value")
        assert results == []
        
        # Standard query still works
        results = clean_registry.query(custom_field="value")
        assert len(results) == 1
    
    def test_transform_resolution_with_prefixes(self):
        """Test transform resolution handles prefixes correctly."""
        from brainsmith.steps.transform_resolver import _get_transform_class, clear_cache
        from brainsmith.plugin.core import transform, get_registry
        
        # Clear cache for clean test
        clear_cache()
        
        # Register a transform
        @transform(name="ResolutionTest", stage="cleanup")
        class ResolutionTest:
            pass
        
        # Also register with prefix
        registry = get_registry()
        registry.register("transform", "qonnx:FakeTransform", object)
        
        # Direct lookup works
        assert _get_transform_class("ResolutionTest") == ResolutionTest
        
        # Prefixed lookup works
        assert _get_transform_class("qonnx:FakeTransform") == object
        
        # Prefix search works (tries qonnx: automatically)
        assert _get_transform_class("FakeTransform") == object
    
    def test_kernel_inference_flexible(self, clean_registry):
        """Test kernel inference transforms are flexible."""
        from brainsmith.plugin.core import transform
        
        # This works - kernel with no stage means kernel inference
        @transform(name="FlexibleInference", kernel="TestKernel")
        class FlexibleInference:
            pass
        
        # Registered as kernel_inference type
        assert clean_registry.get("kernel_inference", "FlexibleInference") == FlexibleInference
        assert clean_registry.get("transform", "FlexibleInference") is None
    
    def test_metadata_type_flexibility(self, clean_registry):
        """Test that metadata types are not strictly validated."""
        from brainsmith.plugin.core import transform
        
        # These all work without type validation
        @transform(
            name="FlexibleMetadata",
            stage="cleanup",
            tags="not-a-list",  # Should be list but works
            description=123,     # Should be string but works
            custom={"nested": "data"}
        )
        class FlexibleMetadata:
            pass
        
        # Can query it
        results = clean_registry.query(name="FlexibleMetadata")
        assert results[0]["tags"] == "not-a-list"
        assert results[0]["description"] == 123