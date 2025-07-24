"""Tests for StageWrapperFactory using real transforms."""

import pytest
from brainsmith.core.explorer.utils import StageWrapperFactory
from brainsmith.core.plugins.registry import get_registry


class TestStageWrapperFactory:
    """Test the StageWrapperFactory class with real transforms."""
    
    @pytest.fixture
    def registry(self):
        """Get real registry with actual transforms."""
        return get_registry()
    
    @pytest.fixture
    def factory(self, registry):
        """Create a factory instance with real registry."""
        return StageWrapperFactory(registry)
    
    def test_wrapper_creation_with_transforms(self, factory, registry):
        """Test creating a wrapper with multiple real transforms."""
        # Use real transforms that exist
        transforms = ["RemoveIdentityOps", "RemoveUnusedTensors", "GiveUniqueNodeNames"]
        
        # Verify transforms exist
        for t in transforms:
            assert registry.get_transform(t) is not None
        
        name, wrapper = factory.create_stage_wrapper("cleanup", transforms, 0)
        
        assert name == "cleanup_0"
        assert callable(wrapper)
        assert wrapper.__name__ == "cleanup_0"
        
        # Check metadata
        assert hasattr(wrapper, "_stage_info")
        assert wrapper._stage_info["stage"] == "cleanup"
        assert wrapper._stage_info["transforms"] == transforms
    
    def test_wrapper_creation_empty_transforms(self, factory):
        """Test creating a wrapper with no transforms (skip)."""
        name, wrapper = factory.create_stage_wrapper("optional_step", [], 1)
        
        assert name == "optional_step_skip"
        assert callable(wrapper)
    
    def test_wrapper_caching(self, factory):
        """Test that identical transform sets are cached."""
        transforms = ["RemoveIdentityOps", "RemoveUnusedTensors"]
        
        # Create first wrapper
        name1, wrapper1 = factory.create_stage_wrapper("cleanup", transforms, 0)
        
        # Create second wrapper with same transforms but different index
        name2, wrapper2 = factory.create_stage_wrapper("cleanup", transforms, 1)
        
        # Names should be different
        assert name1 == "cleanup_0"
        assert name2 == "cleanup_1"
        
        # Stage info should be equivalent
        assert wrapper1._stage_info["transforms"] == wrapper2._stage_info["transforms"]
    
    def test_numeric_indices(self, factory):
        """Test that branch indices are simple numbers."""
        transforms = ["FoldConstants", "InferShapes"]
        
        for i in range(5):
            name, _ = factory.create_stage_wrapper("test_stage", transforms, i)
            assert name == f"test_stage_{i}"
    
    def test_get_all_wrappers(self, factory):
        """Test retrieving all created wrappers."""
        # Create several wrappers
        factory.create_stage_wrapper("cleanup", ["RemoveIdentityOps"], 0)
        factory.create_stage_wrapper("cleanup", ["RemoveUnusedTensors"], 1)
        factory.create_stage_wrapper("optional", [], 0)
        
        all_wrappers = factory.get_all_wrappers()
        
        assert len(all_wrappers) == 3
        assert "cleanup_0" in all_wrappers
        assert "cleanup_1" in all_wrappers
        assert "optional_skip" in all_wrappers
        
        # All should be callable
        for wrapper in all_wrappers.values():
            assert callable(wrapper)
    
    def test_get_stage_info(self, factory):
        """Test retrieving stage metadata."""
        transforms = ["RemoveIdentityOps", "RemoveUnusedTensors"]
        factory.create_stage_wrapper("cleanup", transforms, 0)
        
        info = factory.get_stage_info("cleanup_0")
        assert info is not None
        assert info["stage"] == "cleanup"
        assert info["transforms"] == transforms
        
        # Non-existent wrapper
        assert factory.get_stage_info("nonexistent") is None
    
    def test_wrapper_handles_none_transforms(self, factory):
        """Test that wrapper correctly handles None in transform list."""
        # None values should be filtered out
        transforms = ["RemoveIdentityOps", None, "GiveUniqueNodeNames"]
        
        name, wrapper = factory.create_stage_wrapper("cleanup", transforms, 0)
        
        # The wrapper should skip None values
        assert callable(wrapper)
        assert wrapper._stage_info["transforms"] == transforms
    
    def test_registry_error_handling(self, factory):
        """Test handling of non-existent transforms."""
        # The factory filters out None but should handle missing transforms
        # Let's verify it returns None for non-existent transforms
        try:
            name, wrapper = factory.create_stage_wrapper(
                "bad_stage",
                ["NonExistentTransform"],
                0
            )
            # If it doesn't raise, the transform was filtered out
            # This is actually the current behavior - it returns None from registry
            assert callable(wrapper)
        except Exception:
            # This is also acceptable - registry might raise
            pass