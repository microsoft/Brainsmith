"""Test transform decorator validation logic."""

import pytest
from brainsmith.plugin.core import transform


class TestTransformValidation:
    """Test the mutual exclusion logic for transform decorator."""
    
    def test_kernel_only_valid(self, clean_registry):
        """Test that kernel-only transforms work."""
        @transform(name="TestKernelTransform", kernel="TestKernel")
        class TestKernelTransform:
            pass
        
        # Should register as kernel_inference
        plugins = clean_registry.query(type="kernel_inference", name="TestKernelTransform")
        assert len(plugins) == 1
        assert plugins[0]["name"] == "TestKernelTransform"
        assert plugins[0]["kernel"] == "TestKernel"
        assert plugins[0]["stage"] is None
    
    def test_stage_only_valid(self, clean_registry):
        """Test that stage-only transforms work."""
        @transform(name="TestRegularTransform", stage="cleanup")
        class TestRegularTransform:
            pass
        
        # Should register as regular transform
        plugins = clean_registry.query(type="transform", name="TestRegularTransform")
        assert len(plugins) == 1
        assert plugins[0]["name"] == "TestRegularTransform"
        assert plugins[0]["stage"] == "cleanup"
    
    def test_both_kernel_and_stage_error(self, clean_registry):
        """Test that specifying both kernel and stage raises error."""
        with pytest.raises(ValueError, match="cannot specify both 'kernel' and 'stage'"):
            @transform(name="TestBoth", kernel="TestKernel", stage="cleanup")
            class TestBoth:
                pass
    
    def test_neither_kernel_nor_stage_error(self, clean_registry):
        """Test that specifying neither kernel nor stage raises error."""
        with pytest.raises(ValueError, match="must specify either 'kernel' or 'stage'"):
            @transform(name="TestNeither")
            class TestNeither:
                pass
    
    def test_name_is_required(self, clean_registry):
        """Test that name parameter is required (should be caught by Python signature)."""
        with pytest.raises(TypeError):
            @transform(stage="cleanup")  # Missing required name parameter
            class TestMissingName:
                pass