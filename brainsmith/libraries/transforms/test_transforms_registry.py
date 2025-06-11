"""
Test new registry-based transform discovery functions.
Validates get_transform(), list_transforms().
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from brainsmith.libraries.transforms import get_transform, list_transforms


def test_transforms_registry():
    """Test all new transform registry functions"""
    
    print("Testing transforms registry functions...")
    
    # Test list_transforms()
    try:
        transforms = list_transforms()
        print(f"✅ list_transforms(): Found {len(transforms)} transforms")
        print(f"   Transforms: {transforms[:5]}...")  # Show first 5
        assert isinstance(transforms, list)
        assert "cleanup" in transforms
        assert "streamlining" in transforms
        assert "infer_hardware" in transforms
        assert len(transforms) == 10  # Expected number of transforms
    except Exception as e:
        print(f"❌ list_transforms() failed: {e}")
        return False
    
    # Test get_transform() - success case
    try:
        cleanup_fn = get_transform("cleanup")
        print(f"✅ get_transform('cleanup'): {cleanup_fn.__name__}")
        assert callable(cleanup_fn)
        assert cleanup_fn.__name__ == "cleanup_step"
    except Exception as e:
        print(f"❌ get_transform() success case failed: {e}")
        return False
    
    # Test get_transform() - another success case
    try:
        streamlining_fn = get_transform("streamlining")
        print(f"✅ get_transform('streamlining'): {streamlining_fn.__name__}")
        assert callable(streamlining_fn)
        assert streamlining_fn.__name__ == "streamlining_step"
    except Exception as e:
        print(f"❌ get_transform() streamlining failed: {e}")
        return False
    
    # Test get_transform() - error case
    try:
        transform = get_transform("nonexistent_transform")
        print(f"❌ get_transform() should have failed for nonexistent transform")
        return False
    except KeyError as e:
        print(f"✅ get_transform() error case: {e}")
        assert "not found" in str(e)
        assert "Available:" in str(e)
    except Exception as e:
        print(f"❌ get_transform() error case unexpected exception: {e}")
        return False
    
    print("✅ All transform registry function tests PASSED")
    return True




if __name__ == "__main__":
    registry_success = test_transforms_registry()
    
    if registry_success:
        print("\n🎉 TRANSFORMS TESTS PASSED - Clean API verified!")
        print("\n📊 Key improvements:")
        print("   • Simple dictionary lookup for all transforms")
        print("   • Fail-fast errors with helpful messages")
        print("   • Clean API with no legacy bloat")
    
    sys.exit(0 if registry_success else 1)