"""
Test new registry-based kernel discovery functions.
Validates get_kernel(), list_kernels(), get_kernel_files().
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from brainsmith.libraries.kernels import get_kernel, list_kernels, get_kernel_files


def test_registry_functions():
    """Test all new registry functions"""
    
    print("Testing registry functions...")
    
    # Test list_kernels()
    try:
        kernels = list_kernels()
        print(f"✅ list_kernels(): {kernels}")
        assert isinstance(kernels, list)
        assert "conv2d_hls" in kernels
        assert "matmul_rtl" in kernels
        assert len(kernels) == 2
    except Exception as e:
        print(f"❌ list_kernels() failed: {e}")
        return False
    
    # Test get_kernel() - success case
    try:
        kernel = get_kernel("conv2d_hls")
        print(f"✅ get_kernel('conv2d_hls'): {kernel.name} v{kernel.version}")
        assert kernel.name == "conv2d_hls"
        assert kernel.operator_type == "Convolution"
    except Exception as e:
        print(f"❌ get_kernel() success case failed: {e}")
        return False
    
    # Test get_kernel() - error case
    try:
        kernel = get_kernel("nonexistent_kernel")
        print(f"❌ get_kernel() should have failed for nonexistent kernel")
        return False
    except KeyError as e:
        print(f"✅ get_kernel() error case: {e}")
        assert "not found" in str(e)
        assert "Available:" in str(e)
    except Exception as e:
        print(f"❌ get_kernel() error case unexpected exception: {e}")
        return False
    
    # Test get_kernel_files()
    try:
        files = get_kernel_files("matmul_rtl")
        print(f"✅ get_kernel_files('matmul_rtl'): {list(files.keys())}")
        assert isinstance(files, dict)
        assert "rtl_source" in files
        assert "hw_custom_op" in files
        assert files["rtl_source"].endswith("matmul_source_RTL.sv")
    except Exception as e:
        print(f"❌ get_kernel_files() failed: {e}")
        return False
    
    # Test get_kernel_files() - error case
    try:
        files = get_kernel_files("nonexistent_kernel")
        print(f"❌ get_kernel_files() should have failed for nonexistent kernel")
        return False
    except KeyError as e:
        print(f"✅ get_kernel_files() error case: {e}")
        assert "not found" in str(e)
    except Exception as e:
        print(f"❌ get_kernel_files() error case unexpected exception: {e}")
        return False
    
    print("✅ All registry function tests PASSED")
    return True


if __name__ == "__main__":
    success = test_registry_functions()
    sys.exit(0 if success else 1)