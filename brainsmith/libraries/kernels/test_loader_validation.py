"""
Quick validation test for generic package loader.
Tests loader against existing conv2d_hls and matmul_rtl kernels.
"""

import sys
from pathlib import Path

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from brainsmith.libraries.kernels.package_loader import load_kernel_package


def test_generic_loader():
    """Test generic loader against existing kernels"""
    
    print("Testing generic loader validation...")
    
    # Test conv2d_hls kernel
    try:
        conv2d = load_kernel_package("conv2d_hls")
        print(f"✅ Loaded conv2d_hls: {conv2d.name} v{conv2d.version}")
        print(f"   Operator: {conv2d.operator_type}, Backend: {conv2d.backend}")
        print(f"   Files: {list(conv2d.files.keys())}")
        assert conv2d.name == "conv2d_hls"
        assert conv2d.operator_type == "Convolution"
        assert conv2d.backend == "HLS"
    except Exception as e:
        print(f"❌ Failed to load conv2d_hls: {e}")
        return False
    
    # Test matmul_rtl kernel  
    try:
        matmul = load_kernel_package("matmul_rtl")
        print(f"✅ Loaded matmul_rtl: {matmul.name} v{matmul.version}")
        print(f"   Operator: {matmul.operator_type}, Backend: {matmul.backend}")
        print(f"   Files: {list(matmul.files.keys())}")
        assert matmul.name == "matmul_rtl"
        assert matmul.operator_type == "MatMul"
        assert matmul.backend == "RTL"
    except Exception as e:
        print(f"❌ Failed to load matmul_rtl: {e}")
        return False
    
    print("✅ Generic loader validation PASSED")
    return True


if __name__ == "__main__":
    success = test_generic_loader()
    sys.exit(0 if success else 1)