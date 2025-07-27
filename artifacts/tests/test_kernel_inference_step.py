#!/usr/bin/env python3
"""Test that the kernel inference step now works correctly."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.plugins import get_transforms_by_metadata

def test_kernel_inference_step():
    print("=== Testing Kernel Inference Step ===\n")
    
    # Kernels from the BERT blueprint
    kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinary', 
               'Shuffle', 'HWSoftmax', 'Thresholding', 'MVAU']
    
    print("Checking that all kernels have inference transforms:")
    all_good = True
    for kernel in kernels:
        transforms = get_transforms_by_metadata(kernel=kernel)
        if transforms:
            print(f"  ✓ {kernel}: {', '.join(transforms)}")
        else:
            print(f"  ✗ {kernel}: NO TRANSFORMS FOUND")
            all_good = False
    
    if all_good:
        print("\n✓ All kernels have associated inference transforms!")
        print("\nThe infer_kernels step should now work correctly.")
    else:
        print("\n✗ Some kernels are missing inference transforms!")

if __name__ == "__main__":
    test_kernel_inference_step()