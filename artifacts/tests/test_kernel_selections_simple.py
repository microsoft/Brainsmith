#!/usr/bin/env python3
"""Simple test that kernel selections are properly passed."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force reload to pick up our changes
from brainsmith.core.plugins import get_registry
registry = get_registry()
registry.reset()

from brainsmith.core.plugins import get_transforms_by_metadata

def test_kernel_selections_simple():
    print("=== Testing Kernel Selections (Simple) ===\n")
    
    # Test that all kernels from the blueprint have transforms
    kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinaryOperation', 
               'Shuffle', 'HWSoftmax', 'Thresholding', 'MVAU']
    
    print("Checking kernel -> transform mappings:")
    all_good = True
    for kernel in kernels:
        transforms = get_transforms_by_metadata(kernel=kernel)
        if transforms:
            print(f"  ✓ {kernel}: {', '.join(transforms)}")
        else:
            print(f"  ✗ {kernel}: NO TRANSFORMS FOUND")
            all_good = False
    
    return all_good

if __name__ == "__main__":
    success = test_kernel_selections_simple()
    if not success:
        sys.exit(1)