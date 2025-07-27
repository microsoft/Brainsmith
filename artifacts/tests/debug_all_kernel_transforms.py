#!/usr/bin/env python3
"""Debug all kernel inference transforms."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.plugins import get_registry, get_transforms_by_metadata

def debug_all_kernel_transforms():
    print("=== All Kernel Inference Transforms ===\n")
    
    registry = get_registry()
    
    # Find all transforms with kernel metadata
    print("1. BrainSmith transforms with kernel= metadata:")
    for name, (cls, metadata) in registry._plugins['transform'].items():
        if 'kernel' in metadata and not name.startswith('finn:'):
            print(f"  {name} -> kernel={metadata['kernel']}")
    
    print("\n2. FINN transforms with kernel= metadata:")
    for name, (cls, metadata) in registry._plugins['transform'].items():
        if 'kernel' in metadata and name.startswith('finn:'):
            print(f"  {name} -> kernel={metadata['kernel']}")
    
    print("\n3. Summary by kernel:")
    kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinary', 
               'Shuffle', 'Softmax', 'Thresholding', 'MVAU', 'HWSoftmax']
    
    for kernel in kernels:
        transforms = get_transforms_by_metadata(kernel=kernel)
        if transforms:
            print(f"  {kernel}: {', '.join(transforms)}")

if __name__ == "__main__":
    debug_all_kernel_transforms()