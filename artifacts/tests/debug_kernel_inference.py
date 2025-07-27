#!/usr/bin/env python3
"""Debug kernel inference transform lookup."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.plugins import (
    get_transforms_by_metadata, 
    list_transforms,
    get_registry
)

def debug_kernel_inference():
    print("=== Debugging Kernel Inference Transform Lookup ===\n")
    
    # List all transforms that have kernel_inference metadata
    print("1. Transforms with kernel_inference=True metadata:")
    kernel_inference_transforms = get_transforms_by_metadata(kernel_inference=True)
    for t in kernel_inference_transforms:
        print(f"  - {t}")
    print(f"  Total: {len(kernel_inference_transforms)}")
    
    # Check specific kernels
    kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinary', 
               'Shuffle', 'Softmax', 'Thresholding', 'MVAU']
    
    print("\n2. Looking for transforms by kernel= metadata:")
    for kernel in kernels:
        transforms = get_transforms_by_metadata(kernel=kernel)
        print(f"  {kernel}: {transforms}")
    
    # Check the registry directly
    print("\n3. Direct registry inspection:")
    registry = get_registry()
    
    # Look for specific transform names
    inference_names = [
        'InferLayerNorm', 'InferDuplicateStreamsLayer', 
        'InferElementwiseBinaryOperation', 'InferShuffle',
        'InferHWSoftmax', 'InferThresholdingLayer',
        'InferQuantizedMatrixVectorActivation'
    ]
    
    for name in inference_names:
        found = False
        for key, (cls, metadata) in registry._plugins['transform'].items():
            if name in key:
                print(f"\n  Transform: {key}")
                print(f"    Class: {cls}")
                print(f"    Metadata: {metadata}")
                found = True
                break
        if not found:
            print(f"\n  Transform {name}: NOT FOUND")

if __name__ == "__main__":
    debug_kernel_inference()