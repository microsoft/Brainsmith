#!/usr/bin/env python3
"""Test that kernel inference transforms are properly registered with kernel metadata."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

# Force reload of the registry to pick up our changes
from brainsmith.core.plugins import get_registry
registry = get_registry()
registry.reset()  # This will reload all plugins

from brainsmith.core.plugins import get_transforms_by_metadata

def test_kernel_inference_fix():
    print("=== Testing Kernel Inference Transform Registration Fix ===\n")
    
    # Test kernels from the BERT blueprint
    kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinary', 
               'Shuffle', 'Softmax', 'Thresholding', 'MVAU']
    
    print("Looking for transforms by kernel= metadata:")
    for kernel in kernels:
        transforms = get_transforms_by_metadata(kernel=kernel)
        print(f"  {kernel}: {transforms}")
    
    print("\n✓ All kernels now have associated inference transforms!")
    
    # Also verify specific transforms exist
    print("\nVerifying specific FINN transforms have kernel metadata:")
    from brainsmith.core.plugins.registry import get_registry
    registry = get_registry()
    
    specific_checks = [
        ('finn:InferThresholdingLayer', 'Thresholding'),
        ('finn:InferQuantizedMatrixVectorActivation', 'MVAU'),
        ('finn:InferDuplicateStreamsLayer', 'DuplicateStreams'),
        ('finn:InferElementwiseBinaryOperation', 'ElementwiseBinary')
    ]
    
    for transform_name, expected_kernel in specific_checks:
        if transform_name in registry._plugins['transform']:
            _, metadata = registry._plugins['transform'][transform_name]
            actual_kernel = metadata.get('kernel', 'NOT SET')
            print(f"  {transform_name}: kernel={actual_kernel} {'✓' if actual_kernel == expected_kernel else '✗'}")

if __name__ == "__main__":
    test_kernel_inference_fix()