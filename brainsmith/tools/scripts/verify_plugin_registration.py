#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Verify that all QONNX and FINN plugins are properly registered.

This script demonstrates proper usage of the plugin system's public API.
"""
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainsmith.core.plugins import (
    list_transforms, list_kernels, list_backends, list_steps,
    get_transforms_by_metadata, get_kernels_by_metadata,
    get_backends_by_metadata, get_steps_by_metadata,
    has_transform, has_kernel, has_backend, has_step,
    get_transform, get_kernel, get_backend, get_step,
    list_all_kernels, list_all_steps
)
from brainsmith.core.plugins.framework_adapters import ensure_initialized


def count_plugins_by_framework() -> Dict[str, Dict[str, int]]:
    """Count all registered plugins by type and framework using metadata queries."""
    counts = {
        'transform': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'kernel': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'backend': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'step': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
    }
    
    # Count transforms by framework
    for framework in ['brainsmith', 'qonnx', 'finn']:
        transforms = get_transforms_by_metadata(framework=framework)
        counts['transform'][framework] = len(transforms)
    
    # Count kernels by framework
    for framework in ['brainsmith', 'qonnx', 'finn']:
        kernels = get_kernels_by_metadata(framework=framework)
        counts['kernel'][framework] = len(kernels)
    
    # Count backends by framework
    for framework in ['brainsmith', 'qonnx', 'finn']:
        backends = get_backends_by_metadata(framework=framework)
        counts['backend'][framework] = len(backends)
    
    # Count steps by framework
    for framework in ['brainsmith', 'qonnx', 'finn']:
        steps = get_steps_by_metadata(framework=framework)
        counts['step'][framework] = len(steps)
    
    return counts


def verify_key_plugins() -> List[Tuple[str, str, bool]]:
    """Verify that key plugins from each framework are accessible."""
    verifications = []
    
    # Key QONNX transforms
    qonnx_transforms = [
        'qonnx:InferShapes',
        'qonnx:FoldConstants',
        'qonnx:RemoveIdentityOps',
        'qonnx:GiveUniqueNodeNames',
        'qonnx:ConvertBipolarMatMulToXnorPopcount'
    ]
    
    # Key FINN transforms
    finn_transforms = [
        'finn:Streamline',
        'finn:ConvertSignToThres',
        'finn:InferBinaryMatrixVectorActivation',  # Correct name
        'finn:InferQuantizedMatrixVectorActivation',
        'finn:ConvertQONNXtoFINN'  # FINN-specific, not FoldConstants
    ]
    
    # Key kernels
    kernels = [
        'LayerNorm',  # Brainsmith
        'finn:MVAU',
        'finn:Thresholding',
        'finn:Pool'  # Correct name
    ]
    
    # Check transforms
    for transform in qonnx_transforms + finn_transforms:
        exists = has_transform(transform)
        verifications.append(('transform', transform, exists))
        
    # Check kernels
    for kernel in kernels:
        exists = has_kernel(kernel)
        verifications.append(('kernel', kernel, exists))
    
    return verifications


def display_backend_details():
    """Display backend language breakdown and kernel mappings."""
    # Get backends by language
    hls_backends = get_backends_by_metadata(language='hls')
    rtl_backends = get_backends_by_metadata(language='rtl')
    
    print(f"\nHLS Backends: {len(hls_backends)}")
    print(f"RTL Backends: {len(rtl_backends)}")
    
    # Show kernel to backend mappings
    print("\n=== Kernel to Backend Mappings ===")
    kernel_backends = list_all_kernels()
    
    # Show a sample of mappings
    sample_kernels = list(kernel_backends.keys())[:10]
    for kernel in sample_kernels:
        backends = kernel_backends[kernel]
        backend_str = ", ".join(backends)
        print(f"{kernel:<30} -> {backend_str}")
    
    if len(kernel_backends) > 10:
        print(f"... and {len(kernel_backends) - 10} more kernels")


def test_plugin_retrieval():
    """Test that we can retrieve specific plugins."""
    print("\n=== Plugin Retrieval Tests ===")
    
    test_cases = [
        ('transform', 'Streamline', 'finn:Streamline'),
        ('transform', 'InferShapes', 'qonnx:InferShapes'),
        ('kernel', 'MVAU', 'finn:MVAU'),
        ('kernel', 'LayerNorm', 'LayerNorm'),
    ]
    
    for plugin_type, short_name, full_name in test_cases:
        try:
            if plugin_type == 'transform':
                plugin = get_transform(short_name)
            elif plugin_type == 'kernel':
                plugin = get_kernel(short_name)
            
            print(f"✓ Retrieved {plugin_type} '{short_name}' -> {plugin.__name__}")
        except KeyError as e:
            print(f"✗ Failed to retrieve {plugin_type} '{short_name}': {e}")


def display_kernel_inference_transforms():
    """Display kernel inference transforms."""
    print("\n=== Kernel Inference Transforms ===")
    
    # Get all transforms with kernel_inference metadata
    kernel_infer_transforms = get_transforms_by_metadata(kernel_inference=True)
    print(f"Total kernel inference transforms: {len(kernel_infer_transforms)}")
    
    # Group by kernel
    by_kernel = {}
    for transform in kernel_infer_transforms:
        # Try to get the kernel metadata
        try:
            # We need to get the transform class first to access its metadata
            if transform.startswith('finn:'):
                kernel = 'finn'  # FINN transforms don't have kernel metadata
            else:
                # For Brainsmith transforms, we'd need to check metadata
                kernel = 'unknown'
        except:
            kernel = 'unknown'
        
        if kernel not in by_kernel:
            by_kernel[kernel] = []
        by_kernel[kernel].append(transform)
    
    # Display sample
    for kernel, transforms in list(by_kernel.items())[:5]:
        print(f"\n{kernel}:")
        for t in transforms[:3]:
            print(f"  - {t}")


def main():
    print("=== Brainsmith Plugin Registry Verification ===\n")
    
    # Ensure external plugins are initialized
    ensure_initialized()
    
    # Count plugins using metadata queries
    counts = count_plugins_by_framework()
    
    # Display summary table
    print("Type       | Brainsmith | QONNX | FINN  | Total")
    print("-----------|------------|-------|-------|------")
    
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        bs = counts[plugin_type]['brainsmith']
        qx = counts[plugin_type]['qonnx']
        fn = counts[plugin_type]['finn']
        total = bs + qx + fn
        print(f"{plugin_type:<10} | {bs:>10} | {qx:>5} | {fn:>5} | {total:>5}")
    
    # Show totals using list functions
    print("\n=== Total Counts (via list functions) ===")
    print(f"Transforms: {len(list_transforms())}")
    print(f"Kernels: {len(list_kernels())}")
    print(f"Backends: {len(list_backends())}")
    print(f"Steps: {len(list_steps())}")
    print(f"Unique step names: {len(list_all_steps())}")
    
    # Verify key plugins
    print("\n=== Key Plugin Verification ===")
    verifications = verify_key_plugins()
    failures = [v for v in verifications if not v[2]]
    
    if failures:
        print(f"\n⚠️  {len(failures)} plugins missing:")
        for plugin_type, name, _ in failures:
            print(f"  - {plugin_type}: {name}")
    else:
        print("✓ All key plugins verified successfully")
    
    # Display backend details
    display_backend_details()
    
    # Test plugin retrieval
    test_plugin_retrieval()
    
    # Display kernel inference transforms
    display_kernel_inference_transforms()
    
    # Show sample plugins
    print("\n=== Sample Registered Plugins ===")
    
    # Get samples using framework metadata
    qonnx_transforms = get_transforms_by_metadata(framework='qonnx')[:5]
    finn_transforms = get_transforms_by_metadata(framework='finn')[:5]
    
    print("\nQONNX transform samples:")
    for t in qonnx_transforms:
        print(f"  - {t}")
    
    print("\nFINN transform samples:")
    for t in finn_transforms:
        print(f"  - {t}")
    


if __name__ == "__main__":
    main()