#!/usr/bin/env python3
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""
Registry utility scripts for plugin discovery and verification.

Combined utilities:
- verify_plugin_registration: Verify all plugins are properly registered
- discover_finn_transforms: Discover FINN transforms
- discover_qonnx_transforms: Discover QONNX transforms
"""
import os
import sys
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Tuple

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainsmith.registry import (
    list_transforms, list_kernels, list_backends, list_steps,
    get_transforms_by_metadata, get_kernels_by_metadata,
    get_backends_by_metadata, get_steps_by_metadata,
    has_transform, has_kernel, has_backend, has_step,
    get_transform, get_kernel, get_backend, get_step,
    list_all_kernels, list_all_steps
)
from brainsmith.registry_adapters import ensure_initialized


# ==============================================================================
# Plugin Verification
# ==============================================================================

def count_plugins_by_framework() -> Dict[str, Dict[str, int]]:
    """Count all registered plugins by type and framework using metadata queries."""
    counts = {
        'transform': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'kernel': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'backend': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'step': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
    }

    for framework in ['brainsmith', 'qonnx', 'finn']:
        transforms = get_transforms_by_metadata(framework=framework)
        counts['transform'][framework] = len(transforms)

    for framework in ['brainsmith', 'qonnx', 'finn']:
        kernels = get_kernels_by_metadata(framework=framework)
        counts['kernel'][framework] = len(kernels)

    for framework in ['brainsmith', 'qonnx', 'finn']:
        backends = get_backends_by_metadata(framework=framework)
        counts['backend'][framework] = len(backends)

    for framework in ['brainsmith', 'qonnx', 'finn']:
        steps = get_steps_by_metadata(framework=framework)
        counts['step'][framework] = len(steps)

    return counts


def verify_key_plugins() -> List[Tuple[str, str, bool]]:
    """Verify that key plugins from each framework are accessible."""
    verifications = []

    qonnx_transforms = [
        'qonnx:InferShapes',
        'qonnx:FoldConstants',
        'qonnx:RemoveIdentityOps',
        'qonnx:GiveUniqueNodeNames',
        'qonnx:ConvertBipolarMatMulToXnorPopcount'
    ]

    finn_transforms = [
        'finn:Streamline',
        'finn:ConvertSignToThres',
        'finn:InferBinaryMatrixVectorActivation',
        'finn:InferQuantizedMatrixVectorActivation',
        'finn:ConvertQONNXtoFINN'
    ]

    kernels = [
        'LayerNorm',
        'finn:MVAU',
        'finn:Thresholding',
        'finn:Pool'
    ]

    for transform in qonnx_transforms + finn_transforms:
        exists = has_transform(transform)
        verifications.append(('transform', transform, exists))

    for kernel in kernels:
        exists = has_kernel(kernel)
        verifications.append(('kernel', kernel, exists))

    return verifications


def display_backend_details():
    """Display backend language breakdown and kernel mappings."""
    hls_backends = get_backends_by_metadata(language='hls')
    rtl_backends = get_backends_by_metadata(language='rtl')

    print(f"\nHLS Backends: {len(hls_backends)}")
    print(f"RTL Backends: {len(rtl_backends)}")

    print("\n=== Kernel to Backend Mappings ===")
    kernel_backends = list_all_kernels()

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

    kernel_infer_transforms = get_transforms_by_metadata(kernel_inference=True)
    print(f"Total kernel inference transforms: {len(kernel_infer_transforms)}")

    by_kernel = {}
    for transform in kernel_infer_transforms:
        try:
            if transform.startswith('finn:'):
                kernel = 'finn'
            else:
                kernel = 'unknown'
        except:
            kernel = 'unknown'

        if kernel not in by_kernel:
            by_kernel[kernel] = []
        by_kernel[kernel].append(transform)

    for kernel, transforms in list(by_kernel.items())[:5]:
        print(f"\n{kernel}:")
        for t in transforms[:3]:
            print(f"  - {t}")


def verify_registration():
    """Main verification function."""
    print("=== Brainsmith Plugin Registry Verification ===\n")

    ensure_initialized()

    counts = count_plugins_by_framework()

    print("Type       | Brainsmith | QONNX | FINN  | Total")
    print("-----------|------------|-------|-------|------")

    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        bs = counts[plugin_type]['brainsmith']
        qx = counts[plugin_type]['qonnx']
        fn = counts[plugin_type]['finn']
        total = bs + qx + fn
        print(f"{plugin_type:<10} | {bs:>10} | {qx:>5} | {fn:>5} | {total:>5}")

    print("\n=== Total Counts (via list functions) ===")
    print(f"Transforms: {len(list_transforms())}")
    print(f"Kernels: {len(list_kernels())}")
    print(f"Backends: {len(list_backends())}")
    print(f"Steps: {len(list_steps())}")
    print(f"Unique step names: {len(list_all_steps())}")

    print("\n=== Key Plugin Verification ===")
    verifications = verify_key_plugins()
    failures = [v for v in verifications if not v[2]]

    if failures:
        print(f"\n⚠️  {len(failures)} plugins missing:")
        for plugin_type, name, _ in failures:
            print(f"  - {plugin_type}: {name}")
    else:
        print("✓ All key plugins verified successfully")

    display_backend_details()
    test_plugin_retrieval()
    display_kernel_inference_transforms()

    print("\n=== Sample Registered Plugins ===")

    qonnx_transforms = get_transforms_by_metadata(framework='qonnx')[:5]
    finn_transforms = get_transforms_by_metadata(framework='finn')[:5]

    print("\nQONNX transform samples:")
    for t in qonnx_transforms:
        print(f"  - {t}")

    print("\nFINN transform samples:")
    for t in finn_transforms:
        print(f"  - {t}")


# ==============================================================================
# Transform Discovery
# ==============================================================================

def discover_transforms_in_module(module_name):
    """Discover all Transformation subclasses in a module."""
    from qonnx.transformation.base import Transformation

    transforms = []
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Transformation) and obj is not Transformation:
                if obj.__module__ == module_name:
                    transforms.append((name, f"{module_name}.{name}"))
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
    return transforms


def discover_finn_transforms():
    """Discover all transforms in finn.transformation package."""
    finn_base = Path(__file__).parent.parent / "deps" / "finn" / "src" / "finn" / "transformation"

    transform_modules = []

    for root, dirs, files in os.walk(finn_base):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                rel_path = Path(root) / file
                rel_to_finn = rel_path.relative_to(Path(__file__).parent.parent / "deps" / "finn" / "src")
                module_name = str(rel_to_finn).replace('/', '.').replace('.py', '')
                transform_modules.append(module_name)

    all_transforms = []
    for module_name in sorted(transform_modules):
        transforms = discover_transforms_in_module(module_name)
        if transforms:
            print(f"\n{module_name}:")
            for name, full_path in transforms:
                print(f"  - {name}")
                all_transforms.append((name, full_path))

    return all_transforms


def discover_qonnx_transforms():
    """Discover all transforms in qonnx.transformation package."""
    transform_modules = [
        'qonnx.transformation.batchnorm_to_affine',
        'qonnx.transformation.bipolar_to_xnor',
        'qonnx.transformation.change_3d_tensors_to_4d',
        'qonnx.transformation.change_batchsize',
        'qonnx.transformation.change_datalayout',
        'qonnx.transformation.channels_last',
        'qonnx.transformation.create_generic_partitions',
        'qonnx.transformation.double_to_single_float',
        'qonnx.transformation.expose_intermediate',
        'qonnx.transformation.extend_partition',
        'qonnx.transformation.extract_conv_bias',
        'qonnx.transformation.extract_quant_scale_zeropt',
        'qonnx.transformation.fold_constants',
        'qonnx.transformation.gemm_to_matmul',
        'qonnx.transformation.general',
        'qonnx.transformation.infer_data_layouts',
        'qonnx.transformation.infer_datatypes',
        'qonnx.transformation.infer_shapes',
        'qonnx.transformation.insert',
        'qonnx.transformation.insert_topk',
        'qonnx.transformation.lower_convs_to_matmul',
        'qonnx.transformation.make_input_chanlast',
        'qonnx.transformation.merge_onnx_models',
        'qonnx.transformation.pruning',
        'qonnx.transformation.qcdq_to_qonnx',
        'qonnx.transformation.qonnx_to_qcdq',
        'qonnx.transformation.quant_constant_folding',
        'qonnx.transformation.quantize_graph',
        'qonnx.transformation.rebalance_conv',
        'qonnx.transformation.remove',
        'qonnx.transformation.resize_conv_to_deconv',
        'qonnx.transformation.subpixel_to_deconv',
    ]

    all_transforms = []
    for module_name in transform_modules:
        transforms = discover_transforms_in_module(module_name)
        if transforms:
            print(f"\n{module_name}:")
            for name, full_path in transforms:
                print(f"  - {name}")
                all_transforms.append((name, full_path))

    return all_transforms


# ==============================================================================
# Main entry points
# ==============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Registry utilities")
    parser.add_argument('command', choices=['verify', 'discover-finn', 'discover-qonnx'],
                       help='Command to run')

    args = parser.parse_args()

    if args.command == 'verify':
        verify_registration()
    elif args.command == 'discover-finn':
        print("Discovering FINN transforms...")
        transforms = discover_finn_transforms()
        print(f"\nTotal transforms found: {len(transforms)}")
    elif args.command == 'discover-qonnx':
        print("Discovering QONNX transforms...")
        transforms = discover_qonnx_transforms()
        print(f"\nTotal transforms found: {len(transforms)}")
