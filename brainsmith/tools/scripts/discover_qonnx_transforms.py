#!/usr/bin/env python3
"""
Discover all QONNX transforms by scanning the module.
"""
import os
import sys
import importlib
import inspect
from pathlib import Path

# Add deps to path
deps_path = Path(__file__).parent.parent / "deps"
sys.path.insert(0, str(deps_path / "qonnx" / "src"))
sys.path.insert(0, str(deps_path / "finn" / "src"))

# Import base class
from qonnx.transformation.base import Transformation

def discover_transforms_in_module(module_name):
    """Discover all Transformation subclasses in a module."""
    transforms = []
    try:
        module = importlib.import_module(module_name)
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Transformation) and obj is not Transformation:
                # Only include classes defined in this module
                if obj.__module__ == module_name:
                    transforms.append((name, f"{module_name}.{name}"))
    except Exception as e:
        print(f"Error importing {module_name}: {e}")
    return transforms

def discover_all_qonnx_transforms():
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

if __name__ == "__main__":
    print("Discovering QONNX transforms...")
    transforms = discover_all_qonnx_transforms()
    print(f"\nTotal transforms found: {len(transforms)}")
    
    # Generate registration code
    print("\n\nGenerated registration dictionary:")
    print("QONNX_TRANSFORMS = {")
    
    # Group by module
    by_module = {}
    for name, full_path in transforms:
        module = '.'.join(full_path.split('.')[:-1])
        if module not in by_module:
            by_module[module] = []
        by_module[module].append(name)
    
    for module, names in sorted(by_module.items()):
        print(f"    '{module}': [")
        for name in sorted(names):
            print(f"        '{name}',")
        print("    ],")
    print("}")