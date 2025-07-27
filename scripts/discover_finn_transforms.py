#!/usr/bin/env python3
"""
Discover all FINN transforms by scanning the module.
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

def discover_all_finn_transforms():
    """Discover all transforms in finn.transformation package."""
    # First, let's find all finn transform modules
    finn_base = Path(__file__).parent.parent / "deps" / "finn" / "src" / "finn" / "transformation"
    
    transform_modules = []
    
    # Walk the finn transformation directory
    for root, dirs, files in os.walk(finn_base):
        for file in files:
            if file.endswith('.py') and file != '__init__.py':
                # Convert file path to module name
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

if __name__ == "__main__":
    print("Discovering FINN transforms...")
    transforms = discover_all_finn_transforms()
    print(f"\nTotal transforms found: {len(transforms)}")
    
    # Generate registration code
    print("\n\nGenerated registration dictionary:")
    print("FINN_TRANSFORMS = {")
    
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