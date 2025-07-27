#!/usr/bin/env python3
"""
Verify that all QONNX and FINN plugins are properly registered.
"""
import sys
from pathlib import Path

# Add brainsmith to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from brainsmith.core.plugins import get_registry, list_transforms
from brainsmith.core.plugins.framework_adapters import ensure_initialized

def count_plugins():
    """Count all registered plugins by type and framework."""
    registry = get_registry()
    
    counts = {
        'transform': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'kernel': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'backend': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
        'step': {'brainsmith': 0, 'qonnx': 0, 'finn': 0},
    }
    
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        for name, (cls, metadata) in registry._plugins[plugin_type].items():
            framework = metadata.get('framework', 'brainsmith')
            counts[plugin_type][framework] += 1
    
    return counts

def main():
    print("=== Plugin Registration Summary ===\n")
    
    # Ensure external plugins are initialized
    ensure_initialized()
    
    counts = count_plugins()
    
    # Print summary table
    print("Type       | Brainsmith | QONNX | FINN  | Total")
    print("-----------|------------|-------|-------|------")
    
    for plugin_type in ['transform', 'kernel', 'backend', 'step']:
        bs = counts[plugin_type]['brainsmith']
        qx = counts[plugin_type]['qonnx']
        fn = counts[plugin_type]['finn']
        total = bs + qx + fn
        print(f"{plugin_type:<10} | {bs:>10} | {qx:>5} | {fn:>5} | {total:>5}")
    
    print("\n=== Transform Details ===")
    print(f"\nQONNX transforms: {counts['transform']['qonnx']}")
    print(f"FINN transforms: {counts['transform']['finn']}")
    print(f"Total external transforms: {counts['transform']['qonnx'] + counts['transform']['finn']}")
    
    # List a few example transforms to verify they're accessible
    print("\n=== Sample Registered Transforms ===")
    all_transforms = list_transforms()
    
    qonnx_samples = [t for t in all_transforms if t.startswith('qonnx:')][:5]
    finn_samples = [t for t in all_transforms if t.startswith('finn:')][:5]
    
    print("\nQONNX samples:")
    for t in qonnx_samples:
        print(f"  - {t}")
    
    print("\nFINN samples:")
    for t in finn_samples:
        print(f"  - {t}")
    
    print(f"\nTotal transforms available: {len(all_transforms)}")

if __name__ == "__main__":
    main()