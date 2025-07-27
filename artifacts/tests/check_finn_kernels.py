#!/usr/bin/env python3
"""Check if all FINN kernels are registered in BrainSmith."""

import os
import re

def find_all_finn_kernels():
    """Find all kernel classes in FINN that inherit from HWCustomOp."""
    kernels = set()
    
    # Base directory for FINN custom ops
    base_dir = '/home/tafk/dev/brainsmith-4/deps/finn/src/finn/custom_op/fpgadataflow'
    
    # Find all Python files in the base directory (not subdirectories)
    for filename in os.listdir(base_dir):
        if filename.endswith('.py') and not filename.startswith('__'):
            filepath = os.path.join(base_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
                
                # Find all class definitions that inherit from HWCustomOp
                # This includes direct inheritance and classes that inherit from other kernels
                hwcustomop_classes = re.findall(r'^class\s+(\w+)\s*\([^)]*HWCustomOp[^)]*\):', content, re.MULTILINE)
                for cls in hwcustomop_classes:
                    if cls != 'HWCustomOp':  # Skip the base class itself
                        kernels.add(cls)
                
                # Special case: ElementwiseBinaryOperation subclasses
                if 'ElementwiseBinaryOperation' in content:
                    subclasses = re.findall(r'^class\s+(\w+)\s*\(ElementwiseBinaryOperation\):', content, re.MULTILINE)
                    kernels.update(subclasses)
    
    return kernels

def check_registered_kernels():
    """Check which kernels are registered in BrainSmith."""
    from brainsmith.core.plugins.registry import list_kernels
    
    # Get all registered kernels
    all_registered = list_kernels()
    
    # Filter for FINN kernels
    finn_registered = set()
    for k in all_registered:
        if 'finn:' in k:
            finn_registered.add(k.split(':')[1])
    
    return finn_registered

def main():
    # Find all FINN kernels
    all_kernels = find_all_finn_kernels()
    
    # Check registered kernels
    registered = check_registered_kernels()
    
    # Remove any incorrectly registered items
    # StreamingDataflowPartition inherits from CustomOp, not HWCustomOp
    if 'StreamingDataflowPartition' in registered:
        registered.remove('StreamingDataflowPartition')
    
    print(f"Total FINN kernels found: {len(all_kernels)}")
    print(f"Total registered in BrainSmith: {len(registered)}")
    
    # Find missing kernels
    missing = all_kernels - registered
    extra = registered - all_kernels
    
    if not missing and not extra:
        print("\n✓ All FINN kernels are properly registered!")
    else:
        if missing:
            print("\n✗ Missing kernels:")
            for k in sorted(missing):
                print(f"  - {k}")
        
        if extra:
            print("\n✗ Extra registered (not actual kernels):")
            for k in sorted(extra):
                print(f"  + {k}")
    
    # List all kernels for reference
    print("\nAll FINN kernels:")
    for k in sorted(all_kernels):
        status = "✓" if k in registered else "✗"
        print(f"  {status} {k}")

if __name__ == "__main__":
    main()