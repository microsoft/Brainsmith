#!/usr/bin/env python3
"""Check if all BrainSmith kernels are registered."""

import os
import re
from pathlib import Path

def find_brainsmith_kernels():
    """Find all BrainSmith kernel classes decorated with @kernel."""
    kernels = []
    
    # Search for kernel decorators in brainsmith/kernels
    kernels_dir = Path('/home/tafk/dev/brainsmith-4/brainsmith/kernels')
    
    for py_file in kernels_dir.rglob('*.py'):
        if '__pycache__' in str(py_file):
            continue
            
        with open(py_file, 'r') as f:
            content = f.read()
            
        # Find @kernel decorated classes
        # Pattern matches @kernel or @registry.kernel followed by class definition
        pattern = r'@(?:kernel|registry\.kernel).*?\nclass\s+(\w+)'
        matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
        
        for match in matches:
            kernels.append((match, str(py_file)))
    
    return kernels

def check_registered_kernels():
    """Check which BrainSmith kernels are registered."""
    from brainsmith.core.plugins.registry import list_kernels
    
    # Get all registered kernels
    all_registered = list_kernels()
    
    # Filter for BrainSmith kernels (no framework prefix or brainsmith: prefix)
    brainsmith_registered = set()
    for k in all_registered:
        if ':' not in k:
            brainsmith_registered.add(k)
        elif k.startswith('brainsmith:'):
            brainsmith_registered.add(k.split(':')[1])
    
    return brainsmith_registered

def main():
    # Find all BrainSmith kernels
    kernel_info = find_brainsmith_kernels()
    all_kernels = {name for name, _ in kernel_info}
    
    # Check registered kernels
    registered = check_registered_kernels()
    
    print(f"Total BrainSmith kernels found: {len(all_kernels)}")
    print(f"Total registered: {len(registered)}")
    
    # Find missing kernels
    missing = all_kernels - registered
    extra = registered - all_kernels
    
    if not missing and not extra:
        print("\n✓ All BrainSmith kernels are properly registered!")
    else:
        if missing:
            print("\n✗ Missing kernels:")
            for k in sorted(missing):
                # Find file location
                file_loc = next((f for name, f in kernel_info if name == k), "unknown")
                print(f"  - {k} (in {file_loc})")
        
        if extra:
            print("\n✗ Extra registered (not found in code):")
            for k in sorted(extra):
                print(f"  + {k}")
    
    # List all kernels for reference
    print("\nAll BrainSmith kernels:")
    for k in sorted(all_kernels):
        status = "✓" if k in registered else "✗"
        file_loc = next((f for name, f in kernel_info if name == k), "unknown")
        print(f"  {status} {k} ({Path(file_loc).name})")

if __name__ == "__main__":
    main()