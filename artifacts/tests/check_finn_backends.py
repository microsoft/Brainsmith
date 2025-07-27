#!/usr/bin/env python3
"""Check if all FINN backends are registered in BrainSmith."""

import os
import re
from pathlib import Path

def find_all_finn_backends():
    """Find all backend classes in FINN HLS and RTL directories."""
    backends = {}
    
    # Check HLS backends
    hls_dir = '/home/tafk/dev/brainsmith-4/deps/finn/src/finn/custom_op/fpgadataflow/hls'
    for filename in os.listdir(hls_dir):
        if filename.endswith('_hls.py') and not filename.startswith('__'):
            filepath = os.path.join(hls_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Find class definitions that inherit from HLSBackend
            matches = re.findall(r'^class\s+(\w+)\s*\([^)]*HLSBackend[^)]*\):', content, re.MULTILINE)
            for cls in matches:
                backends[cls] = ('hls', filepath)
    
    # Check RTL backends
    rtl_dir = '/home/tafk/dev/brainsmith-4/deps/finn/src/finn/custom_op/fpgadataflow/rtl'
    for filename in os.listdir(rtl_dir):
        if filename.endswith('_rtl.py') and not filename.startswith('__'):
            filepath = os.path.join(rtl_dir, filename)
            
            with open(filepath, 'r') as f:
                content = f.read()
                
            # Find class definitions that inherit from RTLBackend
            matches = re.findall(r'^class\s+(\w+)\s*\([^)]*RTLBackend[^)]*\):', content, re.MULTILINE)
            for cls in matches:
                backends[cls] = ('rtl', filepath)
    
    return backends

def check_registered_backends():
    """Check which backends are registered in BrainSmith."""
    from brainsmith.core.plugins.registry import list_backends
    
    # Get all registered backends
    all_registered = list_backends()
    
    # Filter for FINN backends
    finn_registered = set()
    for b in all_registered:
        if 'finn:' in b:
            finn_registered.add(b.split(':')[1])
    
    return finn_registered

def main():
    # Find all FINN backends
    all_backends = find_all_finn_backends()
    
    # Check registered backends
    registered = check_registered_backends()
    
    print(f"Total FINN backends found: {len(all_backends)}")
    print(f"  - HLS: {len([b for b, (t, _) in all_backends.items() if t == 'hls'])}")
    print(f"  - RTL: {len([b for b, (t, _) in all_backends.items() if t == 'rtl'])}")
    print(f"Total registered in BrainSmith: {len(registered)}")
    
    # Find missing backends
    missing = set(all_backends.keys()) - registered
    extra = registered - set(all_backends.keys())
    
    if not missing and not extra:
        print("\n✓ All FINN backends are properly registered!")
    else:
        if missing:
            print("\n✗ Missing backends:")
            for b in sorted(missing):
                backend_type, filepath = all_backends[b]
                print(f"  - {b} ({backend_type.upper()}) in {Path(filepath).name}")
        
        if extra:
            print("\n✗ Extra registered (not found in code):")
            for b in sorted(extra):
                print(f"  + {b}")
    
    # List all backends for reference
    print("\nAll FINN backends:")
    for b in sorted(all_backends.keys()):
        status = "✓" if b in registered else "✗"
        backend_type, filepath = all_backends[b]
        print(f"  {status} {b} ({backend_type.upper()})")

if __name__ == "__main__":
    main()