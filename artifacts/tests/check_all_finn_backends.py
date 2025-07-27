#!/usr/bin/env python3
"""Comprehensive check of all FINN backend registrations."""

import os
import re
from pathlib import Path

def find_all_backend_classes():
    """Find all classes inheriting from HLSBackend or RTLBackend in FINN."""
    backends = {}
    
    # Search entire FINN repository
    finn_dir = '/home/tafk/dev/brainsmith-4/deps/finn'
    
    for root, dirs, files in os.walk(finn_dir):
        for filename in files:
            if filename.endswith('.py'):
                filepath = os.path.join(root, filename)
                
                try:
                    with open(filepath, 'r') as f:
                        content = f.read()
                    
                    # Find HLSBackend classes
                    hls_matches = re.findall(r'^class\s+(\w+)\s*\([^)]*HLSBackend[^)]*\):', content, re.MULTILINE)
                    for cls in hls_matches:
                        if cls != 'HLSBackend':  # Skip base class
                            backends[cls] = ('hls', filepath)
                    
                    # Find RTLBackend classes
                    rtl_matches = re.findall(r'^class\s+(\w+)\s*\([^)]*RTLBackend[^)]*\):', content, re.MULTILINE)
                    for cls in rtl_matches:
                        if cls != 'RTLBackend':  # Skip base class
                            backends[cls] = ('rtl', filepath)
                except Exception as e:
                    pass  # Skip files we can't read
    
    return backends

def get_registered_backends():
    """Get all backends registered in BrainSmith."""
    from brainsmith.core.plugins.registry import list_backends
    
    all_registered = list_backends()
    finn_registered = {}
    
    for b in all_registered:
        if 'finn:' in b:
            name = b.split(':')[1]
            finn_registered[name] = b
    
    return finn_registered

def main():
    # Find all backend classes
    all_backends = find_all_backend_classes()
    
    # Get registered backends
    registered = get_registered_backends()
    
    # Separate by type
    hls_backends = {k: v for k, v in all_backends.items() if v[0] == 'hls'}
    rtl_backends = {k: v for k, v in all_backends.items() if v[0] == 'rtl'}
    
    print(f"=== FINN Backend Analysis ===")
    print(f"Total backend classes found: {len(all_backends)}")
    print(f"  - HLS backends: {len(hls_backends)}")
    print(f"  - RTL backends: {len(rtl_backends)}")
    print(f"Total registered in BrainSmith: {len(registered)}")
    
    # Find missing
    missing = set(all_backends.keys()) - set(registered.keys())
    extra = set(registered.keys()) - set(all_backends.keys())
    
    if missing:
        print(f"\n✗ Missing backends ({len(missing)}):")
        for b in sorted(missing):
            backend_type, filepath = all_backends[b]
            rel_path = filepath.replace('/home/tafk/dev/brainsmith-4/deps/finn/', '')
            print(f"  - {b} ({backend_type.upper()}) in {rel_path}")
    
    if extra:
        print(f"\n✗ Extra registered ({len(extra)}):")
        for b in sorted(extra):
            print(f"  + {b}")
    
    if not missing and not extra:
        print("\n✓ All FINN backends are properly registered!")
    
    # List all backends by type
    print("\n=== HLS Backends ===")
    for b in sorted(hls_backends.keys()):
        status = "✓" if b in registered else "✗"
        print(f"  {status} {b}")
    
    print("\n=== RTL Backends ===")
    for b in sorted(rtl_backends.keys()):
        status = "✓" if b in registered else "✗"
        print(f"  {status} {b}")

if __name__ == "__main__":
    main()