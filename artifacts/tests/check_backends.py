#!/usr/bin/env python3
"""Check backend registration."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith.core.plugins import get_registry, list_backends
from brainsmith.core.plugins.registry import list_backends_by_kernel

print("=== Checking Backend Registration ===\n")

# Get registry
registry = get_registry()

# Check all registered backends
print("All registered backends:")
backends = list_backends()
for backend in backends:
    print(f"  - {backend}")

print(f"\nTotal backends: {len(backends)}")

# Check backends by kernel
print("\nBackends by kernel:")
kernels = ['LayerNorm', 'DuplicateStreams', 'ElementwiseBinaryOperation', 
           'Shuffle', 'HWSoftmax', 'Thresholding', 'MVAU']

for kernel in kernels:
    backends = list_backends_by_kernel(kernel)
    print(f"  {kernel}: {backends}")

# Check raw registry
print("\nRaw backend registry:")
for name, (cls, metadata) in registry._plugins['backend'].items():
    kernel = metadata.get('kernel', 'unknown')
    print(f"  {name} -> kernel={kernel}")