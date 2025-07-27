#!/usr/bin/env python3
"""Check all BrainSmith plugin registrations."""

from brainsmith.core.plugins.registry import (
    list_kernels, list_backends, list_transforms, list_steps,
    get_transforms_by_metadata, get_backends_by_metadata
)

# Check kernels
brainsmith_kernels = [k for k in list_kernels() if ':' not in k or k.startswith('brainsmith:')]
print(f"BrainSmith Kernels: {len(brainsmith_kernels)}")
for k in sorted(brainsmith_kernels):
    print(f"  - {k}")

# Check backends
brainsmith_backends = [b for b in list_backends() if ':' not in b or b.startswith('brainsmith:')]
print(f"\nBrainSmith Backends: {len(brainsmith_backends)}")
for b in sorted(brainsmith_backends):
    print(f"  - {b}")

# Check transforms
brainsmith_transforms = [t for t in list_transforms() if ':' not in t or t.startswith('brainsmith:')]
print(f"\nBrainSmith Transforms: {len(brainsmith_transforms)}")
for t in sorted(brainsmith_transforms):
    print(f"  - {t}")

# Check kernel inference transforms
kernel_inferences = get_transforms_by_metadata(kernel_inference=True, framework='brainsmith')
print(f"\nBrainSmith Kernel Inference Transforms: {len(kernel_inferences)}")
for t in sorted(kernel_inferences):
    print(f"  - {t}")

# Check steps
brainsmith_steps = [s for s in list_steps() if ':' not in s or s.startswith('brainsmith:')]
print(f"\nBrainSmith Steps: {len(brainsmith_steps)}")
for s in sorted(brainsmith_steps):
    print(f"  - {s}")

# Summary
print("\n=== SUMMARY ===")
print(f"Total BrainSmith components registered:")
print(f"  - Kernels: {len(brainsmith_kernels)}")
print(f"  - Backends: {len(brainsmith_backends)}")
print(f"  - Transforms: {len(brainsmith_transforms)} (including {len(kernel_inferences)} kernel inferences)")
print(f"  - Steps: {len(brainsmith_steps)}")