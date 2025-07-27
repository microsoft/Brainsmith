#!/usr/bin/env python3
"""Trace kernel flow through the system."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from brainsmith import forge
from brainsmith.core.plugins.registry import list_backends_by_kernel

# Get blueprint path
blueprint_path = Path(__file__).parent.parent.parent / "brainsmith/blueprints/bert.yaml"
model_path = "/tmp/dummy_model.onnx"

print("=== Tracing Kernel Flow ===\n")

# Create dummy model
import onnx
dummy = onnx.helper.make_model(
    onnx.helper.make_graph([], "dummy", [], [])
)
onnx.save(dummy, model_path)

# Parse blueprint
print("1. Parsing blueprint...")
design_space, execution_tree = forge(
    model_path=model_path,
    blueprint_path=str(blueprint_path)
)

print(f"\n2. Design space kernel_backends:")
if hasattr(design_space, 'kernel_backends'):
    for kernel, backends in design_space.kernel_backends:
        print(f"   {kernel}: {backends}")
else:
    print("   No kernel_backends attribute!")

print(f"\n3. Execution tree root steps:")
for step in execution_tree.segment_steps[:10]:  # First 10 steps
    if isinstance(step, dict) and step.get('name') == 'infer_kernels':
        print(f"   infer_kernels step: kernel_backends = {step.get('kernel_backends', 'NOT FOUND')}")
    else:
        print(f"   {step}")

# Check backend lookup
print(f"\n4. Direct backend lookup:")
for kernel in ['LayerNorm', 'MVAU', 'Thresholding']:
    backends = list_backends_by_kernel(kernel)
    print(f"   {kernel}: {backends}")