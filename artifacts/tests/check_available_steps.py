"""Check what steps are available in the registry."""

from brainsmith.core.plugins import list_steps, get_registry

# Ensure plugins are loaded
get_registry().reset()

# List all available steps
steps = list_steps()
print(f"Available steps ({len(steps)}):")
for step in sorted(steps):
    print(f"  - {step}")

# Check if specific steps exist
test_steps = ["cleanup", "streamlining", "infer_kernels", "shell_metadata_handover"]
print("\nChecking test steps:")
for step in test_steps:
    exists = step in steps
    print(f"  - {step}: {'✓' if exists else '✗'}")

# Look for FINN steps
finn_steps = [s for s in steps if s.startswith("finn:")]
print(f"\nFINN steps ({len(finn_steps)}):")
for step in sorted(finn_steps):
    print(f"  - {step}")