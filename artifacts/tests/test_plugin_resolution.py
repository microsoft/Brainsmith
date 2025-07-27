#!/usr/bin/env python3
"""Test plugin name resolution with and without framework prefixes."""

from brainsmith.core.plugins import get_transform, get_kernel, get_step

# Test unique names - should work without prefix
print("Testing unique plugin names (should work without framework prefix):")

# Test a transform that only exists in QONNX
try:
    transform = get_transform('InferShapes')
    print(f"✓ get_transform('InferShapes') found: {transform.__name__}")
except:
    print("✗ get_transform('InferShapes') failed")

# Test a transform that only exists in FINN  
try:
    transform = get_transform('Streamline')
    print(f"✓ get_transform('Streamline') found: {transform.__name__}")
except:
    print("✗ get_transform('Streamline') failed")

# Test a kernel that only exists in FINN
try:
    kernel = get_kernel('MVAU')
    print(f"✓ get_kernel('MVAU') found: {kernel.__name__}")
except:
    print("✗ get_kernel('MVAU') failed")

# Test a step that only exists in FINN
try:
    step = get_step('tidy_up')
    print(f"✓ get_step('tidy_up') found: {step.__name__}")
except:
    print("✗ get_step('tidy_up') failed")

print("\nTesting ambiguous names (multiple frameworks have the same name):")

# Test RemoveIdentityOps which exists in both QONNX and BrainSmith
try:
    transform = get_transform('RemoveIdentityOps')
    # Check which one we got
    import inspect
    module = inspect.getmodule(transform)
    if 'qonnx' in str(module):
        framework = 'qonnx'
    elif 'brainsmith' in str(module):
        framework = 'brainsmith'
    else:
        framework = 'unknown'
    print(f"✓ get_transform('RemoveIdentityOps') found: {framework} version")
except:
    print("✗ get_transform('RemoveIdentityOps') failed")

# Test with explicit prefix to get specific version
try:
    qonnx_version = get_transform('qonnx:RemoveIdentityOps')
    bs_version = get_transform('RemoveIdentityOps')  # Should get BrainSmith version
    print(f"✓ Can disambiguate with prefix: qonnx:RemoveIdentityOps")
except:
    print("✗ Explicit prefix failed")

print("\nTesting transforms that exist in multiple frameworks:")
# List some that might be duplicated
test_names = ['InferDataTypes', 'GiveUniqueNodeNames', 'ConvertDivToMul']
for name in test_names:
    try:
        transform = get_transform(name)
        print(f"  {name}: found (without prefix)")
    except:
        print(f"  {name}: NOT FOUND without prefix")