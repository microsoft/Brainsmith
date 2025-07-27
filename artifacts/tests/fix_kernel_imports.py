#!/usr/bin/env python3
"""Fix kernel imports by removing register_custom_op decorator."""

import re
from pathlib import Path

def fix_kernel_file(file_path):
    """Remove register_custom_op import and decorator from kernel file."""
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Remove the import line
    content = re.sub(r'from qonnx\.custom_op\.registry import register_custom_op\n', '', content)
    
    # Remove the decorator line
    content = re.sub(r'@register_custom_op\([^)]*\)\n', '', content)
    
    # Write back if changes were made
    with open(file_path, 'w') as f:
        f.write(content)
    
    return content != f.read()

# Fix all kernel files
kernel_files = [
    '/home/tafk/dev/brainsmith-4/brainsmith/kernels/crop/crop.py',
    '/home/tafk/dev/brainsmith-4/brainsmith/kernels/layernorm/layernorm.py',
    '/home/tafk/dev/brainsmith-4/brainsmith/kernels/softmax/hwsoftmax.py',
    # shuffle.py already fixed manually
]

for file_path in kernel_files:
    try:
        fix_kernel_file(file_path)
        print(f"✓ Fixed {Path(file_path).name}")
    except Exception as e:
        print(f"✗ Failed to fix {Path(file_path).name}: {e}")

print("\nDone!")