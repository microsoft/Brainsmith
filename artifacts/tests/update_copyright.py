#!/usr/bin/env python3
"""Script to update copyright headers in Python files."""

import os

# Standard Microsoft copyright header
MICROSOFT_COPYRIGHT = """# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
"""

# Files that need copyright header added
files_need_header = [
    "brainsmith/core/__init__.py",
    "brainsmith/core/blueprint_parser.py",
    "brainsmith/core/design_space.py",
    "brainsmith/core/execution_tree.py",
    "brainsmith/core/explorer/executor.py",
    "brainsmith/core/explorer/explorer.py",
    "brainsmith/core/explorer/finn_adapter.py",
    "brainsmith/core/explorer/utils.py",
    "brainsmith/core/forge.py",
    "brainsmith/core/plugins/__init__.py",
    "brainsmith/core/plugins/framework_adapters.py",
    "brainsmith/core/plugins/registry.py",
    "brainsmith/core/utils.py",
    "brainsmith/kernels/__init__.py",
    "brainsmith/steps/__init__.py",
    "brainsmith/steps/kernel_inference.py",
    "brainsmith/transforms/__init__.py",
    "brainsmith/transforms/cleanup/remove_identity.py",
    "brainsmith/transforms/dataflow_opt/infer_finn_loop_op.py",
    "brainsmith/transforms/kernel_opt/set_pumped_compute.py",
    "brainsmith/transforms/kernel_opt/temp_shuffle_fixer.py",
    "brainsmith/transforms/metadata/ensure_custom_opset_imports.py",
    "brainsmith/transforms/metadata/extract_shell_integration_metadata.py",
    "brainsmith/transforms/topology_opt/expand_norms.py",
    "tests/test_blueprint_inheritance.py",
    "tests/test_execution_tree.py",
    "tests/test_executor.py",
    "tests/test_plugin_system.py"
]

# Files with AMD copyright that need to be replaced
files_amd_copyright = [
    "brainsmith/kernels/layernorm/layernorm_hls.py",
    "brainsmith/kernels/shuffle/shuffle.py",
    "brainsmith/kernels/shuffle/shuffle_hls.py",
    "brainsmith/kernels/softmax/hwsoftmax.py",
    "brainsmith/kernels/softmax/hwsoftmax_hls.py",
    "demos/bert_modern/bert_demo.py"
]

# File that needs "All rights reserved." removed
file_needs_cleanup = "brainsmith/__init__.py"

base_dir = "/home/tafk/dev/brainsmith-4"

# Update files that need header added
for file_path in files_need_header:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            content = f.read()
        
        # Add copyright header at the beginning
        new_content = MICROSOFT_COPYRIGHT + "\n" + content
        
        with open(full_path, 'w') as f:
            f.write(new_content)
        
        print(f"Added copyright header to: {file_path}")
    else:
        print(f"File not found: {file_path}")

# Update files with AMD copyright
for file_path in files_amd_copyright:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            lines = f.readlines()
        
        # Find where the copyright header ends
        header_end = 0
        for i, line in enumerate(lines):
            if line.strip() == "" and i > 0:
                # Empty line after header
                header_end = i
                break
            elif not line.startswith("#") and i > 0:
                # Non-comment line
                header_end = i - 1
                break
        
        # Replace header
        new_lines = [MICROSOFT_COPYRIGHT + "\n"] + lines[header_end + 1:]
        
        with open(full_path, 'w') as f:
            f.writelines(new_lines)
        
        print(f"Replaced AMD copyright with Microsoft in: {file_path}")
    else:
        print(f"File not found: {file_path}")

# Clean up "All rights reserved." from the one file
full_path = os.path.join(base_dir, file_needs_cleanup)
if os.path.exists(full_path):
    with open(full_path, 'r') as f:
        content = f.read()
    
    # Replace the line with "All rights reserved."
    content = content.replace(
        "# Copyright (c) Microsoft Corporation. All rights reserved.",
        "# Copyright (c) Microsoft Corporation."
    )
    
    with open(full_path, 'w') as f:
        f.write(content)
    
    print(f"Removed 'All rights reserved.' from: {file_needs_cleanup}")

print("\nDone! All copyright headers have been updated.")