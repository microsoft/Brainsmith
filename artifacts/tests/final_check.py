#!/usr/bin/env python3
"""Final check for copyright headers."""

import os

# All Python files that should have been checked
all_files = [
    "brainsmith/__init__.py",
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
    "brainsmith/kernels/crop/crop.py",
    "brainsmith/kernels/crop/crop_hls.py",
    "brainsmith/kernels/layernorm/layernorm.py",
    "brainsmith/kernels/layernorm/layernorm_hls.py",
    "brainsmith/kernels/shuffle/shuffle.py",
    "brainsmith/kernels/shuffle/shuffle_hls.py",
    "brainsmith/kernels/softmax/hwsoftmax.py",
    "brainsmith/kernels/softmax/hwsoftmax_hls.py",
    "brainsmith/steps/__init__.py",
    "brainsmith/steps/bert_custom_steps.py",
    "brainsmith/steps/core_steps.py",
    "brainsmith/steps/kernel_inference.py",
    "brainsmith/transforms/__init__.py",
    "brainsmith/transforms/cleanup/remove_identity.py",
    "brainsmith/transforms/dataflow_opt/infer_finn_loop_op.py",
    "brainsmith/transforms/kernel_opt/set_pumped_compute.py",
    "brainsmith/transforms/kernel_opt/temp_shuffle_fixer.py",
    "brainsmith/transforms/metadata/ensure_custom_opset_imports.py",
    "brainsmith/transforms/metadata/extract_shell_integration_metadata.py",
    "brainsmith/transforms/topology_opt/expand_norms.py",
    "demos/bert_modern/bert_demo.py",
    "tests/test_blueprint_inheritance.py",
    "tests/test_execution_tree.py",
    "tests/test_executor.py",
    "tests/test_plugin_system.py"
]

base_dir = "/home/tafk/dev/brainsmith-4"
issues = []

for file_path in all_files:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            lines = f.readlines()[:3]  # Read first 3 lines
        
        # Check for correct Microsoft copyright (handle both formats)
        # Format 1: Simple format
        has_simple_format = (
            len(lines) >= 2 and
            lines[0].strip() == "# Copyright (c) Microsoft Corporation." and
            lines[1].strip() == "# Licensed under the MIT License."
        )
        
        # Format 2: Format with borders and author info
        has_bordered_format = (
            len(lines) >= 3 and
            lines[0].strip().startswith("####") and
            "Copyright (c) Microsoft Corporation" in lines[1] and
            "MIT License" in lines[2]
        )
        
        has_correct_copyright = has_simple_format or has_bordered_format
        
        if not has_correct_copyright:
            issues.append(f"{file_path}: Missing or incorrect copyright header")
    else:
        issues.append(f"{file_path}: FILE NOT FOUND")

if issues:
    print("Issues found:")
    for issue in issues:
        print(f"  - {issue}")
else:
    print("✓ All files have correct Microsoft copyright headers!")
    print(f"  Total files checked: {len(all_files)}")
    print(f"  All files have the correct header format:")
    print("    # Copyright (c) Microsoft Corporation.")
    print("    # Licensed under the MIT License.")