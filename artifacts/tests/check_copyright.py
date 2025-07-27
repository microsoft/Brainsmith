#!/usr/bin/env python3
"""Script to check copyright headers in Python files."""

import os

files_to_check = [
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
    "brainsmith/steps/bert_steps.py",
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

for file_path in files_to_check:
    full_path = os.path.join(base_dir, file_path)
    if os.path.exists(full_path):
        with open(full_path, 'r') as f:
            lines = f.readlines()[:5]  # Read first 5 lines
        
        print(f"\n{file_path}:")
        print("First 5 lines:")
        for i, line in enumerate(lines, 1):
            print(f"  {i}: {line.rstrip()}")
        
        # Check if it has Microsoft copyright
        has_microsoft_copyright = any("Copyright" in line and "Microsoft" in line for line in lines)
        has_mit_license = any("MIT License" in line for line in lines)
        
        if has_microsoft_copyright and has_mit_license:
            print("  Status: Has Microsoft copyright and MIT License")
        elif has_microsoft_copyright:
            print("  Status: Has Microsoft copyright but check MIT License")
        else:
            print("  Status: NEEDS COPYRIGHT HEADER")
    else:
        print(f"\n{file_path}: FILE NOT FOUND")