#!/usr/bin/env python3
"""Check if all FINN build steps are registered in BrainSmith."""

from brainsmith.core.plugins import list_steps

# List all registered steps
steps = list_steps()
print(f'Total steps registered: {len(steps)}')
print('\nRegistered steps:')
for step in sorted(steps):
    print(f'  - {step}')

# FINN default build steps
finn_default_steps = [
    "step_qonnx_to_finn",
    "step_tidy_up",
    "step_streamline",
    "step_convert_to_hw",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "step_measure_rtlsim_performance",
    "step_out_of_context_synthesis",
    "step_synthesize_bitfile",
    "step_make_driver",
    "step_deployment_package",
]

print(f'\nFINN default build steps: {len(finn_default_steps)}')

# Check which ones are missing
missing_steps = []
for step in finn_default_steps:
    # Remove 'step_' prefix as BrainSmith might register without it
    step_name = step.replace('step_', '')
    # Also check with finn: prefix
    finn_step_name = f'finn:{step_name}'
    if step_name not in steps and step not in steps and finn_step_name not in steps:
        missing_steps.append(step)

if missing_steps:
    print(f'\n✗ Missing FINN build steps ({len(missing_steps)}):')
    for step in missing_steps:
        print(f'  - {step}')
else:
    print('\n✓ All FINN default build steps are registered!')