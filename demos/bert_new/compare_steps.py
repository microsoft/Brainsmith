#!/usr/bin/env python3
"""Compare step sequences between bert_new and bert_direct demos."""

print("Step sequences comparison:")
print("="*80)

# bert_direct steps (working)
bert_direct_steps = [
    "cleanup_step",
    "remove_head_step",
    "remove_tail_step",
    "qonnx_to_finn_step",
    "generate_reference_io_cached_step",  # Note: cached version
    "streamlining_step",
    "infer_hardware_step",
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    "step_measure_rtlsim_performance",
    "constrain_folding_and_set_pumped_compute_step",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "shell_metadata_handover_step",
]

# bert_new steps (from blueprint)
bert_new_steps = [
    # legacy_preproc
    "cleanup_step",
    "remove_head_step",
    "remove_tail_step",
    "qonnx_to_finn_step",
    "generate_reference_io_step",  # Note: not cached
    "streamlining_step",
    "infer_hardware_step",
    # standard FINN pipeline
    "step_create_dataflow_partition",
    "step_specialize_layers",
    "step_target_fps_parallelization",
    "step_apply_folding_config",
    "step_minimize_bit_width",
    "step_generate_estimate_reports",
    "step_hw_codegen",
    "step_hw_ipgen",
    # legacy_postproc
    "step_measure_rtlsim_performance",
    "constrain_folding_and_set_pumped_compute_step",
    "step_set_fifo_depths",
    "step_create_stitched_ip",
    "shell_metadata_handover_step",
]

print(f"bert_direct: {len(bert_direct_steps)} steps")
print(f"bert_new:    {len(bert_new_steps)} steps")
print()

# Check differences
differences = []
for i, (direct, new) in enumerate(zip(bert_direct_steps, bert_new_steps)):
    if direct != new:
        differences.append((i+1, direct, new))

if differences:
    print("Step differences found:")
    for pos, direct, new in differences:
        print(f"  Position {pos}: {direct} vs {new}")
else:
    print("✅ Step sequences are identical (except cached vs non-cached reference IO)")

print()
print("Key observations:")
print("1. Both have exactly 20 steps in the same order")
print("2. Only difference is generate_reference_io_cached_step vs generate_reference_io_step")
print("3. This confirms the step sequence is correct in bert_new")