# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

from brainsmith.jobs.bert.bert_steps import (
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_generate_reference_io,
        custom_step_cleanup,
        custom_step_infer_hardware,
        custom_streamlining_step,
        custom_step_qonnx2finn,
)

from finn.builder.build_dataflow_steps import (
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    step_create_stitched_ip,
    step_measure_rtlsim_performance
)

BERT_STEPS = [
        # Cleanup and custom graph surgery
        custom_step_cleanup,
        custom_step_remove_head,
        custom_step_remove_tail,
        custom_step_qonnx2finn,

        custom_step_generate_reference_io,
        custom_streamlining_step,
        custom_step_infer_hardware,
        step_create_dataflow_partition,
        step_specialize_layers,
        step_target_fps_parallelization,
        step_apply_folding_config,
        step_minimize_bit_width,
        step_generate_estimate_reports,
        step_hw_codegen,
        step_hw_ipgen,
        step_measure_rtlsim_performance,
        step_set_fifo_depths,
        step_create_stitched_ip,
    ]
