############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

from brainsmith.blueprints.bert import (
    custom_step_cleanup,
    custom_step_qonnx2finn,
    custom_streamlining_step,
    custom_step_extract_loop_body,
    custom_step_loop_rolling,
    custom_step_infer_hardware,
    custom_step_shell_metadata_handover
)

# Debugging
from finn.transformation.fpgadataflow.set_exec_mode import SetExecMode
from finn.transformation.fpgadataflow.prepare_cppsim import PrepareCppSim
from finn.transformation.fpgadataflow.compile_cppsim import CompileCppSim
from finn.transformation.fpgadataflow.prepare_rtlsim import PrepareRTLSim
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

BUILD_FINNLOOP_STEPS = [
        custom_step_cleanup,
        custom_step_qonnx2finn,
        #custom_step_generate_reference_io,
        custom_streamlining_step,
        custom_step_extract_loop_body,
        custom_step_loop_rolling,
        custom_step_infer_hardware,
        #step_create_dataflow_partition,
        #step_specialize_layers,
        #step_target_fps_parallelization,
        #step_apply_folding_config,
        #step_minimize_bit_width,
        #step_generate_estimate_reports,
        #step_hw_codegen,
        #step_hw_ipgen,
        #step_measure_rtlsim_performance,
        #step_set_fifo_depths,
        #step_create_stitched_ip,
        #custom_step_shell_metadata_handover,
    ]
