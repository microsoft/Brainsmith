############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import onnx
import argparse
import os
import shutil
import json
from onnxsim import simplify
import qonnx.custom_op.registry as registry
from qonnx.transformation.general import (
        SortCommutativeInputsInitializerLast,
        RemoveUnusedTensors,
        GiveReadableTensorNames,
        GiveUniqueNodeNames,
        ConvertDivToMul
)
from qonnx.transformation.remove import RemoveIdentityOps
from qonnx.transformation.extract_quant_scale_zeropt import ExtractQuantScaleZeroPt
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.infer_datatypes import InferDataTypes
from finn.builder.build_dataflow_config import DataflowOutputType
import finn.transformation.streamline as absorb
import finn.transformation.streamline.reorder as reorder
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import brainsmith.transformation.convert_to_hw_layers as to_bs_hw
from brainsmith.transformation.expand_norms import ExpandNorms

# Included for getting reference IO from model with head/tail removed
import finn.core.onnx_exec as oxe
from qonnx.util.basic import gen_finn_dt_tensor
from qonnx.core.datatype import DataType
import numpy as np


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
