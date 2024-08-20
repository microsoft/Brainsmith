# Copyright (C) 2024, Advanced Micro Devices, Inc. All rights reserved.
#
# This file is subject to the Xilinx Design License Agreement located
# in the LICENSE.md file in the root directory of this repository.
#
# This file contains confidential and proprietary information of Xilinx, Inc.
# and is protected under U.S. and international copyright and other
# intellectual property laws.
#
# DISCLAIMER
# This disclaimer is not a license and does not grant any rights to the materials
# distributed herewith. Except as otherwise provided in a valid license issued to
# you by Xilinx, and to the maximum extent permitted by applicable law: (1) THESE
# MATERIALS ARE MADE AVAILABLE "AS IS" AND WITH ALL FAULTS, AND XILINX HEREBY
# DISCLAIMS ALL WARRANTIES AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY,
# INCLUDING BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NONINFRINGEMENT, OR
# FITNESS FOR ANY PARTICULAR PURPOSE; and (2) Xilinx shall not be liable (whether
# in contract or tort, including negligence, or under any other theory of
# liability) for any loss or damage of any kind or nature related to, arising
# under or in connection with these materials, including for any direct, or any
# indirect, special, incidental, or consequential loss or damage (including loss
# of data, profits, goodwill, or any type of loss or damage suffered as a result
# of any action brought by a third party) even if such damage or loss was
# reasonably foreseeable or Xilinx had been advised of the possibility of the
# same.
#
# CRITICAL APPLICATIONS
# Xilinx products are not designed or intended to be fail-safe, or for use in
# any application requiring failsafe performance, such as life-support or safety
# devices or systems, Class III medical devices, nuclear facilities, applications
# related to the deployment of airbags, or any other applications that could lead
# to death, personal injury, or severe property or environmental damage
# (individually and collectively, "Critical Applications"). Customer assumes the
# sole risk and liability of any use of Xilinx products in Critical Applications,
# subject only to applicable laws and regulations governing limitations on product
# liability.
#
# THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS PART OF THIS FILE AT ALL TIMES.

import argparse

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from custom_steps import (
    custom_step_duplicate_streams,
    custom_step_stitched_ip_partition_0,
    custom_step_stitched_ip_partition_2,
)
from finn.builder.build_dataflow_steps import (
    step_apply_folding_config,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_minimize_bit_width,
    step_set_fifo_depths,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_tidy_up,
)

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", default="")
parser.add_argument("--model_name", default="partition_0")
parser.add_argument("--version_name", default="")
# parse arguments
args = parser.parse_args()
build_dir = args.build_dir
model_name = args.model_name
version_name = args.version_name

# check if argument is allowed
assert model_name in [
    "partition_0",
    "partition_2",
], "Cannot generate FINN IP automatically for this part of the graph."

if model_name == "partition_0":
    stitching_step = custom_step_stitched_ip_partition_0
    prefix = "p0_"
elif model_name == "partition_2":
    stitching_step = custom_step_stitched_ip_partition_2
    prefix = "p2_"

gen_hardware_steps = [
    step_tidy_up,
    custom_step_duplicate_streams,
    step_convert_to_hw,
    step_create_dataflow_partition,
    step_specialize_layers,
    step_target_fps_parallelization,
    step_apply_folding_config,
    step_minimize_bit_width,
    step_generate_estimate_reports,
    step_hw_codegen,
    step_hw_ipgen,
    step_set_fifo_depths,
    stitching_step,
]

my_output_dir = build_dir + "/output_gen_ip_" + model_name + version_name
model_file = build_dir + "/output_create_partitions/intermediate_models/partitions/" + model_name + ".onnx"

cfg = build_cfg.DataflowBuildConfig(
    auto_fifo_depths=False,
    steps=gen_hardware_steps,
    output_dir=my_output_dir,
    prefix_node_names=prefix,
    synth_clk_period_ns=3.3,
    fpga_part="xcv80-lsva4737-2MHP-e-S",
    enable_build_pdb_debug=True,
    standalone_thresholds=True,
    folding_config_file="../config/folding_%s.json" % model_name,
    specialize_layers_config_file="../config/specialize_layers_%s.json" % model_name,
    verbose=True,
    rtlsim_batch_size=10,
    generate_outputs=[
        build_cfg.DataflowOutputType.ESTIMATE_REPORTS,
        build_cfg.DataflowOutputType.STITCHED_IP,
        build_cfg.DataflowOutputType.RTLSIM_PERFORMANCE,
    ],
)

print("Running FINN flow: generate hardware IP for " + model_name)
build.build_dataflow_cfg(model_file, cfg)
