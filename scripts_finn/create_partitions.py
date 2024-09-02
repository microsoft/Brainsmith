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
from custom_steps import custom_step_create_dataflow_partitions, custom_step_tidy_up
from finn.builder.build_dataflow_steps import step_qonnx_to_finn, step_tidy_up
import os

gen_hardware_steps = [
    step_qonnx_to_finn,
    step_tidy_up,
    custom_step_tidy_up,
    custom_step_create_dataflow_partitions,
]

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", default="")
parser.add_argument("--model_name", default="")

# parse arguments
args = parser.parse_args()
build_dir = args.build_dir
model_name = args.model_name

my_output_dir = build_dir + "/output_create_partitions"
model_file = os.path.abspath("../models/" + model_name)

cfg = build_cfg.DataflowBuildConfig(
    standalone_thresholds=True,
    steps=gen_hardware_steps,
    output_dir=my_output_dir,
    synth_clk_period_ns=5,
    fpga_part="xcv80-lsva4737-2MHP-e-S",
    generate_outputs=[],
)

print("Running FINN flow: create partitions")
build.build_dataflow_cfg(model_file, cfg)
