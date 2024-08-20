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
import os
from distutils.dir_util import copy_tree

import finn.transformation.fpgadataflow.convert_to_hw_layers as to_hw
import finn.transformation.streamline as absorb
import finn.util.data_packing as dpk
import numpy as np
from finn.transformation.fpgadataflow.create_stitched_ip import CreateStitchedIP
from finn.transformation.streamline.round_thresholds import RoundAndClipThresholds
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.create_generic_partitions import PartitionFromDict
from qonnx.util.basic import gen_finn_dt_tensor


def custom_step_tidy_up(model, cfg):
    # Absorb negative bias from Add into MultiThreshold nodes
    # and round and clip threshold values
    model = model.transform(absorb.AbsorbSignBiasIntoMultiThreshold())
    model = model.transform(RoundAndClipThresholds())
    return model


def custom_step_create_dataflow_partitions(model, cfg):
    partition_dict = {
        0: [0, 1, 2, 3, 4, 5],
        1: [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17],
        2: [18, 19, 20, 21, 22, 23, 24, 25, 26, 27],
    }
    model.transform(
        PartitionFromDict(
            partition_dict, partition_dir=cfg.output_dir + "/intermediate_models/partitions"
        )
    )
    return model


def custom_step_duplicate_streams(model, cfg):
    # infer duplicate streams
    model = model.transform(to_hw.InferDuplicateStreamsLayer())
    model = model.transform(to_hw.InferAddStreamsLayer())
    model = model.transform(to_hw.InferStreamingEltwise())
    return model


def custom_step_stitched_ip_partition_0(model, cfg):
    stitched_ip_dir = cfg.output_dir + "/stitched_ip"
    model = model.transform(
        CreateStitchedIP(
            cfg._resolve_fpga_part(),
            cfg.synth_clk_period_ns,
            ip_name="partition_0",
            vitis=cfg.stitched_ip_gen_dcp,
            signature=cfg.signature,
            early_exit=True,
        )
    )
    copy_tree(model.get_metadata_prop("vivado_stitch_proj"), stitched_ip_dir)
    print("Vivado stitched IP written into " + stitched_ip_dir)

    return model


def custom_step_stitched_ip_partition_2(model, cfg):
    stitched_ip_dir = cfg.output_dir + "/stitched_ip"
    model = model.transform(
        CreateStitchedIP(
            cfg._resolve_fpga_part(),
            cfg.synth_clk_period_ns,
            ip_name="partition_2",
            vitis=cfg.stitched_ip_gen_dcp,
            signature=cfg.signature,
            early_exit=True,
        )
    )
    copy_tree(model.get_metadata_prop("vivado_stitch_proj"), stitched_ip_dir)
    print("Vivado stitched IP written into " + stitched_ip_dir)

    return model
