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

from distutils.dir_util import copy_tree

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", default="")
parser.add_argument("--model_name", default="partition_0")
parser.add_argument("--dev_name", default="xcv80-lsva4737-2MHP-e-S")

# parse arguments
args = parser.parse_args()
build_dir = args.build_dir
model_name = args.model_name
dev_name = args.dev_name

if __name__ == "__main__":
    model = ModelWrapper(build_dir + "/output_gen_ip_%s/intermediate_models/step_set_fifo_depths.onnx" % model_name)
    mvaus = model.get_nodes_by_op_type("MVAU_hls")
    mvaus += model.get_nodes_by_op_type("MVAU_rtl")
    i = 0
    for mvau in mvaus:
        inst = getCustomOp(mvau)
        # extract accumulator width
        acc_width = inst.get_nodeattr("accDataType")
        inst.set_nodeattr("SIMD", 1)
        inst.code_generation_ipgen(model, dev_name, 3.3)
        ip_path = inst.get_nodeattr("ip_path")
        out_path = build_dir + "/mvau_weight_files_%s" % model_name
        copy_tree(ip_path, out_path + "/%s" % mvau.name)
        acc_filename = out_path + "/" + mvau.name + "/accumulator_width.txt"
        with open(acc_filename, "w") as f:
            f.write(acc_width)
        i += 1

    thresholds = model.get_nodes_by_op_type("Thresholding_hls")
    thresholds += model.get_nodes_by_op_type("Thresholding_rtl")
    i = 0
    for thresh in thresholds:
        inst = getCustomOp(thresh)
        ip_path = inst.get_nodeattr("ip_path")
        out_path = build_dir + "/mvau_weight_files_%s" % model_name
        copy_tree(ip_path, out_path + "/%s" % thresh.name)
        i += 1
