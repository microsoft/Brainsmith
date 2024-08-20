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
import sys
import numpy as np

import argparse

import finn.util.data_packing as dpk
import numpy as np
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx
from qonnx.custom_op.registry import getCustomOp
from qonnx.util.basic import gen_finn_dt_tensor

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", default="")
# parse arguments
args = parser.parse_args()
build_dir = args.build_dir

# load original model containing all partitions (tidied up)
model_full = ModelWrapper(build_dir + "/output_create_partitions/intermediate_models/custom_step_tidy_up.onnx")

# load partition 0 to extract information about global input
model0 = ModelWrapper(build_dir + "/output_gen_ip_partition_0/intermediate_models/step_set_fifo_depths.onnx")
# load partition 2 to extract information about global output
model2 = ModelWrapper(build_dir + "/output_gen_ip_partition_2/intermediate_models/step_set_fifo_depths.onnx")


# Function to convert an int8 value to signed hexadecimal
def int8_to_signed_hex(val):
    # Convert int8 to signed hexadecimal representation
    hex_val = format(np.uint8(val), '02x') 
    return hex_val

def int32_to_signed_hex(val):
    # Convert int32 to signed hexadecimal representation
    hex_val = format(np.uint32(val), '08x')
    return hex_val

if __name__ == "__main__":
    refio_dir = build_dir + "/refio"
    os.makedirs(refio_dir, exist_ok=True)
    os.makedirs(refio_dir + "/int_results", exist_ok=True)
    refio_dir = str(os.path.abspath(refio_dir))
    # first, generate random input of appropriate shape
    top_iname = model_full.graph.input[0].name
    top_idt = model_full.get_tensor_datatype(top_iname)
    top_ishape = model_full.get_tensor_shape(top_iname)
    top_oname = model_full.graph.output[0].name
    np.random.seed(42)
    top_inp = gen_finn_dt_tensor(top_idt, top_ishape)
    # disable extra checks for faster golden ref generation
    os.environ["SANITIZE_QUANT_TENSORS"] = "0"
    # execute and get all intermediate outputs using checkpoint A
    golden_ret = execute_onnx(model_full, {top_iname: top_inp}, return_full_exec_context=True)
    # save top input, golden output and full context
    np.save(refio_dir + "/global_input.npy", top_inp)
    np.save(refio_dir + "/global_expected_output.npy", golden_ret[top_oname])
    np.savez(refio_dir + "/global_expected_full.npz", **golden_ret)

    # load the provided input data
    inp_data = top_inp
    batchsize = inp_data.shape[0]
    # query the parallelism-dependent folded input shape from the
    # node consuming the graph input (model0)
    inp_name = model0.graph.input[0].name
    inp_node = getCustomOp(model0.find_consumer(inp_name))
    inp_shape_folded = list(inp_node.get_folded_input_shape())
    inp_stream_width = inp_node.get_instream_width_padded()
    # fix first dimension (N: batch size) to correspond to input data
    # since FINN model itself always uses N=1
    inp_shape_folded[0] = batchsize
    inp_shape_folded = tuple(inp_shape_folded)
    inp_dtype = model0.get_tensor_datatype(inp_name)
    # now re-shape input data into the folded shape and do hex packing
    inp_data = inp_data.reshape(inp_shape_folded)
    inp_data_packed = dpk.pack_innermost_dim_as_hex_string(
        inp_data, inp_dtype, inp_stream_width, prefix="", reverse_inner=True
    )
    inp_data_packed = inp_data_packed.flatten()
    np.savetxt(refio_dir + "/global_input.dat", inp_data_packed, fmt="%s", delimiter="\n")

    # load expected output and calculate folded shape (model2)
    exp_out = golden_ret[top_oname]

    out_name = model2.graph.output[0].name
    out_node = getCustomOp(model2.find_producer(out_name))
    out_shape_folded = list(out_node.get_folded_output_shape())
    out_stream_width = out_node.get_outstream_width_padded()
    out_shape_folded[0] = batchsize
    out_shape_folded = tuple(out_shape_folded)
    out_dtype = model2.get_tensor_datatype(out_name)
    exp_out = exp_out.reshape(out_shape_folded)
    out_data_packed = dpk.pack_innermost_dim_as_hex_string(
        exp_out, out_dtype, out_stream_width, prefix="", reverse_inner=True
    )
    out_data_packed = out_data_packed.flatten()
    np.savetxt(
        refio_dir + "/global_expected_output.dat",
        out_data_packed,
        fmt="%s",
        delimiter="\n",
    )

    data = np.load(refio_dir + "/global_expected_full.npz")

    for array_name in data.files:
        # Get the array
        array = data[array_name]
        
        # Create the output file path
        out_path = os.path.join(refio_dir + "/int_results/", array_name + '.txt')
        
        # Open a text file to write the output
        with open(out_path, 'w') as f:
            # Write each int8 value as signed hexadecimal to the file
            for val in np.nditer(array):
                hex_val = int8_to_signed_hex(val)
                f.write(hex_val + '\n')

