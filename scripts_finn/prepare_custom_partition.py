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

# configuration for thresholding layers

import argparse
import os
import json

from distutils.dir_util import copy_tree

import numpy as np
import onnx.helper as helper
from finn.transformation.fpgadataflow.convert_to_hw_layers import InferThresholdingLayer
from finn.transformation.fpgadataflow.prepare_ip import PrepareIP
from finn.transformation.fpgadataflow.specialize_layers import SpecializeLayers
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.custom_op.registry import getCustomOp
from qonnx.transformation.general import ApplyConfig, GiveReadableTensorNames, GiveUniqueNodeNames
from qonnx.util.basic import qonnx_make_model

parser = argparse.ArgumentParser()
parser.add_argument("--build_dir", default="")
parser.add_argument("--pe0", default="16")
parser.add_argument("--pe1", default="4")
# parse arguments
args = parser.parse_args()
build_dir = args.build_dir
pe0 = int(args.pe0)
pe1 = int(args.pe1)

fpga_part = "xcv80-lsva4737-2MHP-e-S"
clk_ns = 5.0

folding_config = {
    "Defaults": {},
    "Thresholding_rtl_0": {
        "PE": pe0,
    },
    "Thresholding_rtl_1": {
        "PE": pe1,
    },
}

idt = [DataType["INT21"], DataType["INT23"]] # This should be parametrized as well

def create_model_from_node(node, graph):
    node_inputs = list(filter(lambda x: x.name in node.input, graph.input))
    node_inputs += list(filter(lambda x: x.name in node.input, graph.output))
    node_inputs += list(filter(lambda x: x.name in node.input, graph.value_info))
    node_outputs = list(filter(lambda x: x.name in node.output, graph.output))
    node_outputs += list(filter(lambda x: x.name in node.output, graph.value_info))
    for attr in node.attribute:
        if attr.type == 5:
            subgraph = attr.g
            for subgraph_node in subgraph.node:
                subgraph_node_inputs = list(
                    filter(lambda x: x.name in subgraph_node.input, graph.value_info)
                )
                new_inps = list(filter(lambda x: x not in node_inputs, subgraph_node_inputs))
                node_inputs += new_inps
    node_graph = helper.make_graph(
        nodes=[node],
        name="thresh-single-node",
        inputs=node_inputs,
        outputs=node_outputs,
    )
    node_model = qonnx_make_model(node_graph)

    return ModelWrapper(node_model)


if __name__ == "__main__":
    config_path = os.path.abspath("../config/top_template.json")
    with open(config_path, "r") as f:
        top_model_config = json.load(f)
    top_model_file = top_model_config["TL"]["Model_name"]
    top_model = ModelWrapper(os.path.abspath("../models/" + top_model_file))
    # extract matrix height and matrix width
    matmul = top_model.get_nodes_by_op_type("MatMul")
    input_matrix = matmul[0].input[0]
    mh, mw = top_model.get_tensor_shape(input_matrix)[1:]
    top_model_config["TL"]["Mtrx_height"] = mh
    top_model_config["TL"]["Mtrx_width"] = mw
    # extract head size from shuffle nodes
    reshape = top_model.get_nodes_by_op_type("Reshape")
    reshape_out = reshape[0].output[0]
    headsize = top_model.get_tensor_shape(reshape_out)[3]
    top_model_config["TL"]["Head_size"] = headsize
    # extract activation bitwidth
    quant = top_model.get_nodes_by_op_type("Quant")
    bitwidth_tensor = quant[0].input[3]
    bitwidth = int(top_model.get_initializer(bitwidth_tensor))
    top_model_config["TL"]["Activation_width"] = bitwidth
    json_filename = os.path.abspath("../config/top_new.json")
    with open(json_filename, "w") as f:
        json.dump(top_model_config, f, indent=2)
    # extract generated threshold files
    model = ModelWrapper(build_dir + "/output_create_partitions/intermediate_models/partitions/partition_1.onnx")
    # model = model.transform(RoundAndClipThresholds())
    model = model.transform(InferThresholdingLayer())
    model = model.transform(SpecializeLayers(fpga_part))
    model = model.transform(GiveUniqueNodeNames())
    model = model.transform(GiveReadableTensorNames())
    model = model.transform(ApplyConfig(folding_config))
    thresholds = model.get_nodes_by_op_type("Thresholding_rtl")
    i = 0
    for thresh in thresholds:
        # store threshold values temporarily and reset after single node model
        threshold_vals = model.get_initializer(thresh.input[1])
        thresh_model = create_model_from_node(thresh, model.graph)
        thresh_model.set_initializer(thresh.input[1], threshold_vals)
        # set data types according to user setting and threshold values
        min_thresh = np.min(threshold_vals)
        if min_thresh < 0:
            tdt = DataType.get_smallest_possible(min_thresh)
        else:
            tdt = DataType.get_smallest_possible(np.max(threshold_vals))
        thresh_model.set_tensor_datatype(thresh_model.graph.node[0].input[0], idt[i])
        thresh_model.set_tensor_datatype(thresh_model.graph.node[0].input[1], idt[i])
        inst = getCustomOp(thresh_model.graph.node[0])
        inst.set_nodeattr("inputDataType", idt[i].name)
        inst.set_nodeattr("weightDataType", idt[i].name)
        # code generation
        thresh_model = thresh_model.transform(PrepareIP(fpga_part, clk_ns))
        # copy result
        inst = getCustomOp(thresh_model.graph.node[0])
        ip_path = inst.get_nodeattr("ip_path")
        out_path = build_dir + "/thresholds_partition_1"
        copy_tree(ip_path, out_path + "/%s" % thresh_model.graph.node[0].name)
        i += 1
