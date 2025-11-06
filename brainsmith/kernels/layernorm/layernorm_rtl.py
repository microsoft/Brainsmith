############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Jakoba Petri-Koenig <jakoba.petri-koenig@amd.com>
############################################################################

import os
import shutil

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.registry import backend


@backend(name='LayerNorm_rtl', target_kernel='brainsmith:LayerNorm', language='rtl')
class LayerNorm_rtl(LayerNorm, RTLBackend):
    """RTL backend implementation for LayerNorm kernel.

    Generates RTL code for hardware synthesis of LayerNorm operations.

    Metadata for registry (namespace-based component registry):
    - target_kernel: Which kernel this backend implements
    - language: Backend language (hls/rtl/etc)
    """

    # Metadata for namespace-based registry
    target_kernel = 'brainsmith:LayerNorm'
    language = 'rtl'

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        return my_attrs


    def generate_hdl(self, model, fpgapart, clk):
        # wrapper file is in the same directory as this file
        rtlsrc = os.path.dirname(os.path.abspath(__file__))
        template_path = rtlsrc + "/layernorm_wrapper_template.v"
        simd = self.get_nodeattr("SIMD")
        topname = self.get_verilog_top_module_name()
        code_gen_dict = {
            "$N$": int(self.get_normal_input_shape()[-1]),
            "$SIMD$": int(simd),
            "$TOP_MODULE_NAME$": topname,
        }
        # save top module name so we can refer to it after this node has been renamed
        # (e.g. by GiveUniqueNodeNames(prefix) during MakeZynqProject)
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # apply code generation to templates
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        with open(template_path, "r") as f:
            template = f.read()
        for key in code_gen_dict:
            template = template.replace(key, str(code_gen_dict[key]))

        with open(
            os.path.join(code_gen_dir, self.get_verilog_top_module_name() + ".v"),
            "w",
        ) as f:
            f.write(template)

        sv_files = ["layernorm.sv", "queue.sv", "accuf.sv", "binopf.sv", "rsqrtf.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + "/" + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.dirname(os.path.abspath(__file__))
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "/layernorm.sv",
            rtllib_dir + "/queue.sv",
            rtllib_dir + "/accuf.sv",
            rtllib_dir + "/binopf.sv",
            rtllib_dir + "/rsqrtf.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def code_generation_ipi(self):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "layernorm.sv",
            "queue.sv",
            "accuf.sv",
            "binopf.sv",
            "rsqrtf.sv",
            self.get_nodeattr("gen_top_module") + ".v",
        ]

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LayerNorm.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
