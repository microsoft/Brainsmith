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

from finn import xsi
finnxsi = xsi if xsi.is_available() else None

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import make_build_dir

from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.core.plugins import backend

@backend(
    name="LayerNorm_rtl",
    kernel="LayerNorm",
    language="rtl",
    author="FINN"
        )
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
        rtlsrc = os.environ["BSMITH_DIR"] + "/deps/finn/finn-rtllib/layernorm/"
        template_path = rtlsrc + "layernorm_wrapper_template.v"
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
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + ".v"),
            "w",
        ) as f:
            f.write(template.replace("$FORCE_BEHAVIORAL$", str(0)))
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_sim.v"),
            "w",
        ) as f:
            f.write(template.replace("$FORCE_BEHAVIORAL$", str(1)))

        sv_files = ["layernorm.sv", "queue.sv", "accuf.sv", "binopf.sv", "rsqrtf.sv"]
        for sv_file in sv_files:
            shutil.copy(rtlsrc + sv_file, code_gen_dir)
        # set ipgen_path and ip_path so that HLS-Synth transformation
        # and stich_ip transformation do not complain
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def get_rtl_file_list(self, abspath=False):
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = rtlsrc = os.environ["BSMITH_DIR"] + "/deps/finn/finn-rtllib/layernorm/"
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            rtllib_dir + "layernorm.sv",
            rtllib_dir + "queue.sv",
            rtllib_dir + "accuf.sv",
            rtllib_dir + "binopf.sv",
            rtllib_dir + "rsqrtf.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + ".v",
        ]
        return verilog_files

    def code_generation_ipi(self, behavioral=False):
        """Constructs and returns the TCL for node instantiation in Vivado IPI."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        sourcefiles = [
            "layernorm.sv",
            "queue.sv",
            "accuf.sv",
            "binopf.sv",
            "rsqrtf.sv",
        ]
        if behavioral is True:
            sourcefiles.append(self.get_nodeattr("gen_top_module") + "_sim.v")
        else:
            sourcefiles.append(self.get_nodeattr("gen_top_module") + ".v")

        sourcefiles = [os.path.join(code_gen_dir, f) for f in sourcefiles]

        cmd = []
        for f in sourcefiles:
            cmd += ["add_files -norecurse %s" % (f)]
        cmd += [
            "create_bd_cell -type module -reference %s %s"
            % (self.get_nodeattr("gen_top_module"), self.onnx_node.name)
        ]
        return cmd

    def prepare_rtlsim(self):
        """Creates a xsi emulation library for the RTL code generated
        for this node, sets the rtlsim_so attribute to its path."""

        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
        rtllib_dir = rtlsrc = os.environ["BSMITH_DIR"] + "/deps/finn/finn-rtllib/layernorm/"

        verilog_files = [
            rtllib_dir + "layernorm.sv",
            rtllib_dir + "queue.sv",
            rtllib_dir + "accuf.sv",
            rtllib_dir + "binopf.sv",
            rtllib_dir + "rsqrtf.sv",
            code_gen_dir + self.get_nodeattr("gen_top_module") + "_sim.v",
        ]

        single_src_dir = make_build_dir("rtlsim_" + self.onnx_node.name + "_")
        trace_file = self.get_nodeattr("rtlsim_trace")
        debug = not (trace_file is None or trace_file == "")
        ret = finnxsi.compile_sim_obj(
            self.get_verilog_top_module_name(), verilog_files, single_src_dir, debug
        )
        # save generated lib filename in attribute
        self.set_nodeattr("rtlsim_so", ret[0] + "/" + ret[1])

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        if mode == "cppsim":
            LayerNorm.execute_node(self, context, graph)
        elif mode == "rtlsim":
            RTLBackend.execute_node(self, context, graph)
