############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
############################################################################

import numpy as np
import os

from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import CppBuilder
from brainsmith.kernels.layernorm.layernorm import LayerNorm
from brainsmith.registry import backend

@backend(
    name="LayerNormHLS", 
    kernel="LayerNorm",
    language="hls",
    description="HLS backend for LayerNorm kernel",
    author="Shane Fleming",
)
class LayerNorm_hls(LayerNorm, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        my_attrs.update(LayerNorm.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "layernorm.hpp"',
            '#include "bs_utils.hpp"'
        ]

    def defines(self, var):
        idtype = self.get_input_datatype()
        odtype = self.get_output_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned SIMD = {self.get_nodeattr('SIMD')};",
            f"constexpr unsigned W = {self.get_nodeattr('ifm_dim')[-1]};",
            f"constexpr float epsilon = {self.get_nodeattr('epsilon')};",
            f"using TI = {idtype.get_hls_datatype_str()};",
            f"using TO = {odtype.get_hls_datatype_str()};"
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
                layernorm_pipeline<TI, TO, W, SIMD>(epsilon, in0_V, out0_V);
            """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<TO,SIMD>> &out0_V
                )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            f"#pragma HLS interface AXIS port=in0_V",
            f"#pragma HLS interface AXIS port=out0_V",
            f"#pragma HLS aggregate variable=in0_V compact=bit",
            f"#pragma HLS aggregate variable=out0_V compact=bit",
            f"#pragma HLS interface ap_ctrl_none port=return",
            f"#pragma HLS dataflow disable_start_propagation",
        ]

    def execute_node(self, context, graph):
        # Get the configured execution mode
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        folded_ishape = self.get_folded_input_shape()
        export_idt = self.get_input_datatype()

        # Generate input
        inp = context[node.input[0]]
        inp = inp.reshape(folded_ishape)
        inp = inp.astype(np.float32)

        if mode == "python":
            self._execute_node_python(context, graph)
        elif mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)
            # Execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # Load output npy file
            super().npy_to_dynamic_output(context)
        elif mode == "rtlsim":
            # Generate & format input
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            # Setup RTLsim
            sim = self.get_rtlsim()
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)
            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs":{"out0": []}
                    }
            self.rtlsim_multi_io(sim, io_dict)
            out = io_dict["outputs"]["out0"]

            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output_0.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(out, out_npy_path, odt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(f"Unsupported execution mode: {mode}")

    def get_exp_cycles(self):
        oshape = self.get_normal_output_shape()
        return  int(oshape[-1] + 68 + 4)

    def code_generation_cppsim(self, model):
        """Generates c++ code for simulation (cppsim)."""
        self.code_gen_dict["$READNPYDATA$"] = [""]
        self.code_gen_dict["$DATAOUTSTREAM$"] = [""]
        self.code_gen_dict["$STREAMDECLARATIONS$"] = [""]
        node = self.onnx_node
        path = self.get_nodeattr("code_gen_dir_cppsim")
        self.code_gen_dict["$AP_INT_MAX_W$"] = [str(self.get_ap_int_max_w())]
        self.generate_params(model, path)
        self.global_includes()
        self.defines("cppsim")
        self.pragmas()
        oshape = self.get_folded_output_shape()
        oshape_str = str(oshape).replace("(", "{").replace(")", "}")
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            static hls::stream<hls::vector<TI,SIMD>> in0_V;
            static hls::stream<hls::vector<TO,SIMD>> out0_V;

            npy2vectorstream<TI, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            while(out0_V.size() != stream_size){{
                layernorm_pipeline<TI, TO, W, SIMD>(epsilon, in0_V, out0_V);
            }}

            vectorstream2npy<TO, float, SIMD>(out0_V, {oshape_str}, "{path}/output_0.npy");
            """
        ]
        self.save_as_npy()

        template = templates.docompute_template

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim") + f"/execute_{node.op_type}.cpp"
        with open(code_gen_dir, "w") as f:
            for key in self.code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(self.code_gen_dict[key])
                template = template.replace(key, code_gen_line)
            f.write(template)

    def compile_singlenode_code(self):
        """Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$BSMITH_DIR/deps/finn/src/finn/qnn-data/cpp")
        builder.append_includes("-I$BSMITH_DIR/deps/cnpy/")
        builder.append_includes("-I$BSMITH_DIR/deps/finn-hlslib")
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        builder.append_includes(f"-I{kernel_dir}")
        builder.append_includes(f"-I{utils_dir}")
        #builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$BSMITH_DIR/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.append_includes(
            '-fno-builtin -fno-inline -Wl,-rpath,"$VITIS_PATH/lnx64/lib/csim" -L$VITIS_PATH/lnx64/lib/csim -lhlsmc++-GCC46'
        )
        builder.append_includes(
            "-L$VITIS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel"
        )
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates c++ code and tcl script for IP generation."""
        # Call parent implementation which handles the basic HLS IP generation
        super().code_generation_ipgen(model, fpgapart, clk)
        
    def ipgen_extra_includes(self):
        """Add kernel-specific include paths."""
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        return f"-I{kernel_dir} -I{utils_dir}"
        
    def generate_params(self, model, path):
        """Generate any parameters needed by the kernel."""
        # LayerNorm doesn't need parameter files
        pass