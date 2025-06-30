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
from brainsmith.kernels.softmax.hwsoftmax import HWSoftmax
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import CppBuilder
from brainsmith.plugin.decorators import backend

@backend(
    name="HWSoftmaxHLS",
    kernel="HWSoftmax",
    backend_type="hls",
    description="HLS implementation of HWSoftmax",
    author="shane-fleming",
    version="1.0.0"
)
class HWSoftmax_hls(HWSoftmax, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        my_attrs = {}
        my_attrs.update(HWSoftmax.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "softmax.hpp"',
            '#include "utils.hpp"',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("SIMD")
        idtype = self.get_input_datatype()
        odtype = self.get_output_datatype()
        w = self.get_nodeattr("ifm_dim")[-1]
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd};
            constexpr unsigned  W = {w};
            using  TI = {idtype.get_hls_datatype_str()};
            using  F = float;
           """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
                static hls::stream<hls::vector<TI,SIMD>>  src0;
                static hls::stream<hls::vector<float,SIMD>>  dst0;

                move(in0_V, src0);
                static SoftMax<TI, float, W, SIMD> sm_inst;
                sm_inst.execute(src0, dst0);
                move(dst0, out0_V);
        """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<float,SIMD>> &out0_V
                )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            f"""
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate  variable=in0_V compact=bit
            #pragma HLS aggregate  variable=out0_V compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        exp_ishape = self.get_normal_input_shape()
        exp_oshape = self.get_normal_output_shape()
        folded_ishape = self.get_folded_input_shape()
        export_idt = self.get_input_datatype()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")


        inp = context[node.input[0]]
        inp = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)        

        if mode == "cppsim":
            # # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # # load output npy file
            super().npy_to_dynamic_output(context)
        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits    
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            #rtlsim_output = self.rtlsim(sim, rtlsim_inp)
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
        builder.append_includes(f"-I{kernel_dir}/hls")
        builder.append_includes(f"-I{utils_dir}")
        builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
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
            '-Wl,-rpath,"$VITIS_PATH/lnx64/tools/fpo_v7_1" -L$VITIS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel'
        )
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

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
            static hls::stream<hls::vector<TI,SIMD>>  in0_V;
            static hls::stream<hls::vector<float,SIMD>>  out0_V;

            npy2vectorstream<TI, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();
            static SoftMax<TI, float, W, SIMD> sm_inst;

            while(out0_V.size() != stream_size){{
                sm_inst.execute(in0_V, out0_V);
            }}

            vectorstream2npy<float, float, SIMD>(out0_V,{oshape_str}, "{path}/output_0.npy");
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
    
    def ipgen_extra_includes(self):
        """Add kernel-specific include paths."""
        import os
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        return f"-I{kernel_dir}/hls -I{utils_dir}"
