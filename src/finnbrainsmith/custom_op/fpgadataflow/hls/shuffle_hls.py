# Copyright (C) 2024, Advanced Micro Devices, Inc.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.
#
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
#
# * Neither the name of FINN nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os

from finn.custom_op.fpgadataflow import templates
from finnbrainsmith.custom_op.fpgadataflow.brainsmith_hlsbackend import BS_HLSBackend
from finnbrainsmith.custom_op.fpgadataflow.shuffle import Shuffle 
from finn.util.basic import CppBuilder

class Shuffle_hls(Shuffle, BS_HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        return Shuffle.get_nodeattr_types(self) | BS_HLSBackend.get_nodeattr_types(self)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "input_gen.hpp"',
            '#include <ap_int.h>',
            '#include <hls_vector.h>',
            '#include <hls_stream.h>',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("simd")
        dtype = self.get_input_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd}; 
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def docompute(self):
        simd = self.get_nodeattr("simd")
        out_reshaped = self.get_nodeattr("out_reshaped")
        loop_coeffs = [x/simd for x in self.get_nodeattr("loop_coeffs")]
        interleaved = [int(item) for pair in zip(loop_coeffs, out_reshaped) for item in pair] 
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
	    hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(src, src0);
	    input_gen<-1, {','.join(map(str,interleaved))}>(src0, dst0);
	    move(dst0, dst);
            """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name} (
                hls::stream<TV> &in0_{self.hls_sname()},
	        hls::stream<TV> &out_{self.hls_sname()}
            )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            f"""
            #pragma HLS interface AXIS port=src
            #pragma HLS interface AXIS port=dst
	    #pragma HLS aggregate  variable=src compact=bit
	    #pragma HLS aggregate  variable=dst compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]


    def execute_node(self, context, graph):
        raise NotImplementedError("This function is not yet immplemented.")


    def compile_singlenode_code(self):
        """
        Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$FINN_ROOT/src/finn/qnn-data/cpp")
        builder.append_includes("-I$FINN_ROOT/deps/cnpy/")
        builder.append_includes("-I$FINN_ROOT/deps/finn-hlslib")
        builder.append_includes("-I$FINN_ROOT/deps/finnbrainsmith/hlslib_extensions")
        builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$FINN_ROOT/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.append_includes(
            '-fno-builtin -fno-inline -Wl,-rpath,"$HLS_PATH/lnx64/lib/csim" -L$HLS_PATH/lnx64/lib/csim -lhlsmc++-GCC46'
        )
        builder.append_includes( #TODO: [STF]I have a feeling this should/could be removed for shuffle as it's all FP related?
            "-L$HLS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel"
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
            static hls::stream<hls::vector<TO,SIMD>>  out_V;

            npy2vectorstream<TI, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            // TODO: Call Kernel

            vectorstream2npy<TO, float, SIMD>(out_V,{oshape_str}, "{path}/output.npy");
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

    def prepare_rtlsim(self):
        # this node currently does not support rtlsim
        raise NotImplementedError("Shuffle_hls does not yet support rtlsim")
