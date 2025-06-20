############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Josh Monson <joshmonson@microsoft.com>
############################################################################


import numpy as np
import os

from brainsmith.libraries.kernels.utils import brainsmith_templates
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.libraries.kernels.crop.crop import Crop
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import CppBuilder

class Crop_hls(Crop, HLSBackend):
    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        return Crop.get_nodeattr_types(self) | HLSBackend.get_nodeattr_types(self)

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "crop.hpp"',
            '#include <bs_utils.hpp>',
            '#include <ap_int.h>',
            '#include <hls_vector.h>',
            '#include <hls_stream.h>',
            '#include <iostream>',
        ]

    def defines(self, var):
        simd = self.get_nodeattr("simd")
        dtype = self.get_input_datatype()
        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD   = {simd};
            constexpr unsigned  H      = {self.get_nodeattr("height")};
            constexpr unsigned  W      = {self.get_nodeattr("width")/simd};
            constexpr unsigned  CF     = {self.get_nodeattr("channel_fold")};
            constexpr unsigned  CROP_N = {self.get_nodeattr("crop_north")};
            constexpr unsigned  CROP_E = {self.get_nodeattr("crop_east")};
            constexpr unsigned  CROP_S = {self.get_nodeattr("crop_south")};
            constexpr unsigned  CROP_W = {self.get_nodeattr("crop_west")};
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def docompute(self):
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            crop< H, W,	CF,	CROP_N, CROP_E, CROP_S, CROP_W, TV>(src0, dst0);
            move(dst0, out0_V);
            """
        ]

    def blackboxfunction(self):
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name} (
                hls::stream<TV> &in0_V,
                hls::stream<TV> &out0_V
            )
            """
        ]

    def pragmas(self):
        self.code_gen_dict["$PRAGMAS$"] = [
            f"""
            #pragma HLS interface AXIS port=in0_V
            #pragma HLS interface AXIS port=out0_V
            #pragma HLS aggregate variable=in0_V compact=bit
            #pragma HLS aggregate variable=out0_V compact=bit

            #pragma HLS interface ap_ctrl_none port=return
            #pragma HLS dataflow disable_start_propagation
            """
        ]

    def execute_node(self, context, graph):
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node
        folded_ishape = self.get_folded_input_shape()
        export_dt = self.get_input_datatype()

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

        inp = context[node.input[0]]
        inp = inp.reshape(folded_ishape)
        np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            # execute the precompiled model
            super().exec_precompiled_singlenode_model()
            # Load output npy file
            super().npy_to_dynamic_output(context)
        elif mode =="rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                f"{code_gen_dir}/input_0.npy", export_dt, nbits
            )
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            io_dict = {
                "inputs" : {"in0" : rtlsim_inp},
                "outputs" : {"out0" : []}
            }
            self.rtlsim_multi_io(sim, io_dict)

            out = io_dict["outputs"]["out0"]
            target_bits = export_dt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(out, out_npy_path, export_dt, out_shape, packed_bits, target_bits)

            # load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32,).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(f"Unsupported execution mode: {mode}")

    def compile_singlenode_code(self):
        """
        Builds the bash script for compilation using the CppBuilder from
        finn.util.basic and executes the script to produce the executable
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()
        # to enable additional debug features please uncommand the next line
        # builder.append_includes("-DDEBUG")
        builder.append_includes("-I$BSMITH_DIR/deps/finn/src/finn/qnn-data/cpp")
        builder.append_includes("-I$BSMITH_DIR/deps/cnpy/")
        builder.append_includes("-I$BSMITH_DIR/deps/finn-hlslib")
        builder.append_includes("-I$BSMITH_DIR/brainsmith/hw_kernels/hls")
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$BSMITH_DIR/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
        builder.append_includes(
            '-fno-builtin -fno-inline -Wl,-rpath,"$VITIS_PATH/lnx64/lib/csim" -L$VITIS_PATH/lnx64/lib/csim -lhlsmc++-GCC46'
        )
        builder.append_includes( #TODO: [STF]I have a feeling this should/could be removed for shuffle as it's all FP related?
            "-L$VITIS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr -lIp_floating_point_v7_1_bitacc_cmodel"
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

        simd = self.get_nodeattr("simd")


        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            static hls::stream<TV>  in0_V;
            static hls::stream<TV>  out0_V;
            std::cout << "reading in data" << std::endl;
            npy2vectorstream<TE, float, SIMD>("{path}/input_0.npy", in0_V);

            std::cout << "computing" << std::endl;
            unsigned in0_size = in0_V.size();
            for (int i = 0; i < in0_size; i++)
                crop< H, W,	CF,	CROP_N, CROP_E, CROP_S, CROP_W, TV>(in0_V, out0_V);
            std::cout << "writing out data " << out0_V.size() << std::endl;
            vectorstream2npy<TE, float, SIMD>(out0_V,{oshape_str}, "{path}/output_0.npy");
            std::cout << "done" << std::endl;
            std::cout << "in0_V size: " << in0_V.size() << std::endl;
            std::cout << "out0_V size: " << out0_V.size() << std::endl;
            """
        ]
        self.save_as_npy()

        template = brainsmith_templates.docompute_template

        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim") + f"/execute_{node.op_type}.cpp"
        with open(code_gen_dir, "w") as f:
            for key in self.code_gen_dict:
                # transform list into long string separated by '\n'
                code_gen_line = "\n".join(self.code_gen_dict[key])
                template = template.replace(key, code_gen_line)
            f.write(template)
        #raise NotImplementedError("This function is not yet immplemented.")
    
    def ipgen_extra_includes(self):
        """Add kernel-specific include paths."""
        import os
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        return f"-I{kernel_dir} -I{utils_dir}"
