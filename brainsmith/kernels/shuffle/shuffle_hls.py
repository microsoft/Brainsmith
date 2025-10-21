############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Shane T. Fleming <shane.fleming@amd.com>
# @author       Thomas Keller <thomaskeller@microsoft.com> (AutoShuffle adaptation)
############################################################################

import numpy as np
import os

from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.kernels.shuffle.shuffle import Shuffle
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy
from finn.util.basic import CppBuilder
from brainsmith.core.plugins import backend

@backend(
    name="ShuffleHLS",
    kernel="Shuffle",
    language="hls",
    description="HLS implementation of Shuffle",
    author="Shane Fleming"
)
class Shuffle_hls(Shuffle, HLSBackend):
    """HLS backend for Shuffle kernel (KernelOp-based).

    This backend adapts the schema-driven Shuffle implementation
    to work with FINN's HLS code generation system.

    Key features:
    - Uses uppercase "SIMD" (KernelOp convention)
    - Extracts shapes from kernel_instance (not nodeattrs)
    - Follows Arete principle: no shape storage
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Define nodeattrs for Shuffle_hls backend.

        Combines:
        - Shuffle's schema-derived nodeattrs (interface datatypes, kernel params)
        - HLSBackend's execution nodeattrs
        - Manual overrides for HLS-specific parameters

        We override SIMD here to ensure it's uppercase (KernelOp convention).
        """
        # Get Shuffle's schema-derived nodeattrs (includes dynamic interface datatypes)
        my_attrs = Shuffle.get_nodeattr_types(self)

        # Add HLSBackend nodeattrs
        my_attrs.update(HLSBackend.get_nodeattr_types(self))

        # Override SIMD to ensure uppercase (KernelOp convention vs legacy lowercase)
        my_attrs["SIMD"] = ("i", False, 1)

        return my_attrs

    def global_includes(self):
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "input_gen.hpp"',
            '#include <ap_int.h>',
            '#include <hls_vector.h>',
            '#include <hls_stream.h>',
        ]

    def defines(self, var):
        """Generate HLS constant definitions.

        Extracts values from kernel_instance instead of nodeattrs (Arete principle).
        Uses uppercase "SIMD" for KernelOp convention.
        """
        # Get parallelization parameter (uppercase for KernelOp)
        simd = self.get_nodeattr("SIMD")
        dtype = self.get_input_datatype()

        self.code_gen_dict["$DEFINES$"] = [
            f"""
            constexpr unsigned  SIMD = {simd};
            using  TE = {dtype.get_hls_datatype_str()};
            using  TV = hls::vector<TE, SIMD>;
            """
        ]

    def get_exp_cycles(self):
        """Compute expected cycles for execution.

        Extracts output shape from kernel_instance (Arete principle).
        """
        # Extract from cached kernel instance
        ki = self.kernel_instance
        out_shape = ki.outputs["output"].tensor_shape
        simd = self.get_nodeattr("SIMD")

        return int(np.prod(out_shape) / simd)

    def docompute(self):
        """Generate HLS docompute code for hardware execution.

        Uses input_gen template with loop coefficients computed during transformation.
        """
        simd = self.get_nodeattr("SIMD")

        # Extract from kernel instance (Arete principle)
        ki = self.kernel_instance
        out_shape = list(ki.outputs["output"].tensor_shape)

        # Adjust output shape for SIMD (last dimension divided)
        out_shape[-1] = int(out_shape[-1] / simd)

        # Get loop coefficients (computed during transformation)
        loop_coeffs = self.get_nodeattr("loop_coeffs")

        # Adjust loop coefficients for SIMD
        loop_coeffs = [1 if x == 1 else int(x / simd) for x in loop_coeffs]

        # Create interleaved array for input_gen template: [N0, C0, N1, C1, ...]
        interleaved = [int(item) for pair in zip(out_shape, loop_coeffs) for item in pair]

        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            hls::stream<TV>  src0;
            hls::stream<TV>  dst0;
            #pragma HLS stream variable=src0 depth=2
            #pragma HLS stream variable=dst0 depth=2

            move(in0_V, src0);
            input_gen<-1,{np.prod(out_shape)},{','.join(map(str,interleaved))}>(src0, dst0);
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
        else:
            raise Exception(f"Unsupported execution mode: {mode}")

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
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        # input_gen.hpp is in kernel_dir, not kernel_dir/hls
        builder.append_includes(f"-I{kernel_dir}")
        builder.append_includes(f"-I{utils_dir}")
        #builder.append_includes("-I{}/include".format(os.environ["HLS_PATH"]))
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$BSMITH_DIR/deps/cnpy/cnpy.cpp")
        builder.append_includes("-lz")
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

        simd = self.get_nodeattr("SIMD")

        # Extract from kernel instance (Arete principle)
        ki = self.kernel_instance
        out_shape = list(ki.outputs["output"].tensor_shape)

        # Adjust for SIMD
        out_shape[-1] = int(out_shape[-1] / simd)

        # Get loop coefficients
        loop_coeffs = self.get_nodeattr("loop_coeffs")
        loop_coeffs = [1 if x == 1 else int(x / simd) for x in loop_coeffs]

        # Create interleaved array
        interleaved = [int(item) for pair in zip(out_shape, loop_coeffs) for item in pair]

        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            static hls::stream<TV>  in0_V;
            static hls::stream<TV>  out0_V;

            npy2vectorstream<TE, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            while(out0_V.size() != stream_size) {{
                input_gen<-1,{np.prod(out_shape)},{','.join(map(str,interleaved))}>(in0_V, out0_V);
            }}

            vectorstream2npy<TE, float, SIMD>(out0_V,{oshape_str}, "{path}/output_0.npy");
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
        # input_gen.hpp is in kernel_dir, not kernel_dir/hls
        return f"-I{kernel_dir} -I{utils_dir}"
