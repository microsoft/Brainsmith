############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Softmax HLS Backend - HLS implementation using KernelOp and Dataflow Modeling.

This backend provides cppsim and rtlsim execution modes for Softmax,
generating optimized HLS code using the kernel instance for automatic shape inference.

Key differences from legacy implementation:
- Uses kernel instance inputs[0].stream_shape[-1] instead of get_nodeattr("SIMD")
- Uses kernel instance inputs[0].tensor_shape[-1] instead of get_nodeattr("ifm_dim")[-1]
- Automatic shape validation via declarative constraints
- Intelligent two-level caching (design space + kernel instance)
"""

import os

import numpy as np
from finn.custom_op.fpgadataflow import templates
from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.basic import CppBuilder
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

from brainsmith.kernels.softmax.softmax import Softmax
from brainsmith.registry import backend


@backend(
    name="Softmax_hls",
    target_kernel="brainsmith:Softmax",
    language="hls",
    description="HLS backend for Softmax kernel using Dataflow Modeling",
    author="Thomas Keller",
)
class Softmax_hls(Softmax, HLSBackend):
    """HLS backend for Softmax using kernel instance for shape information.

    This class inherits from both Softmax (provides kernel_schema and Python execution)
    and HLSBackend (provides HLS code generation infrastructure).

    The key innovation is using kernel instance properties instead of nodeattrs for shape info:
    - self.design_point.inputs["input"].stream_shape[-1] for SIMD (design_point property returns KernelDesignPoint)
    - self.design_point.inputs["input"].tensor_shape[-1] for width
    - self.design_point.inputs["input"].datatype for input type
    - self.design_point.outputs["output"].datatype for output type

    This ensures consistency with the declarative KernelSchema and automatic validation.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Merge node attributes from both parent classes.

        Returns:
            dict: Combined attributes from HLSBackend and Softmax
        """
        my_attrs = {}
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        my_attrs.update(Softmax.get_nodeattr_types(self))
        return my_attrs

    def global_includes(self):
        """Define global C++ includes for HLS code generation.

        Sets code_gen_dict["$GLOBALS$"] with required headers:
        - hls_vector.h: HLS vector types
        - softmax.hpp: Softmax kernel implementation
        - bs_utils.hpp: Brainsmith utility functions
        """
        self.code_gen_dict["$GLOBALS$"] = [
            "#include <hls_vector.h>",
            '#include "softmax.hpp"',
            '#include "bs_utils.hpp"'
        ]

    def defines(self, var):
        """Define C++ constants using kernel instance for shape information.

        Args:
            var: Variable name (cppsim/ipgen) - unused but required by interface

        Sets code_gen_dict["$DEFINES$"] with:
        - SIMD: Stream parallelism from kernel instance inputs["input"].stream_shape[-1]
        - W: Total channels from kernel instance inputs["input"].tensor_shape[-1]
        - TI: Input type from kernel instance inputs["input"].datatype
        - TO: Output type from kernel instance outputs["output"].datatype
        """
        # Use kernel instance for shape information (not nodeattrs)
        input_cfg = self.design_point.inputs["input"]
        output_cfg = self.design_point.outputs["output"]

        simd = input_cfg.stream_shape[-1]
        width = input_cfg.tensor_shape[-1]

        idtype = input_cfg.datatype
        odtype = output_cfg.datatype

        self.code_gen_dict["$DEFINES$"] = [
            f"constexpr unsigned SIMD = {simd};",
            f"constexpr unsigned W = {width};",
            f"using TI = {idtype.get_hls_datatype_str()};",
            f"using TO = {odtype.get_hls_datatype_str()};"
        ]

    def docompute(self):
        """Define computation function call for IPgen mode.

        Sets code_gen_dict["$DOCOMPUTE$"] with the kernel function call.
        This is used by the IP generation template to invoke the Softmax pipeline.
        """
        self.code_gen_dict["$DOCOMPUTE$"] = [
            """
            static hls::stream<hls::vector<TI,SIMD>>  src0;
            static hls::stream<hls::vector<TO,SIMD>>  dst0;

            move(in0_V, src0);
            static SoftMax<TI, TO, W, SIMD> sm_inst;
            sm_inst.execute(src0, dst0);
            move(dst0, out0_V);
            """
        ]

    def blackboxfunction(self):
        """Define top-level function signature for HLS synthesis.

        Sets code_gen_dict["$BLACKBOXFUNCTION$"] with the function signature.
        Uses node name for the function name to ensure uniqueness in stitched designs.
        """
        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"""
            void {self.onnx_node.name}(
                hls::stream<hls::vector<TI,SIMD>> &in0_V,
                hls::stream<hls::vector<TO,SIMD>> &out0_V
            )
            """
        ]

    def pragmas(self):
        """Define HLS pragmas for interface and optimization directives.

        Sets code_gen_dict["$PRAGMAS$"] with:
        - AXIS interface pragmas for input/output streams
        - Aggregate compact=bit for efficient bit packing
        - ap_ctrl_none for continuous streaming operation
        - dataflow with start propagation disabled for performance
        """
        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS interface AXIS port=in0_V",
            "#pragma HLS interface AXIS port=out0_V",
            "#pragma HLS aggregate variable=in0_V compact=bit",
            "#pragma HLS aggregate variable=out0_V compact=bit",
            "#pragma HLS interface ap_ctrl_none port=return",
            "#pragma HLS dataflow disable_start_propagation",
        ]

    def execute_node(self, context, graph):
        """Execute Softmax in cppsim, rtlsim, or python mode.

        Args:
            context: Execution context containing input/output tensors
            graph: ONNX graph

        Modes:
        - python: Delegates to Softmax._execute_python() from parent
        - cppsim: Executes compiled C++ simulation
        - rtlsim: Executes RTL simulation with PyVerilator

        Raises:
            Exception: If execution mode is unsupported
        """
        mode = self.get_nodeattr("exec_mode")
        node = self.onnx_node

        # Python mode delegates to parent implementation
        if mode == "python":
            self._execute_python(context, graph)
            return

        # Prepare input for cppsim/rtlsim
        folded_ishape = self.get_folded_input_shape()
        inp = context[node.input[0]].reshape(folded_ishape).astype(np.float32)

        if mode == "cppsim":
            # C++ simulation mode
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)

            # Execute the precompiled C++ model
            super().exec_precompiled_singlenode_model()

            # Load output from generated npy file
            super().npy_to_dynamic_output(context)

        elif mode == "rtlsim":
            # RTL simulation mode
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            np.save(os.path.join(code_gen_dir, "input_0.npy"), inp)

            # Convert input to RTL simulation format
            export_idt = self.get_input_datatype()
            nbits = self.get_instream_width()
            rtlsim_inp = npy_to_rtlsim_input(
                f"{code_gen_dir}/input_0.npy",
                export_idt,
                nbits
            )

            # Setup and run RTL simulation
            sim = self.get_rtlsim()
            super().reset_rtlsim(sim)
            super().toggle_clk(sim)

            io_dict = {
                "inputs": {"in0": rtlsim_inp},
                "outputs": {"out0": []}
            }
            self.rtlsim_multi_io(sim, io_dict)

            # Convert RTL output back to numpy format
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = f"{code_gen_dir}/output_0.npy"
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                io_dict["outputs"]["out0"],
                out_npy_path,
                odt,
                out_shape,
                packed_bits,
                target_bits
            )

            # Load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(f"Unsupported execution mode: {mode}")

    def code_generation_cppsim(self, model):
        """Generate C++ simulation code for functional verification.

        Args:
            model: ModelWrapper instance

        Generates:
        - execute_<op_type>.cpp: C++ wrapper that loads input, calls kernel, saves output

        Uses template substitution to fill in:
        - $AP_INT_MAX_W$: Maximum ap_int width
        - $GLOBALS$: Include directives
        - $DEFINES$: Type and constant definitions
        - $PRAGMAS$: HLS directives (unused in cppsim but included for consistency)
        - $DOCOMPUTE$: Main simulation loop
        """
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

        # Generate main simulation code
        self.code_gen_dict["$DOCOMPUTE$"] = [
            f"""
            static hls::stream<hls::vector<TI,SIMD>> in0_V;
            static hls::stream<hls::vector<TO,SIMD>> out0_V;

            npy2vectorstream<TI, float, SIMD>("{path}/input_0.npy", in0_V);
            int stream_size = in0_V.size();

            static SoftMax<TI, TO, W, SIMD> sm_inst;

            while(out0_V.size() != stream_size){{
                sm_inst.execute(in0_V, out0_V);
            }}

            vectorstream2npy<TO, float, SIMD>(out0_V, {oshape_str}, "{path}/output_0.npy");
            """
        ]
        self.save_as_npy()

        # Fill in template
        template = templates.docompute_template
        code_gen_path = f"{self.get_nodeattr('code_gen_dir_cppsim')}/execute_{node.op_type}.cpp"

        with open(code_gen_path, "w") as f:
            for key in self.code_gen_dict:
                code_gen_line = "\n".join(self.code_gen_dict[key])
                template = template.replace(key, code_gen_line)
            f.write(template)

    def compile_singlenode_code(self):
        """Compile C++ simulation code into executable.

        Uses CppBuilder to:
        1. Add include paths for FINN, cnpy, finn-hlslib, kernel headers
        2. Add source files (generated code + cnpy)
        3. Link against Vitis HLS simulation libraries
        4. Build executable at code_gen_dir/node_model

        Sets nodeattr "executable_path" for later execution.
        """
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        builder = CppBuilder()

        # Include paths
        builder.append_includes("-I$BSMITH_DIR/deps/finn/src/finn/qnn-data/cpp")
        builder.append_includes("-I$BSMITH_DIR/deps/cnpy/")
        builder.append_includes("-I$BSMITH_DIR/deps/finn-hlslib")

        # Kernel-specific includes
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        builder.append_includes(f"-I{kernel_dir}")
        builder.append_includes(f"-I{utils_dir}")

        # Vitis HLS includes
        builder.append_includes("-I{}/include".format(os.environ["VITIS_PATH"]))

        # Compiler flags
        builder.append_includes("--std=c++14")
        builder.append_includes("-O3")

        # Source files
        builder.append_sources(code_gen_dir + "/*.cpp")
        builder.append_sources("$BSMITH_DIR/deps/cnpy/cnpy.cpp")

        # Libraries
        builder.append_includes("-lz")
        builder.append_includes(
            '-fno-builtin -fno-inline -Wl,-rpath,"$VITIS_PATH/lnx64/lib/csim" '
            '-L$VITIS_PATH/lnx64/lib/csim -lhlsmc++-GCC46'
        )
        builder.append_includes(
            "-L$VITIS_PATH/lnx64/tools/fpo_v7_1 -lgmp -lmpfr "
            "-lIp_floating_point_v7_1_bitacc_cmodel"
        )

        # Build
        builder.set_executable_path(code_gen_dir + "/node_model")
        builder.build(code_gen_dir)
        self.set_nodeattr("executable_path", builder.executable_path)

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generate Vivado IP for hardware synthesis.

        Args:
            model: ModelWrapper instance
            fpgapart: Target FPGA part number
            clk: Target clock period in ns

        Delegates to parent HLSBackend implementation which handles:
        - HLS C++ code generation
        - TCL script generation
        - Vivado HLS project creation
        - IP packaging
        """
        super().code_generation_ipgen(model, fpgapart, clk)

    def ipgen_extra_includes(self):
        """Provide kernel-specific include paths for IP generation.

        Returns:
            str: Include flags for kernel headers and utilities
        """
        kernel_dir = os.path.dirname(os.path.abspath(__file__))
        utils_dir = os.path.join(os.path.dirname(kernel_dir), 'utils')
        return f"-I{kernel_dir} -I{utils_dir}"

    def generate_params(self, model, path):
        """Generate parameter files for kernel (no-op for Softmax).

        Args:
            model: ModelWrapper instance
            path: Output directory path

        Softmax has no weight matrices or lookup tables, so this is empty.
        Other kernels (e.g., MatrixVectorActivation) use this to generate
        weight/threshold parameter files.
        """
        pass
