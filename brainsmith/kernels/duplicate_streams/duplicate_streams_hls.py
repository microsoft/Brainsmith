############################################################################
# Portions derived from FINN project
# Copyright (C) 2023, Advanced Micro Devices, Inc.
# Licensed under BSD-3-Clause License
#
# Modifications and additions Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################

"""DuplicateStreams HLS backend."""

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from brainsmith.kernels.duplicate_streams.duplicate_streams import DuplicateStreams
from brainsmith.registry import backend


@backend(
    name="DuplicateStreamsHLS",
    target_kernel="brainsmith:DuplicateStreams",
    language="hls",
    author="Migrated from AMD FINN"
)
class DuplicateStreams_hls(DuplicateStreams, HLSBackend):
    """HLS backend for DuplicateStreams.

    Generates simple read-once, write-N loop with II=1.
    Variable-arity code generation adapts to N output streams.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Combine DuplicateStreams and HLSBackend nodeattrs."""
        my_attrs = DuplicateStreams.get_nodeattr_types(self)
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    # ================================================================
    # Resource Estimation
    # ================================================================

    def lut_estimation(self):
        """Estimate LUT usage (minimal for simple fanout)."""
        # DuplicateStreams is just wire splitting - minimal cost
        # Small overhead for stream read/write logic
        return 100

    def bram_estimation(self):
        """Estimate BRAM usage (none needed)."""
        return 0

    def dsp_estimation(self):
        """Estimate DSP usage (none needed)."""
        return 0

    # ================================================================
    # HLS Code Generation
    # ================================================================

    def generate_params(self, model, path):
        """Generate HLS implementation code with variable output count."""
        n_outputs = self.get_num_output_streams()
        i_stream_w = self.get_instream_width()
        o_stream_w = self.get_outstream_width()

        # Build function signature (variable arity)
        inp_streams = [f"hls::stream<ap_uint<{i_stream_w}>> &in0_V"]
        for i in range(n_outputs):
            inp_streams.append(f"hls::stream<ap_uint<{o_stream_w}>> &out{i}_V")

        # Build loop body (read once, write N times)
        commands = [f"ap_uint<{i_stream_w}> e = in0_V.read();"]
        for i in range(n_outputs):
            commands.append(f"out{i}_V.write(e);")

        # Compute iteration count
        # get_number_output_values() returns dict for multi-output kernels (FINN API)
        output_vals = self.get_number_output_values()
        iters = output_vals["out0"] if isinstance(output_vals, dict) else output_vals

        # Generate implementation
        impl_hls_code = f"""#pragma once
#include <ap_int.h>
#include <hls_stream.h>

/**
 * DuplicateStreams - Stream fanout (1 input → {n_outputs} outputs)
 *
 * Simple read-once, write-N loop with II=1.
 * Generated for {n_outputs} output streams.
 */
void DuplicateStreamsCustom({', '.join(inp_streams)}) {{
    for(unsigned int i = 0; i < {iters}; i++) {{
        #pragma HLS PIPELINE II=1
        {chr(10).join('        ' + cmd for cmd in commands)}
    }}
}}
"""

        impl_filename = f"{path}/duplicate_impl.hpp"
        with open(impl_filename, "w") as f:
            f.write(impl_hls_code)

    def execute_node(self, context, graph):
        """Execute via HLS simulation."""
        HLSBackend.execute_node(self, context, graph)

    # ================================================================
    # HLS Code Generation (Template Filling)
    # ================================================================

    def global_includes(self):
        """Include generated implementation."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "duplicate_impl.hpp"']

    def defines(self, var):
        """No defines needed for simple fanout."""
        self.code_gen_dict["$DEFINES$"] = []

    def strm_decl(self):
        """Declare input and N output streams."""
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

        # Input stream
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_instream_width()}>> in0_V ("in0_V");'
        )

        # Output streams (variable count)
        for i in range(n_outputs):
            out_name = f"out{i}_V"
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{self.get_outstream_width()}>> {out_name} ("{out_name}");'
            )

    def docompute(self):
        """Generate function call with N outputs."""
        n_outputs = self.get_num_output_streams()
        ostreams = [f"out{i}_V" for i in range(n_outputs)]
        dc = f"DuplicateStreamsCustom(in0_V, {', '.join(ostreams)});"
        self.code_gen_dict["$DOCOMPUTE$"] = [dc]

    def blackboxfunction(self):
        """Generate blackbox function signature (variable arity)."""
        n_outputs = self.get_num_output_streams()
        i_stream_w = self.get_instream_width()
        o_stream_w = self.get_outstream_width()

        inp_streams = [f"hls::stream<ap_uint<{i_stream_w}>> &in0_V"]
        for i in range(n_outputs):
            inp_streams.append(f"hls::stream<ap_uint<{o_stream_w}>> &out{i}_V")

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({', '.join(inp_streams)})"
        ]

    def pragmas(self):
        """Generate interface pragmas for all streams."""
        n_outputs = self.get_num_output_streams()
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]

        for i in range(n_outputs):
            self.code_gen_dict["$PRAGMAS$"].append(
                f"#pragma HLS INTERFACE axis port=out{i}_V"
            )

        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )

    def read_npy_data(self):
        """Read NPY data for simulation."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = f"{code_gen_dir}/input_0.npy"

        self.code_gen_dict["$READNPYDATA$"] = [
            f'npy2apintstream<{packed_hls_type}, {elem_hls_type}, {elem_bits}, '
            f'{npy_type}>("{npy_in}", in0_V, false);'
        ]

    def dataoutstrm(self):
        """Write NPY data for simulation (all outputs)."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()
        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = f"ap_uint<{packed_bits}>"
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"

        # Folded shape for HLS I/O (FINN-specific format)
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = []

        # Generate write for each output stream
        n_outputs = self.get_num_output_streams()
        for i in range(n_outputs):
            npy_out = f"{code_gen_dir}/output_{i}.npy"
            self.code_gen_dict["$DATAOUTSTREAM$"].append(
                f'apintstream2npy<{packed_hls_type}, {elem_hls_type}, {elem_bits}, '
                f'{npy_type}>(out{i}_V, {shape_cpp_str}, "{npy_out}", false);'
            )

    def save_as_npy(self):
        """Save output as NPY (all outputs)."""
        self.code_gen_dict["$SAVEASCNPY$"] = []
        n_outputs = self.get_num_output_streams()

        for i in range(n_outputs):
            self.code_gen_dict["$SAVEASCNPY$"].append(f'saveAsNpy<{i}>(out{i}_V, "{i}");')
