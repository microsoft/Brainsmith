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
    name="DuplicateStreams_hls",
    target_kernel="brainsmith:DuplicateStreams",
    language="hls",
    author="AMD FINN team"
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

    def execute_node(self, context, graph):
        """Execute via HLS simulation."""
        HLSBackend.execute_node(self, context, graph)

    # ================================================================
    # HLS Code Generation
    # ================================================================

    def generate_params(self, model, path):
        """Generate HLS implementation code with variable output count."""
        self._ensure_ready(model)

        inp = self.design_point.input_list[0]
        i_stream_w = inp.stream_width_bits

        # Build function signature (variable arity)
        inp_streams = [f"hls::stream<ap_uint<{i_stream_w}>> &in0_V"]
        for i, output in enumerate(self.design_point.output_list):
            o_stream_w = output.stream_width_bits
            inp_streams.append(f"hls::stream<ap_uint<{o_stream_w}>> &out{i}_V")

        # Build loop body (read once, write N times)
        commands = [f"ap_uint<{i_stream_w}> e = in0_V.read();"]
        for i in range(len(self.design_point.output_list)):
            commands.append(f"out{i}_V.write(e);")

        # Compute iteration count
        # get_number_output_values() returns dict for multi-output kernels (FINN API)
        output_vals = self.get_number_output_values()
        iters = output_vals["out0"] if isinstance(output_vals, dict) else output_vals

        n_outputs = len(self.design_point.output_list)

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
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

        # Input stream
        inp = self.design_point.input_list[0]
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{inp.stream_width_bits}>> in0_V ("in0_V");'
        )

        # Output streams (variable count)
        for i, output in enumerate(self.design_point.output_list):
            out_name = f"out{i}_V"
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{output.stream_width_bits}>> {out_name} ("{out_name}");'
            )

    def docompute(self):
        """Generate function call with N outputs."""
        ostreams = [f"out{i}_V" for i in range(len(self.design_point.output_list))]
        dc = f"DuplicateStreamsCustom(in0_V, {', '.join(ostreams)});"
        self.code_gen_dict["$DOCOMPUTE$"] = [dc]

    def blackboxfunction(self):
        """Generate blackbox function signature (variable arity)."""
        inp = self.design_point.input_list[0]
        i_stream_w = inp.stream_width_bits

        inp_streams = [f"hls::stream<ap_uint<{i_stream_w}>> &in0_V"]
        for i, output in enumerate(self.design_point.output_list):
            o_stream_w = output.stream_width_bits
            inp_streams.append(f"hls::stream<ap_uint<{o_stream_w}>> &out{i}_V")

        self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
            f"void {self.onnx_node.name}({', '.join(inp_streams)})"
        ]

    def pragmas(self):
        """Generate interface pragmas for all streams."""
        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]

        for i in range(len(self.design_point.output_list)):
            self.code_gen_dict["$PRAGMAS$"].append(
                f"#pragma HLS INTERFACE axis port=out{i}_V"
            )

        self.code_gen_dict["$PRAGMAS$"].append(
            "#pragma HLS INTERFACE ap_ctrl_none port=return"
        )
