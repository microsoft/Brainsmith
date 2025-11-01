# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""MVAU HLS backend implementation."""

import numpy as np
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from .mvau import MVAU
from brainsmith.registry import backend


@backend(
    name="MVAU_HLS",
    target_kernel="brainsmith:MVAU",
    language="hls",
    author="AMD FINN Team / Microsoft"
)
class MVAU_hls(MVAU, HLSBackend):
    """HLS backend for MVAU kernel.

    Diamond inheritance: MVAU (kernel logic) + HLSBackend (code generation)
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Merge MVAU + HLSBackend nodeattrs."""
        base_attrs = {}
        base_attrs.update(MVAU.get_nodeattr_types(self))
        base_attrs.update(HLSBackend.get_nodeattr_types(self))
        # Override resType default for HLS
        base_attrs["resType"] = ("s", False, "lut", {"auto", "lut", "dsp"})
        return base_attrs

    def execute_node(self, context, graph):
        """Delegate to HLSBackend execution (cppsim)."""
        HLSBackend.execute_node(self, context, graph)

    # ================================================================
    # HLS Code Generation (Phase 3: All memory modes)
    # Supports: internal_embedded, internal_decoupled, external
    # ================================================================

    def get_template_param_values(self):
        """Generate HLS template parameters for data type wrappers.

        Returns:
            Dict with TSrcI, TDstI, TWeightI template args
        """
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()

        # Phase 2: No bipolar/binary (binaryXnorMode=0)
        # Phase 3 will add binary/bipolar optimizations
        return {
            "TSrcI": f"Slice<{inp_hls_str}>",
            "TDstI": f"Slice<{out_hls_str}>",
            "TWeightI": "Identity",
        }

    def global_includes(self):
        """Add HLS includes to code generation dict."""
        self.code_gen_dict["$GLOBALS$"] = [
            '#include "weights.hpp"',
            '#include "activations.hpp"',
            '#include "mvau.hpp"',
        ]

    def defines(self, var):
        """Generate HLS #defines for matrix dimensions and folding.

        Uses design_point for dimensions (modern pattern).
        """
        # Access dimensions via design_point (modern pattern)
        dp = self.design_point
        weight_interface = dp.inputs["weights"]
        input_interface = dp.inputs["input"]
        output_interface = dp.outputs["output"]

        # numReps = number of blocks to cover full input tensor
        numReps = input_interface.tensor_folding_factor

        self.code_gen_dict["$DEFINES$"] = [
            f"""#define MW1 {weight_interface.block_shape[-2]}
#define MH1 {weight_interface.block_shape[-1]}
#define SIMD1 {input_interface.stream_shape[-1]}
#define PE1 {output_interface.stream_shape[-1]}
#define WMEM1 {self.calc_wmem()}
#define TMEM1 {self.calc_tmem()}
#define numReps {numReps}"""
        ]

        # For decoupled/external: add weight precision
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            wdt = self.get_input_datatype(1)
            self.code_gen_dict["$DEFINES$"].append(f"#define WP1 {wdt.bitwidth()}")


    def strm_decl(self):
        """Generate HLS stream declarations."""
        mem_mode = self.get_nodeattr("mem_mode")

        self.code_gen_dict["$STREAMDECLARATIONS$"] = []
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_instream_width(0)}>> in0_V ("in0_V");'
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            f'hls::stream<ap_uint<{self.get_outstream_width()}>> out0_V ("out0_V");'
        )

        # For decoupled/external: add weight stream
        if mem_mode in ["internal_decoupled", "external"]:
            wwidth = self.get_instream_width(1)
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                f'hls::stream<ap_uint<{wwidth}>> in1_V ("in1_V");'
            )

    def docompute(self):
        """Generate HLS compute function call.

        Handles activation modes:
        - noActivation=1: PassThroughActivation (MV mode)
        - noActivation=0: ThresholdsActivation (MVTU mode)

        Handles memory modes:
        - internal_embedded: Weights compiled into header
        - internal_decoupled: Weights streamed from in1_V
        - external: Weights read from external memory
        """
        tmpl_args = self.get_template_param_values()
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }

        # Determine activation type based on noActivation
        odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
        acc_dtype_hls_str = self.get_internal_datatype("accDataType").get_hls_datatype_str()

        if self.get_nodeattr("noActivation") == 1:
            # MV mode: No activation, pass through accumulator values
            threshs = f"PassThroughActivation<{odtype_hls_str}>()"
        else:
            # MVTU mode: Multi-threshold activation
            act_val = self.get_nodeattr("ActVal")
            # ThresholdsActivation template: <input_type, output_type, n_threshold_steps>
            n_thres_steps = (1 << act_val) - 1 if act_val > 0 else 0
            threshs = f"ThresholdsActivation<{acc_dtype_hls_str}, {odtype_hls_str}, {n_thres_steps}>(thresholds)"

        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "internal_embedded":
            # Embedded mode: weights compiled into C++ header
            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"""Matrix_Vector_Activate_Batch<MW1, MH1, SIMD1, PE1, 1, {tmpl_args["TSrcI"]}, {tmpl_args["TDstI"]}, {tmpl_args["TWeightI"]}>
                (in0_V, out0_V, weights, {threshs}, numReps, {map_to_hls_mult_style[self.get_nodeattr("resType")]});"""
            ]
        elif mem_mode in ["internal_decoupled", "external"]:
            # Decoupled/external mode: weights streamed from in1_V
            # Get weight datatype for template arg
            wdt = self.get_input_datatype(1)
            if wdt == DataType["BIPOLAR"]:
                # Bipolar exported as binary
                export_wdt = DataType["BINARY"]
            else:
                export_wdt = wdt

            export_wdt_hls_str = export_wdt.get_hls_datatype_str()

            self.code_gen_dict["$DOCOMPUTE$"] = [
                f"""Matrix_Vector_Activate_Batch_ext<MW1, MH1, SIMD1, PE1, 1, {tmpl_args["TSrcI"]}, {tmpl_args["TDstI"]}, {export_wdt_hls_str}>
                (in0_V, in1_V, out0_V, {threshs}, numReps, {map_to_hls_mult_style[self.get_nodeattr("resType")]});"""
            ]
        else:
            raise Exception(
                f'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                f'currently "{mem_mode}" is not supported!'
            )

    def blackboxfunction(self):
        """Generate HLS top-level function signature."""
        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "internal_embedded":
            # Embedded: only input and output streams
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                    hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                )"""
            ]
        elif mem_mode in ["internal_decoupled", "external"]:
            # Decoupled/external: add weight stream
            wwidth = self.get_instream_width(1)
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                f"""void {self.onnx_node.name}(
                    hls::stream<ap_uint<{self.get_instream_width(0)}>> &in0_V,
                    hls::stream<ap_uint<{wwidth}>> &in1_V,
                    hls::stream<ap_uint<{self.get_outstream_width()}>> &out0_V
                )"""
            ]
        else:
            raise Exception(
                f'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                f'currently "{mem_mode}" is not supported!'
            )

    def pragmas(self):
        """Generate HLS pragmas."""
        mem_mode = self.get_nodeattr("mem_mode")

        self.code_gen_dict["$PRAGMAS$"] = [
            "#pragma HLS INTERFACE axis port=in0_V",
            "#pragma HLS INTERFACE axis port=out0_V",
            "#pragma HLS INTERFACE ap_ctrl_none port=return",
        ]

        if mem_mode == "internal_embedded":
            # Embedded mode: include params.h and partition weights array
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1"
            )
        elif mem_mode in ["internal_decoupled", "external"]:
            # Decoupled/external mode: add axis interface for weight stream
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
        else:
            raise Exception(
                f'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external", '
                f'currently "{mem_mode}" is not supported!'
            )
