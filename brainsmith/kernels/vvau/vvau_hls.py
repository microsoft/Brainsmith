############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import math
import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.hlsbackend import HLSBackend
from finn.util.basic import is_versal
from finn.util.data_packing import (
    numpy_to_hls_code,
    pack_innermost_dim_as_hex_string,
    npy_to_rtlsim_input,
    rtlsim_output_to_npy,
    roundup_to_integer_multiple,
)
import textwrap

from brainsmith.kernels.vvau.vvau import VectorVectorActivation
from brainsmith.core.plugins import backend


@backend(
    name="VectorVectorActivationHLS",
    kernel="VectorVectorActivation",
    style="hls"
)
class VectorVectorActivation_hls(VectorVectorActivation, HLSBackend):
    """HLS backend for VectorVectorActivation (inherits schema from VectorVectorActivation).

    Corresponds to finn-hlslib Vector_Vector_Activate_Batch function.
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Add HLS-specific nodeattrs."""
        my_attrs = {}
        my_attrs.update(VectorVectorActivation.get_nodeattr_types(self))
        my_attrs.update(HLSBackend.get_nodeattr_types(self))
        return my_attrs

    # ================================================================
    # Resource Estimation
    # ================================================================

    def lut_estimation(self):
        """Calculates resource estimations for LUTs."""
        # TODO add in/out FIFO contributions
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()

        # Parameters from FINN-R paper
        c0 = 300
        c1 = 1.1
        c2 = 0

        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")

        if (mmode == "internal_decoupled" and mstyle == "distributed") or (
            mmode == "internal_embedded" and self.calc_wmem() <= 128
        ):
            c2 = (P * Q * W) * math.ceil(self.calc_wmem() / 64)

        # Multiplication
        res_type = self.get_nodeattr("res_type")
        if res_type == "dsp":
            mult_luts = 0
        else:
            mult_luts = Q * (2 * math.ceil((W + A) / 6) - 1) * (W + A)

        # Adder tree
        addertree_luts = (W + A) * (2 * Q - 1)

        # Accumulator
        acc_datatype = self.get_accumulator_datatype()
        acc_bits = acc_datatype.bitwidth()
        k_h, k_w = self.get_nodeattr("Kernel")

        # Minimum accumulator estimate
        alpha = math.log(k_h * k_w, 2) + W + A - 1 - int(idt.signed())
        acc_bits = min(
            acc_datatype.bitwidth(),
            np.ceil(alpha + math.log(1 + pow(2, -alpha), 2) + 1),
        )
        acc_luts = acc_bits

        # Thresholds and comparators
        thr_luts = 0
        comp_luts = 0
        noact = self.get_nodeattr("no_activation")

        if noact == 0:
            odt = self.get_output_datatype()
            B = odt.bitwidth()
            thr_luts = (2**B - 1) * acc_bits * self.calc_tmem() / 64
            comp_luts = (2**B - 1) * acc_bits

        return int(
            c0 + c1 * (P * (mult_luts + addertree_luts + acc_luts + thr_luts + comp_luts)) + c2
        )

    def dsp_estimation(self, fpgapart):
        """DSP resource estimation."""
        P = self.get_nodeattr("PE")
        res_type = self.get_nodeattr("res_type")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        idt = self.get_input_datatype(0)
        A = idt.bitwidth()

        if res_type == "dsp":
            mult_dsp = P * np.ceil((W + A) / 48)
        else:
            mult_dsp = 0

        return int(mult_dsp)

    def uram_estimation(self):
        """URAM resource estimation."""
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        mem_width = Q * W * P
        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")

        if (
            (mmode == "internal_decoupled" and mstyle != "ultra")
            or (mmode == "internal_embedded")
            or (mmode == "external")
        ):
            return 0

        width_multiplier = math.ceil(mem_width / 72)
        depth_multiplier = math.ceil(omega / 4096)
        return width_multiplier * depth_multiplier

    def bram_estimation(self):
        """BRAM resource estimation."""
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        mem_width = Q * W * P

        mmode = self.get_nodeattr("mem_mode")
        mstyle = self.get_nodeattr("ram_style")

        if (
            (mmode == "internal_decoupled" and mstyle in ["distributed", "ultra"])
            or (mstyle == "auto" and self.calc_wmem() <= 128)
            or (mmode == "internal_embedded" and self.calc_wmem() <= 128)
            or (mmode == "external")
        ):
            return 0

        if mem_width == 1:
            return math.ceil(omega / 16384)
        elif mem_width == 2:
            return math.ceil(omega / 8192)
        elif mem_width <= 4:
            return (math.ceil(omega / 4096)) * (math.ceil(mem_width / 4))
        elif mem_width <= 9:
            return (math.ceil(omega / 2048)) * (math.ceil(mem_width / 8))
        elif mem_width <= 18 or omega > 512:
            return (math.ceil(omega / 1024)) * (math.ceil(mem_width / 16))
        else:
            return (math.ceil(omega / 512)) * (math.ceil(mem_width / 32))

    def bram_efficiency_estimation(self):
        """BRAM efficiency estimation."""
        P = self.get_nodeattr("PE")
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        omega = self.calc_wmem()
        bram16_est = self.bram_estimation()

        if bram16_est == 0:
            return 1

        wbits = W * P * omega
        bram16_est_capacity = bram16_est * 36 * 512
        return wbits / bram16_est_capacity

    def uram_efficiency_estimation(self):
        """URAM efficiency estimation."""
        wdt = self.get_input_datatype(1)
        W = wdt.bitwidth()
        D_in = int(np.prod(self.get_nodeattr("Kernel")))
        D_out = self.get_nodeattr("Channels")
        uram_est = self.uram_estimation()

        if uram_est == 0:
            return 1

        wbits = W * D_in * D_out
        uram_est_capacity = uram_est * 72 * 4096
        return wbits / uram_est_capacity

    # ================================================================
    # Execution
    # ================================================================

    def execute_node(self, context, graph):
        """Execute node (cppsim or rtlsim)."""
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        if mode == "cppsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        else:
            raise Exception(
                f"Invalid exec_mode: {mode}. Must be 'cppsim' or 'rtlsim'"
            )

        # Create npy file for each input
        in_ind = 0
        for inputs in node.input:
            if in_ind == 0:
                assert str(context[inputs].dtype) == "float32", "Input datatype is not float32"
                expected_inp_shape = self.get_folded_input_shape()
                reshaped_input = context[inputs].reshape(expected_inp_shape)

                if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                    reshaped_input = (reshaped_input + 1) / 2
                    export_idt = DataType["BINARY"]
                else:
                    export_idt = self.get_input_datatype(0)

                reshaped_input = reshaped_input.copy()
                np.save(
                    os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                    reshaped_input,
                )
            elif in_ind > 2:
                raise Exception("Unexpected input found for VectorVectorActivation")
            in_ind += 1

        if mode == "cppsim":
            super().exec_precompiled_singlenode_model()
            super().npy_to_dynamic_output(context)

            # Reinterpret binary output as bipolar where needed
            if self.get_output_datatype() == DataType["BIPOLAR"]:
                out = context[node.output[0]]
                out = 2 * out - 1
                context[node.output[0]] = out

            assert (
                context[node.output[0]].shape == self.get_normal_output_shape()
            ), "cppsim did not produce expected output shape"

        elif mode == "rtlsim":
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)

            if mem_mode in ["external", "internal_decoupled"]:
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)

                if self.get_input_datatype(1) == DataType["BIPOLAR"]:
                    export_wdt = DataType["BINARY"]

                wei = npy_to_rtlsim_input(
                    "{}/weights.npy".format(code_gen_dir), export_wdt, wnbits
                )
                dim_h, dim_w = self.get_nodeattr("Dim")
                num_w_reps = dim_h * dim_w

                io_dict = {
                    "inputs": {"in0": inp, "in1": wei * num_w_reps},
                    "outputs": {"out0": []},
                }
            else:
                io_dict = {
                    "inputs": {"in0": inp},
                    "outputs": {"out0": []},
                }

            self.rtlsim_multi_io(sim, io_dict)
            super().close_rtlsim(sim)

            output = io_dict["outputs"]["out0"]
            odt = self.get_output_datatype()
            target_bits = odt.bitwidth()
            packed_bits = self.get_outstream_width()
            out_npy_path = "{}/output_0.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()
            rtlsim_output_to_npy(
                output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # Load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

    # ================================================================
    # Code Generation
    # ================================================================

    def code_generation_ipgen(self, model, fpgapart, clk):
        """Generates C++ code and tcl script for IP generation."""
        super().code_generation_ipgen(model, fpgapart, clk)
        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "internal_decoupled":
            if self.get_nodeattr("ram_style") == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                assert runtime_writeable == 1, (
                    "Layer with URAM weights must have runtime_writeable_weights=1 "
                    "if Ultrascale device is targeted"
                )
            self.generate_hdl_memstream(fpgapart)

    def get_template_param_values(self):
        """Returns template parameter values for HLS code generation."""
        ret = dict()
        inp_hls_str = self.get_input_datatype(0).get_hls_datatype_str()
        out_hls_str = self.get_output_datatype().get_hls_datatype_str()
        inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
        wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
        bin_xnor_mode = self.get_nodeattr("binary_xnor_mode") == 1

        if (inp_is_binary or wt_is_binary) and (not bin_xnor_mode):
            raise Exception("True binary (non-bipolar) inputs not yet supported")

        inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
        wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]

        # Reinterpret inp/wt as bipolar if bin_xnor_mode is set
        inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
        wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)

        # Fill in TSrcI and TWeightI
        if inp_is_bipolar and wt_is_bipolar:
            ret["TSrcI"] = "Recast<XnorMul>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and wt_is_bipolar:
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Recast<Binary>"
        elif inp_is_bipolar and (not wt_is_bipolar):
            ret["TSrcI"] = "Recast<Binary>"
            ret["TWeightI"] = "Identity"
        elif (not inp_is_bipolar) and (not wt_is_bipolar):
            ret["TSrcI"] = "Slice<%s>" % inp_hls_str
            ret["TWeightI"] = "Identity"

        # Fill in TDstI
        ret["TDstI"] = "Slice<%s>" % out_hls_str

        return ret

    def global_includes(self):
        """Generate global includes for HLS code."""
        self.code_gen_dict["$GLOBALS$"] = ['#include "weights.hpp"']
        self.code_gen_dict["$GLOBALS$"] += ['#include "activations.hpp"']

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode not in ["internal_embedded", "internal_decoupled", "external"]:
            raise Exception(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external"'
            )

        if self.calc_tmem() != 0:
            self.code_gen_dict["$GLOBALS$"] += ['#include "thresh.h"']

    def defines(self, var):
        """Generate #define directives for HLS code."""
        dim_h, dim_w = self.get_nodeattr("Dim")
        numReps = 1 * dim_h * dim_w
        k_h, k_w = self.get_nodeattr("Kernel")
        innerProdDim = k_h * k_w
        mem_mode = self.get_nodeattr("mem_mode")

        self.code_gen_dict["$DEFINES$"] = [
            """#define Channels1 {}\n #define InnerProdDim {}\n
            #define SIMD1 {}\n #define PE1 {}\n #define numReps {}""".format(
                self.get_nodeattr("Channels"),
                innerProdDim,
                self.get_nodeattr("SIMD"),
                self.get_nodeattr("PE"),
                numReps,
            )
        ]

        if mem_mode in ["internal_decoupled", "external"]:
            wdt = self.get_input_datatype(1)
            self.code_gen_dict["$DEFINES$"].append(
                "#define WP1 {}\n".format(wdt.bitwidth())
            )

    def read_npy_data(self):
        """Generate code to read npy data for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_input_datatype(0)

        if dtype == DataType["BIPOLAR"]:
            dtype = DataType["BINARY"]

        elem_bits = dtype.bitwidth()
        packed_bits = self.get_instream_width(0)
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_in = "%s/input_0.npy" % code_gen_dir

        self.code_gen_dict["$READNPYDATA$"] = []
        self.code_gen_dict["$READNPYDATA$"].append(
            'npy2apintstream<%s, %s, %d, %s>("%s", in0_V, false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
        )

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            wdt = self.get_input_datatype(1)
            elem_bits = wdt.bitwidth()
            packed_bits = self.get_instream_width(1)
            packed_hls_type = "ap_uint<%d>" % packed_bits
            elem_hls_type = wdt.get_hls_datatype_str()
            npy_type = "float"
            npy_in = "%s/weights.npy" % code_gen_dir

            self.code_gen_dict["$READNPYDATA$"].append(
                'npy2apintstream<%s, %s, %d, %s>("%s", in1_V, false, numReps);'
                % (packed_hls_type, elem_hls_type, elem_bits, npy_type, npy_in)
            )

    def strm_decl(self):
        """Generate stream declarations for HLS code."""
        mem_mode = self.get_nodeattr("mem_mode")
        self.code_gen_dict["$STREAMDECLARATIONS$"] = []

        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> in0_V ("in0_V");'.format(self.get_instream_width(0))
        )
        self.code_gen_dict["$STREAMDECLARATIONS$"].append(
            'hls::stream<ap_uint<{}>> out0_V ("out0_V");'.format(self.get_outstream_width())
        )

        if mem_mode in ["internal_decoupled", "external"]:
            self.code_gen_dict["$STREAMDECLARATIONS$"].append(
                'hls::stream<ap_uint<{}>> in1_V ("in1_V");'.format(self.get_instream_width(1))
            )

    def docompute(self):
        """Generate compute function call for HLS code."""
        mem_mode = self.get_nodeattr("mem_mode")
        map_to_hls_mult_style = {
            "auto": "ap_resource_dflt()",
            "lut": "ap_resource_lut()",
            "dsp": "ap_resource_dsp()",
        }

        tmpl_args = self.get_template_param_values()

        if self.calc_tmem() == 0:
            odtype_hls_str = self.get_output_datatype().get_hls_datatype_str()
            threshs = "PassThroughActivation<%s>()" % odtype_hls_str
        else:
            threshs = "threshs"

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$DOCOMPUTE$"] = [
                """Vector_Vector_Activate_Batch<Channels1, InnerProdDim, SIMD1, PE1, 1, {}, {}, {}>
                (in0_V, out0_V, weights, {}, numReps, {});""".format(
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("res_type")],
                )
            ]
        elif mem_mode in ["internal_decoupled", "external"]:
            wdt = self.get_input_datatype(1)
            if wdt == DataType["BIPOLAR"]:
                export_wdt = DataType["BINARY"]
            else:
                export_wdt = wdt

            wdtype_hls_str = export_wdt.get_hls_datatype_str()

            self.code_gen_dict["$DOCOMPUTE$"] = [
                """{}<Channels1, InnerProdDim, SIMD1, PE1, 1, {}, {}, {}, {}>
                (in0_V, out0_V, in1_V, {}, numReps, {});""".format(
                    "Vector_Vector_Activate_Stream_Batch",
                    tmpl_args["TSrcI"],
                    tmpl_args["TDstI"],
                    tmpl_args["TWeightI"],
                    wdtype_hls_str,
                    threshs,
                    map_to_hls_mult_style[self.get_nodeattr("res_type")],
                )
            ]
        else:
            raise Exception(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external"'
            )

    def dataoutstrm(self):
        """Generate code to write output stream to npy for cppsim."""
        code_gen_dir = self.get_nodeattr("code_gen_dir_cppsim")
        dtype = self.get_output_datatype()

        if dtype == DataType["BIPOLAR"]:
            dtype = DataType["BINARY"]

        elem_bits = dtype.bitwidth()
        packed_bits = self.get_outstream_width()
        packed_hls_type = "ap_uint<%d>" % packed_bits
        elem_hls_type = dtype.get_hls_datatype_str()
        npy_type = "float"
        npy_out = "%s/output_0.npy" % code_gen_dir
        shape = self.get_folded_output_shape()
        shape_cpp_str = str(shape).replace("(", "{").replace(")", "}")

        self.code_gen_dict["$DATAOUTSTREAM$"] = [
            'apintstream2npy<%s, %s, %d, %s>(out0_V, %s, "%s", false);'
            % (packed_hls_type, elem_hls_type, elem_bits, npy_type, shape_cpp_str, npy_out)
        ]

    def save_as_npy(self):
        """Placeholder for save_as_npy."""
        self.code_gen_dict["$SAVEASCNPY$"] = []

    def blackboxfunction(self):
        """Generate blackbox function signature for HLS code."""
        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(hls::stream<ap_uint<{}>> &in0_V,
                hls::stream<ap_uint<{}>> &out0_V
                )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    self.get_outstream_width(),
                )
            ]
        elif mem_mode in ["internal_decoupled", "external"]:
            self.code_gen_dict["$BLACKBOXFUNCTION$"] = [
                """void {}(
                    hls::stream<ap_uint<{}>> &in0_V,
                    hls::stream<ap_uint<{}>> &in1_V,
                    hls::stream<ap_uint<{}>> &out0_V
                    )""".format(
                    self.onnx_node.name,
                    self.get_instream_width(0),
                    self.get_instream_width(1),
                    self.get_outstream_width(),
                )
            ]
        else:
            raise Exception(
                'Please set mem_mode to "internal_embedded" or "internal_decoupled"'
            )

    def pragmas(self):
        """Generate HLS pragmas for code generation."""
        mem_mode = self.get_nodeattr("mem_mode")

        self.code_gen_dict["$PRAGMAS$"] = ["#pragma HLS INTERFACE axis port=in0_V"]
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=out0_V")
        self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE ap_ctrl_none port=return")

        if mem_mode == "internal_embedded":
            self.code_gen_dict["$PRAGMAS$"].append('#include "params.h"')
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=weights.m_weights complete dim=1"
            )
        elif mem_mode in ["internal_decoupled", "external"]:
            self.code_gen_dict["$PRAGMAS$"].append("#pragma HLS INTERFACE axis port=in1_V")
        else:
            raise Exception(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external"'
            )

        if self.calc_tmem() != 0:
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=1"
            )
            self.code_gen_dict["$PRAGMAS$"].append(
                "#pragma HLS ARRAY_PARTITION variable=threshs.m_thresholds complete dim=3"
            )

    def instantiate_ip(self, cmd):
        """Instantiate the HLS IP in Vivado IPI."""
        vlnv = self.get_nodeattr("ip_vlnv")
        node_name = self.onnx_node.name

        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type ip -vlnv %s /%s/%s" % (vlnv, node_name, node_name)
            )
        else:
            cmd.append("create_bd_cell -type ip -vlnv %s %s" % (vlnv, node_name))

    # ================================================================
    # Weight/Threshold File Generation
    # ================================================================

    def make_weight_file(self, weights, weight_file_mode, weight_file_name):
        """Produce file containing weights in appropriate format."""
        # Convert weights into hlslib-compatible format
        weight_tensor = self.get_hw_compatible_weight_tensor(weights)
        export_wdt = self.get_input_datatype(1)

        # Convert bipolar weights to binary for export
        if self.get_input_datatype(1) == DataType["BIPOLAR"]:
            export_wdt = DataType["BINARY"]

        if weight_file_mode == "hls_header":
            weight_hls_code = numpy_to_hls_code(
                weight_tensor, export_wdt, "weights", True, True
            )

            f_weights = open(weight_file_name, "w")
            if export_wdt.bitwidth() != 1:
                f_weights.write(
                    "const FixedPointWeights<{},{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        export_wdt.get_hls_datatype_str(),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            else:
                f_weights.write(
                    "const BinaryWeights<{},{},{}> weights = ".format(
                        self.get_nodeattr("SIMD"),
                        self.get_nodeattr("PE"),
                        self.calc_wmem(),
                    )
                )
            f_weights.write(weight_hls_code)
            f_weights.close()

        elif "decoupled" in weight_file_mode:
            # Transpose and flip for decoupled mode
            weight_tensor_unflipped = np.transpose(weight_tensor, (0, 2, 1, 3))
            weight_tensor_simd_flipped = np.flip(weight_tensor_unflipped, axis=-1)
            weight_tensor_pe_flipped = np.flip(weight_tensor_unflipped, axis=-2)
            weight_tensor_pe_simd_flipped = np.flip(weight_tensor_pe_flipped, axis=-1)

            pe = self.get_nodeattr("PE")
            simd = self.get_nodeattr("SIMD")

            weight_tensor_simd_flipped = weight_tensor_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_simd_flipped = weight_tensor_simd_flipped.copy()

            weight_tensor_pe_flipped = weight_tensor_pe_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_flipped = weight_tensor_pe_flipped.copy()

            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.reshape(1, -1, pe * simd)
            weight_tensor_pe_simd_flipped = weight_tensor_pe_simd_flipped.copy()

            if weight_file_mode == "decoupled_npy":
                if self.onnx_node.op_type == "VectorVectorActivation_rtl":
                    weight_tensor_unflipped = weight_tensor_unflipped.reshape(1, -1, pe * simd)
                    weight_tensor_unflipped = weight_tensor_unflipped.copy()
                    np.save(weight_file_name, weight_tensor_unflipped)
                else:
                    np.save(weight_file_name, weight_tensor_simd_flipped)

            elif weight_file_mode == "decoupled_verilog_dat":
                weight_width = self.get_instream_width(1)
                weight_width_padded = roundup_to_integer_multiple(weight_width, 4)

                if self.onnx_node.op_type == "VectorVectorActivation_rtl":
                    weight_arr = pack_innermost_dim_as_hex_string(
                        weight_tensor_pe_simd_flipped, export_wdt, weight_width_padded, prefix=""
                    )
                else:
                    weight_arr = pack_innermost_dim_as_hex_string(
                        weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                    )

                weight_stream = weight_arr.flatten()
                weight_stream = weight_stream.copy()
                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        f.write(val + "\n")

            elif weight_file_mode == "decoupled_runtime":
                weight_width = self.get_instream_width(1)
                words_per_memwidth = 2 ** math.ceil(math.log2(weight_width / 32))
                if words_per_memwidth < 1:
                    words_per_memwidth = 1
                weight_width_padded = words_per_memwidth * 32

                weight_tensor_pe_flipped = pack_innermost_dim_as_hex_string(
                    weight_tensor_pe_flipped, export_wdt, weight_width_padded, prefix=""
                )
                weight_stream = weight_tensor_pe_flipped.flatten()
                weight_stream = weight_stream.copy()

                with open(weight_file_name, "w") as f:
                    for val in weight_stream:
                        words_32b = textwrap.wrap(val, 8)
                        words_32b.reverse()
                        for word_32b in words_32b:
                            f.write(word_32b + "\n")
            else:
                raise Exception("Unknown weight_file_mode")
        else:
            raise Exception("Unknown weight_file_mode")

    def generate_params(self, model, path):
        """Generate parameter files (weights, thresholds)."""
        mem_mode = self.get_nodeattr("mem_mode")
        code_gen_dir = path

        # Generate weight files
        weights = model.get_initializer(self.onnx_node.input[1])

        if mem_mode == "internal_embedded":
            weight_filename = "{}/params.h".format(code_gen_dir)
            self.make_weight_file(weights, "hls_header", weight_filename)
        elif mem_mode in ["internal_decoupled", "external"]:
            weight_filename_sim = "{}/weights.npy".format(code_gen_dir)
            self.make_weight_file(weights, "decoupled_npy", weight_filename_sim)

            if mem_mode == "internal_decoupled":
                weight_filename_rtl = "{}/memblock.dat".format(code_gen_dir)
                self.make_weight_file(weights, "decoupled_verilog_dat", weight_filename_rtl)
        else:
            raise Exception(
                'Please set mem_mode to "internal_embedded", "internal_decoupled", or "external"'
            )

        # Generate threshold files
        if len(self.onnx_node.input) > 2:
            thresholds = model.get_initializer(self.onnx_node.input[2])
            if thresholds is not None:
                threshold_tensor = self.get_hw_compatible_threshold_tensor(thresholds)

                # Use UINT32 threshold export for bipolar × bipolar
                inp_is_bipolar = self.get_input_datatype(0) == DataType["BIPOLAR"]
                wt_is_bipolar = self.get_input_datatype(1) == DataType["BIPOLAR"]
                inp_is_binary = self.get_input_datatype(0) == DataType["BINARY"]
                wt_is_binary = self.get_input_datatype(1) == DataType["BINARY"]
                bin_xnor_mode = self.get_nodeattr("binary_xnor_mode") == 1
                inp_is_bipolar = inp_is_bipolar or (inp_is_binary and bin_xnor_mode)
                wt_is_bipolar = wt_is_bipolar or (wt_is_binary and bin_xnor_mode)

                tdt = DataType[self.get_nodeattr("acc_dtype")]

                assert np.vectorize(tdt.allowed)(threshold_tensor).all(), (
                    f"Thresholds in {self.onnx_node.name} can't be expressed with type {str(tdt)}"
                )

                thresholds_hls_code = numpy_to_hls_code(
                    threshold_tensor, tdt, "thresholds", False, True
                )

                # Write thresholds into thresh.h
                f_thresh = open("{}/thresh.h".format(code_gen_dir), "w")
                tdt_hls = tdt.get_hls_datatype_str()

                export_odt = self.get_output_datatype()
                if self.get_output_datatype() == DataType["BIPOLAR"]:
                    export_odt = DataType["BINARY"]
                odt_hls = export_odt.get_hls_datatype_str()

                f_thresh.write(
                    "static ThresholdsActivation<{},{},{},{},{},{},{}> threshs = ".format(
                        self.calc_tmem(),
                        self.get_nodeattr("PE"),
                        threshold_tensor.shape[-1],
                        tdt_hls,
                        odt_hls,
                        self.get_nodeattr("act_val"),
                        "comp::less_equal<%s, %s>" % (tdt_hls, tdt_hls),
                    )
                )
                f_thresh.write(thresholds_hls_code)
                f_thresh.close()

    def get_op_and_param_counts(self):
        """Get operation and parameter counts for this layer."""
        k_h, k_w = self.get_nodeattr("Kernel")
        fm = self.get_nodeattr("Channels")
        dim_h, dim_w = self.get_nodeattr("Dim")
        weight_bits = self.get_input_datatype(1).bitwidth()
        inp_bits = self.get_input_datatype(0).bitwidth()
        num_repetitions = int(dim_h * dim_w)
        mac_count = k_h * k_w * fm * num_repetitions

        # Canonicalize op type
        bw1 = min(inp_bits, weight_bits)
        bw2 = max(inp_bits, weight_bits)
        mac_op_type = "op_mac_%dbx%db" % (bw1, bw2)
        weight_param_type = "param_weight_%db" % (weight_bits)
        weight_count = k_h * k_w * fm

        ret_dict = {mac_op_type: mac_count, weight_param_type: weight_count}

        if self.get_nodeattr("no_activation") == 0:
            tdt = DataType[self.get_nodeattr("acc_dtype")]
            thres_bits = tdt.bitwidth()
            thres_param_type = "param_threshold_%db" % (thres_bits)
            thres_count = fm
            ret_dict[thres_param_type] = thres_count

        return ret_dict

    def derive_characteristic_fxns(self, period):
        """Derive characteristic functions for RTL simulation."""
        n_inps = np.prod(self.get_folded_input_shape()[:-1])
        io_dict = {
            "inputs": {"in0": [0 for i in range(n_inps)]},
            "outputs": {"out0": []},
        }

        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["internal_decoupled", "external"]:
            n_weight_inps = self.calc_wmem()
            dim_h, dim_w = self.get_nodeattr("Dim")
            num_w_reps = dim_h * dim_w
            io_dict["inputs"]["in1"] = [0 for i in range(num_w_reps * n_weight_inps)]

        super().derive_characteristic_fxns(period, override_rtlsim_dict=io_dict)

    def get_verilog_top_module_intf_names(self):
        """Get Verilog top module interface names."""
        intf_names = super().get_verilog_top_module_intf_names()
        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "external":
            intf_names["s_axis"].append(("in1_V", self.get_instream_width_padded(1)))

        if mem_mode == "internal_decoupled":
            runtime_writeable = self.get_nodeattr("runtime_writeable_weights") == 1
            if runtime_writeable:
                intf_names["axilite"] = ["s_axilite"]

        return intf_names

    def code_generation_ipi(self):
        """Generate IPI integration code for Vivado."""
        source_target = "./ip/verilog/rtl_ops/%s" % self.onnx_node.name
        cmd = ["file mkdir %s" % source_target]

        mem_mode = self.get_nodeattr("mem_mode")

        if mem_mode == "internal_decoupled":
            runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
            node_name = self.onnx_node.name

            # Create hierarchy
            clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
            rst_name = self.get_verilog_top_module_intf_names()["rst"][0]
            dout_name = self.get_verilog_top_module_intf_names()["m_axis"][0][0]
            din_name = self.get_verilog_top_module_intf_names()["s_axis"][0][0]

            cmd.append("create_bd_cell -type hier %s" % node_name)
            cmd.append("create_bd_pin -dir I -type clk /%s/%s" % (node_name, clk_name))
            cmd.append("create_bd_pin -dir I -type rst /%s/%s" % (node_name, rst_name))
            cmd.append(
                "create_bd_intf_pin -mode Master "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, dout_name)
            )
            cmd.append(
                "create_bd_intf_pin -mode Slave "
                "-vlnv xilinx.com:interface:axis_rtl:1.0 /%s/%s" % (node_name, din_name)
            )

            # Instantiate HLS IP
            self.instantiate_ip(cmd)

            # Instantiate memstream
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
            axi_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/axi/hdl/")
            ms_rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/memstream/hdl/")
            file_suffix = "_memstream_wrapper.v"

            for fname in os.listdir(code_gen_dir):
                if fname.endswith(file_suffix):
                    strm_tmpl = fname
                    break

            strm_tmpl_name = strm_tmpl[:-2]
            sourcefiles = [
                os.path.join(code_gen_dir, strm_tmpl),
                axi_dir + "axilite.sv",
                ms_rtllib_dir + "memstream_axi.sv",
                ms_rtllib_dir + "memstream.sv",
            ]

            for f in sourcefiles:
                cmd += ["add_files -copy_to %s -norecurse %s" % (source_target, f)]

            strm_inst = node_name + "_wstrm"
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (strm_tmpl_name, node_name, strm_inst)
            )

            # Connect memstream to IP
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s/m_axis_0] "
                "[get_bd_intf_pins %s/%s/in1_V]" % (node_name, strm_inst, node_name, node_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_rst_n]"
                % (node_name, rst_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
                % (node_name, clk_name, node_name, strm_inst)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, rst_name, node_name, node_name, rst_name)
            )
            cmd.append(
                "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/%s]"
                % (node_name, clk_name, node_name, node_name, clk_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, din_name, node_name, node_name, din_name)
            )
            cmd.append(
                "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                "[get_bd_intf_pins %s/%s/%s]"
                % (node_name, dout_name, node_name, node_name, dout_name)
            )

            if runtime_writeable:
                axilite_name = self.get_verilog_top_module_intf_names()["axilite"][0]
                cmd.append(
                    "create_bd_intf_pin -mode Slave "
                    "-vlnv xilinx.com:interface:aximm_rtl:1.0 /%s/%s" % (node_name, axilite_name)
                )
                cmd.append(
                    "connect_bd_intf_net [get_bd_intf_pins %s/%s] "
                    "[get_bd_intf_pins %s/%s/%s]"
                    % (node_name, axilite_name, node_name, strm_inst, axilite_name)
                )
                cmd.append("assign_bd_address")

            cmd.append("save_bd_design")

        elif mem_mode in ["internal_embedded", "external"]:
            self.instantiate_ip(cmd)
        else:
            raise Exception("Unrecognized mem_mode for VectorVectorActivation")

        return cmd
