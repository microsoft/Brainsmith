############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import numpy as np
import os
from qonnx.core.datatype import DataType

from finn.custom_op.fpgadataflow.rtlbackend import RTLBackend
from finn.util.basic import is_versal
from finn.util.data_packing import npy_to_rtlsim_input, rtlsim_output_to_npy

from brainsmith.kernels.vvau.vvau import VectorVectorActivation
from brainsmith.core.plugins import backend


@backend(
    name="VectorVectorActivationRTL",
    kernel="VectorVectorActivation",
    style="rtl",
    description="RTL implementation of VectorVectorActivation using DSP58-based MVU",
    author="Microsoft Corporation"
)
class VectorVectorActivation_rtl(VectorVectorActivation, RTLBackend):
    """RTL backend for VectorVectorActivation (KernelOp-based).

    This backend uses finn-rtllib's DSP58-based Matrix-Vector Unit (MVU)
    implementation optimized for Versal devices. Provides predictable
    performance and resource usage.

    Key features:
    - DSP58-based computation (3 multipliers per DSP)
    - Automatic pipeline register insertion for timing closure
    - Support for all memory modes (internal_embedded, internal_decoupled, external)
    - Narrow-range weight optimization
    - Versal-only (DSP58 requirement)

    Arete principles:
    - Shapes from kernel_instance (not nodeattrs)
    - Backend registration via decorator
    - Unified architecture with HLS variant
    """

    def __init__(self, onnx_node, **kwargs):
        super().__init__(onnx_node, **kwargs)

    def get_nodeattr_types(self):
        """Define nodeattrs for VectorVectorActivation_rtl backend."""
        my_attrs = {}
        my_attrs.update(VectorVectorActivation.get_nodeattr_types(self))
        my_attrs.update(RTLBackend.get_nodeattr_types(self))
        return my_attrs

    # ================================================================
    # Resource Estimation
    # ================================================================

    def lut_estimation(self):
        """Return LUT estimate (RTL uses DSPs primarily, minimal LUT usage)."""
        return 0

    def dsp_estimation(self, fpgapart):
        """Return DSP estimate for RTL implementation.

        DSP58 packs 3 8-bit multipliers, so:
        DSPs = PE * ceil(SIMD / 3)
        """
        P = self.get_nodeattr("PE")
        Q = self.get_nodeattr("SIMD")
        return int(P * np.ceil(Q / 3))

    def bram_estimation(self):
        """Return BRAM estimate (weights handled externally or via memstream)."""
        return 0

    def uram_estimation(self):
        """Return URAM estimate (not used in RTL implementation)."""
        return 0

    # ================================================================
    # Execution
    # ================================================================

    def execute_node(self, context, graph):
        """Execute node in cppsim or rtlsim mode."""
        mode = self.get_nodeattr("exec_mode")
        mem_mode = self.get_nodeattr("mem_mode")
        node = self.onnx_node

        if mode == "cppsim":
            # Use parent class Python execution
            VectorVectorActivation.execute_node(self, context, graph)

        elif mode == "rtlsim":
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")

            # Create npy file for each input
            in_ind = 0
            for inputs in node.input:
                # First input is data, second is weights, third is thresholds
                if in_ind == 0:
                    assert str(context[inputs].dtype) == "float32", (
                        "Input datatype is not float32 as expected"
                    )

                    expected_inp_shape = self.get_folded_input_shape()
                    reshaped_input = context[inputs].reshape(expected_inp_shape)

                    if self.get_input_datatype(0) == DataType["BIPOLAR"]:
                        # Store bipolar activations as binary
                        reshaped_input = (reshaped_input + 1) / 2
                        export_idt = DataType["BINARY"]
                    else:
                        export_idt = self.get_input_datatype(0)

                    # Make copy before saving
                    reshaped_input = reshaped_input.copy()
                    np.save(
                        os.path.join(code_gen_dir, "input_{}.npy".format(in_ind)),
                        reshaped_input,
                    )

                elif in_ind > 2:
                    raise Exception("Unexpected input found for VectorVectorActivation")

                in_ind += 1

            # Run RTL simulation
            sim = self.get_rtlsim()
            nbits = self.get_instream_width(0)
            inp = npy_to_rtlsim_input(
                "{}/input_0.npy".format(code_gen_dir), export_idt, nbits
            )
            super().reset_rtlsim(sim)

            if mem_mode in ["external", "internal_decoupled"]:
                wnbits = self.get_instream_width(1)
                export_wdt = self.get_input_datatype(1)

                # Convert bipolar weights to binary for export
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
            out_npy_path = "{}/output.npy".format(code_gen_dir)
            out_shape = self.get_folded_output_shape()

            rtlsim_output_to_npy(
                output, out_npy_path, odt, out_shape, packed_bits, target_bits
            )

            # Load and reshape output
            output = np.load(out_npy_path)
            oshape = self.get_normal_output_shape()
            output = np.asarray([output], dtype=np.float32).reshape(*oshape)
            context[node.output[0]] = output

        else:
            raise Exception(
                f"Invalid exec_mode: {mode}. Must be 'cppsim' or 'rtlsim'"
            )

    # ================================================================
    # HDL Generation
    # ================================================================

    def generate_hdl(self, model, fpgapart, clk):
        """Generate HDL for RTL implementation.

        Args:
            model: ONNX model wrapper
            fpgapart: Target FPGA part
            clk: Target clock period (ns)
        """
        # Generate params as part of IP preparation
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        self.generate_params(model, code_gen_dir)

        # Get template and code generation dictionary
        template_path, code_gen_dict = self.prepare_codegen_default(fpgapart, clk)

        # Determine if weights are narrow range
        weights = model.get_initializer(self.onnx_node.input[1])
        wdt = self.get_input_datatype(1)
        narrow_weights = 0 if np.min(weights) == wdt.min() else 1
        code_gen_dict["$NARROW_WEIGHTS$"] = str(narrow_weights)

        # Add general parameters to dictionary
        code_gen_dict["$MODULE_NAME_AXI_WRAPPER$"] = [self.get_verilog_top_module_name()]

        # Save top module name for later reference
        self.set_nodeattr("gen_top_module", self.get_verilog_top_module_name())

        # Apply code generation to template
        with open(template_path, "r") as f:
            template_wrapper = f.read()

        for key in code_gen_dict:
            # Transform list into long string separated by '\n'
            code_gen_line = "\n".join(code_gen_dict[key])
            template_wrapper = template_wrapper.replace(key, code_gen_line)

        # Write synthesis variant
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(0)))

        # Write simulation variant
        with open(
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper_sim.v"),
            "w",
        ) as f:
            f.write(template_wrapper.replace("$FORCE_BEHAVIORAL$", str(1)))

        # Handle internal_decoupled mode
        if self.get_nodeattr("mem_mode") == "internal_decoupled":
            if self.get_nodeattr("ram_style") == "ultra" and not is_versal(fpgapart):
                runtime_writeable = self.get_nodeattr("runtime_writeable_weights")
                assert runtime_writeable == 1, (
                    "Layer with URAM weights must have runtime_writeable_weights=1 "
                    "if Ultrascale device is targeted"
                )
            self.generate_hdl_memstream(fpgapart)

        # Set ipgen_path and ip_path for FINN transformations
        self.set_nodeattr("ipgen_path", code_gen_dir)
        self.set_nodeattr("ip_path", code_gen_dir)

    def _resolve_segment_len(self, clk):
        """Calculate DSP chain segment length for target clock period.

        DSP58 timing model:
        - First DSP: ~0.741 ns delay
        - Subsequent DSPs: ~0.605 ns delay each

        Formula: clk >= (n_dsps - 1) * 0.605 + 0.741

        Args:
            clk: Target clock period (ns)

        Returns:
            Number of DSPs in pipeline segment
        """
        assert clk > 0.741, (
            f"Infeasible clk target of {clk} ns has been set. "
            "Consider lowering the targeted clock frequency!"
        )

        # Calculate how many DSPs can fit in one clock cycle
        critical_path_dsps = np.floor((clk - 0.741) / 0.605 + 1)

        # Clamp to maximum chain length (SIMD / 3 since DSP58 packs 3 mults)
        max_chain_len = np.ceil(self.get_nodeattr("SIMD") / 3)
        dsp_chain_len = critical_path_dsps if critical_path_dsps < max_chain_len else max_chain_len

        return dsp_chain_len

    def _resolve_dsp_version(self, fpgapart):
        """Resolve DSP version based on target device.

        RTL implementation currently only supports DSP58 (Versal).

        Args:
            fpgapart: Target FPGA part

        Returns:
            DSP version identifier (3 = DSP58)
        """
        # Check resource type
        assert self.get_nodeattr("res_type") != "lut", (
            f"LUT-based RTL-VVU implementation currently not supported! "
            f"Please change res_type for {self.onnx_node.name} to 'dsp' or "
            f"consider switching to HLS-based VVAU!"
        )

        # Check for Versal family
        is_versal_family = is_versal(fpgapart)
        assert is_versal_family, (
            "DSP-based (RTL) VVU currently only supported on Versal (DSP58) devices"
        )

        return 3  # DSP58 version

    def prepare_codegen_default(self, fpgapart, clk):
        """Prepare code generation dictionary for RTL template.

        Args:
            fpgapart: Target FPGA part
            clk: Target clock period (ns)

        Returns:
            Tuple of (template_path, code_gen_dict)
        """
        template_path = os.environ["FINN_ROOT"] + "/finn-rtllib/mvu/mvu_vvu_axi_wrapper.v"

        code_gen_dict = {}
        code_gen_dict["$IS_MVU$"] = [str(0)]  # VVU mode (not MVU)
        code_gen_dict["$VERSION$"] = [str(self._resolve_dsp_version(fpgapart))]
        code_gen_dict["$PUMPED_COMPUTE$"] = [str(0)]  # No compute pumping

        # Matrix dimensions
        mw = int(np.prod(self.get_nodeattr("Kernel")))
        code_gen_dict["$MW$"] = [str(mw)]
        code_gen_dict["$MH$"] = [str(self.get_nodeattr("Channels"))]

        # Parallelization
        code_gen_dict["$PE$"] = [str(self.get_nodeattr("PE"))]
        code_gen_dict["$SIMD$"] = [str(self.get_nodeattr("SIMD"))]

        # Datatypes
        code_gen_dict["$ACTIVATION_WIDTH$"] = [str(self.get_input_datatype(0).bitwidth())]
        code_gen_dict["$WEIGHT_WIDTH$"] = [str(self.get_input_datatype(1).bitwidth())]
        code_gen_dict["$ACCU_WIDTH$"] = [str(self.get_output_datatype().bitwidth())]

        # Signedness
        code_gen_dict["$SIGNED_ACTIVATIONS$"] = (
            [str(1)] if (self.get_input_datatype(0).min() < 0) else [str(0)]
        )

        # Pipeline depth
        code_gen_dict["$SEGMENTLEN$"] = [str(self._resolve_segment_len(clk))]

        return template_path, code_gen_dict

    # ================================================================
    # File Management
    # ================================================================

    def get_rtl_file_list(self, abspath=False):
        """Get list of RTL source files.

        Args:
            abspath: If True, return absolute paths

        Returns:
            List of RTL file paths
        """
        if abspath:
            code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen") + "/"
            rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")
        else:
            code_gen_dir = ""
            rtllib_dir = ""

        verilog_files = [
            "mvu_pkg.sv",
            "mvu_vvu_axi.sv",
            "replay_buffer.sv",
            "mvu.sv",
            "mvu_vvu_8sx9_dsp58.sv",
            "add_multi.sv",
        ]
        verilog_files = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v")
        ] + [rtllib_dir + _ for _ in verilog_files]

        return verilog_files

    def get_verilog_paths(self):
        """Get list of directories containing Verilog sources."""
        verilog_paths = super().get_verilog_paths()
        verilog_paths.append(os.environ["FINN_ROOT"] + "/finn-rtllib/mvu")
        return verilog_paths

    # ================================================================
    # IPI Integration
    # ================================================================

    def instantiate_ip(self, cmd):
        """Generate TCL commands to instantiate RTL IP in Vivado IPI.

        Args:
            cmd: List to append TCL commands to
        """
        node_name = self.onnx_node.name
        code_gen_dir = self.get_nodeattr("code_gen_dir_ipgen")
        rtllib_dir = os.path.join(os.environ["FINN_ROOT"], "finn-rtllib/mvu/")

        # Source files
        sourcefiles = [
            "mvu_pkg.sv",
            "mvu_vvu_axi.sv",
            "replay_buffer.sv",
            "mvu.sv",
            "mvu_vvu_8sx9_dsp58.sv",
            "add_multi.sv",
        ]
        sourcefiles = [
            os.path.join(code_gen_dir, self.get_nodeattr("gen_top_module") + "_wrapper.v")
        ] + [rtllib_dir + _ for _ in sourcefiles]

        # Add files
        for f in sourcefiles:
            cmd.append("add_files -norecurse %s" % (f))

        # Create BD cell
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode == "internal_decoupled":
            cmd.append(
                "create_bd_cell -type hier -reference %s /%s/%s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                    node_name,
                )
            )
        else:
            cmd.append(
                "create_bd_cell -type hier -reference %s %s"
                % (
                    self.get_nodeattr("gen_top_module"),
                    node_name,
                )
            )

        # Connect 2x clk to regular clk port
        clk_name = self.get_verilog_top_module_intf_names()["clk"][0]
        cmd.append(
            "connect_bd_net [get_bd_pins %s/%s] [get_bd_pins %s/%s/ap_clk2x]"
            % (node_name, clk_name, node_name, node_name)
        )

    def code_generation_ipi(self):
        """Generate complete TCL script for IPI integration.

        Returns:
            List of TCL commands
        """
        cmd = []
        self.instantiate_ip(cmd)
        return cmd

    # ================================================================
    # FINN Integration
    # ================================================================

    def get_verilog_top_module_name(self):
        """Get Verilog top module name."""
        return f"{self.onnx_node.name}_VectorVectorActivation_rtl"

    def get_verilog_top_module_intf_names(self):
        """Get Verilog top module interface names."""
        # Get base interface names from parent
        intf_names = {
            "clk": ["ap_clk"],
            "rst": ["ap_rst_n"],
            "s_axis": [("in0_V_tdata", self.get_instream_width(0))],
            "m_axis": [("out0_V_tdata", self.get_outstream_width())],
        }

        # Add weight stream for external/decoupled modes
        mem_mode = self.get_nodeattr("mem_mode")
        if mem_mode in ["external", "internal_decoupled"]:
            intf_names["s_axis"].append(("in1_V_tdata", self.get_instream_width(1)))

        return intf_names
