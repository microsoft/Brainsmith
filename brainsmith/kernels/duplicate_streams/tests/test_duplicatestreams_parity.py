# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Parity tests: FINN DuplicateStreams vs Brainsmith DuplicateStreams

This test suite validates that Brainsmith's DuplicateStreams implementation
is functionally equivalent to FINN's reference implementation.

Test Coverage:
- Base parity: 25 tests (shapes, datatypes, widths, cycles, execution)
- HLS parity: 25 tests + 7 HLS codegen tests + 3 resource value tests
- Total: ~60 automated tests

Why parity testing?
- Validates correctness against proven FINN reference
- Catches divergence immediately
- Includes cppsim compilation + execution validation
- Reduces maintenance burden (auto-generated tests)
"""

import pytest
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.custom_op.registry import getCustomOp

from tests.parity.base_parity_test import ParityTestBase
from tests.parity.hls_codegen_parity import HLSCodegenParityMixin


class TestDuplicateStreamsBaseParity(ParityTestBase):
    """Parity: FINN DuplicateStreams vs Brainsmith DuplicateStreams (base)

    Validates that Brainsmith's base DuplicateStreams kernel matches FINN's
    reference implementation across all HWCustomOp methods.

    Runs 25 automatic tests:
    - 15 base tests (shapes, datatypes, widths, cycles, execution)
    - 1 cppsim test (skipped for base class, only applicable to HLS)
    - 6 RTL tests (resource estimation methods)
    - 3 efficiency tests

    All tests compare FINN (manual) vs Brainsmith (auto) implementations.
    """

    @property
    def manual_op_class(self):
        """FINN reference implementation"""
        from finn.custom_op.fpgadataflow.duplicatestreams import DuplicateStreams
        return DuplicateStreams

    @property
    def auto_op_class(self):
        """Brainsmith implementation (being validated)"""
        from brainsmith.kernels.duplicate_streams.duplicate_streams import DuplicateStreams
        return DuplicateStreams

    def make_test_model(self):
        """Create ONNX model with DuplicateStreams node

        Creates a model with:
        - Shape: [1, 8, 8, 64] (batch=1, H=8, W=8, C=64)
        - PE: 8 (8-way parallelization)
        - 2 output streams
        - INT8 datatype

        Returns:
            (model, node_name): ModelWrapper and name of DuplicateStreams node
        """
        # Create DuplicateStreams node with FINN domain
        node = helper.make_node(
            "DuplicateStreams",
            inputs=["inp"],
            outputs=["out0", "out1"],
            domain="finn.custom_op.fpgadataflow",
            NumChannels=64,
            PE=8,
            NumOutputStreams=2,
            inputDataType="INT8",
            numInputVectors=[1, 8, 8]
        )

        shape = [1, 8, 8, 64]
        inp_vi = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
        out0_vi = helper.make_tensor_value_info("out0", TensorProto.FLOAT, shape)
        out1_vi = helper.make_tensor_value_info("out1", TensorProto.FLOAT, shape)

        graph = helper.make_graph([node], "test", [inp_vi], [out0_vi, out1_vi])
        model = ModelWrapper(helper.make_model(graph))

        # Set datatypes
        model.set_tensor_datatype("inp", DataType["INT8"])
        model.set_tensor_datatype("out0", DataType["INT8"])
        model.set_tensor_datatype("out1", DataType["INT8"])

        return model, node.name

    def get_num_outputs(self):
        """DuplicateStreams creates 2 output streams"""
        return 2

    def setup_manual_op(self):
        """Setup FINN reference implementation

        Creates a DuplicateStreams node with FINN domain and returns
        the operator instance.

        Returns:
            (op, model): FINN DuplicateStreams operator and model
        """
        model, node_name = self.make_test_model()

        # Find the DuplicateStreams node
        node = None
        for n in model.graph.node:
            if n.op_type == "DuplicateStreams":
                node = n
                break

        assert node is not None, "DuplicateStreams node not found in model"

        # Get FINN operator
        op = getCustomOp(node)
        return op, model

    def setup_auto_op(self):
        """Setup Brainsmith implementation

        Creates a DuplicateStreams node with Brainsmith domain and returns
        the operator instance with design space built.

        Returns:
            (op, model): Brainsmith DuplicateStreams operator and model
        """
        model, node_name = self.make_test_model()

        # Change domain to brainsmith
        node = None
        for n in model.graph.node:
            if n.op_type == "DuplicateStreams":
                n.domain = "brainsmith.kernels"
                node = n
                break

        assert node is not None, "DuplicateStreams node not found in model"

        # Get Brainsmith operator
        op = getCustomOp(node)

        # Build design space for Brainsmith KernelOp
        if hasattr(op, 'build_design_space'):
            op.build_design_space(model)

        return op, model


class TestDuplicateStreamsHLSParity(ParityTestBase, HLSCodegenParityMixin):
    """Parity: FINN DuplicateStreams_hls vs Brainsmith DuplicateStreams_hls

    Validates that Brainsmith's HLS backend matches FINN's reference HLS
    implementation across all code generation and execution methods.

    Runs 35 automatic tests:
    - 25 base tests (shapes, datatypes, widths, cycles, execution)
    - 7 HLS codegen tests:
      * global_includes() - Header includes
      * defines() - Macro definitions
      * pragmas() - HLS synthesis pragmas
      * docompute() - Main computation body
      * blackboxfunction() - Function signature
      * strm_decl() - Stream declarations
      * dataoutstrm() - Output stream handling
    - 3 resource estimation value tests:
      * test_lut_estimation_value() - LUT count validation
      * test_bram_estimation_value() - BRAM count validation (should be 0)
      * test_dsp_estimation_value() - DSP count validation (should be 0)

    CRITICAL TEST: test_cppsim_execution_parity()
    - Generates C++ code for both backends
    - Compiles C++ with Vivado HLS
    - Executes compiled binaries
    - Validates outputs match exactly

    This is the gold standard for HLS correctness validation.
    """

    @property
    def manual_op_class(self):
        """FINN HLS reference implementation"""
        from finn.custom_op.fpgadataflow.hls.duplicatestreams_hls import DuplicateStreams_hls
        return DuplicateStreams_hls

    @property
    def auto_op_class(self):
        """Brainsmith HLS implementation (being validated)"""
        from brainsmith.kernels.duplicate_streams.duplicate_streams_hls import DuplicateStreams_hls
        return DuplicateStreams_hls

    def make_test_model(self):
        """Create ONNX model with DuplicateStreams node

        Same as base test model - will be specialized to HLS domain
        in setup methods.

        Returns:
            (model, node_name): ModelWrapper and name of DuplicateStreams node
        """
        # Create DuplicateStreams node (will be specialized to HLS)
        node = helper.make_node(
            "DuplicateStreams",
            inputs=["inp"],
            outputs=["out0", "out1"],
            domain="finn.custom_op.fpgadataflow",
            NumChannels=64,
            PE=8,
            NumOutputStreams=2,
            inputDataType="INT8",
            numInputVectors=[1, 8, 8]
        )

        shape = [1, 8, 8, 64]
        inp_vi = helper.make_tensor_value_info("inp", TensorProto.FLOAT, shape)
        out0_vi = helper.make_tensor_value_info("out0", TensorProto.FLOAT, shape)
        out1_vi = helper.make_tensor_value_info("out1", TensorProto.FLOAT, shape)

        graph = helper.make_graph([node], "test", [inp_vi], [out0_vi, out1_vi])
        model = ModelWrapper(helper.make_model(graph))

        # Set datatypes
        model.set_tensor_datatype("inp", DataType["INT8"])
        model.set_tensor_datatype("out0", DataType["INT8"])
        model.set_tensor_datatype("out1", DataType["INT8"])

        return model, node.name

    def get_num_outputs(self):
        """DuplicateStreams creates 2 output streams"""
        return 2

    def setup_manual_op(self):
        """Setup FINN HLS backend via direct instantiation

        Returns:
            (op, model): FINN DuplicateStreams_hls operator and model
        """
        model, _ = self.make_test_model()

        # Change to FINN HLS domain
        node = None
        for n in model.graph.node:
            if n.op_type == "DuplicateStreams":
                n.domain = "finn.custom_op.fpgadataflow.hls"
                n.op_type = "DuplicateStreams_hls"
                node = n
                break

        assert node is not None, "DuplicateStreams node not found in model"

        # Get FINN HLS operator
        op = getCustomOp(node)
        return op, model

    def setup_auto_op(self):
        """Setup Brainsmith HLS backend

        Returns:
            (op, model): Brainsmith DuplicateStreams_hls operator and model
        """
        model, _ = self.make_test_model()

        # Change to Brainsmith HLS domain
        node = None
        for n in model.graph.node:
            if n.op_type == "DuplicateStreams":
                n.domain = "brainsmith.kernels.hls"
                n.op_type = "DuplicateStreams_hls"
                node = n
                break

        assert node is not None, "DuplicateStreams node not found in model"

        # Get Brainsmith HLS operator
        op = getCustomOp(node)

        # Build design space for Brainsmith KernelOp
        if hasattr(op, 'build_design_space'):
            op.build_design_space(model)

        return op, model

    # ================================================================
    # Resource Estimation Value Tests
    # ================================================================

    def test_lut_estimation_value(self):
        """Validate LUT estimation is reasonable for simple wire fanout.

        DuplicateStreams is a simple fanout (read once, write N times) with
        minimal control logic. LUT usage should be very low (<200 LUTs).

        This test validates:
        - FINN and Brainsmith produce identical estimates
        - Estimate is reasonable for combinational fanout logic
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_lut = manual_op.lut_estimation()
        auto_lut = auto_op.lut_estimation()

        # Verify parity
        assert manual_lut == auto_lut, (
            f"LUT estimation mismatch: FINN={manual_lut}, Brainsmith={auto_lut}"
        )

        # Verify reasonableness (simple wire fanout should be < 200 LUTs)
        assert auto_lut < 200, (
            f"DuplicateStreams LUT estimate too high: {auto_lut} LUTs. "
            f"Expected < 200 for simple fanout logic."
        )

    def test_bram_estimation_value(self):
        """Validate BRAM estimation is 0 (combinational fanout, no buffering).

        DuplicateStreams is purely combinational - it reads once and writes
        to all outputs in the same cycle. No BRAMs should be required.

        This test validates:
        - FINN and Brainsmith both estimate 0 BRAMs
        - Confirms no accidental buffering in implementation
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        manual_bram = manual_op.bram_estimation()
        auto_bram = auto_op.bram_estimation()

        # Verify parity
        assert manual_bram == auto_bram, (
            f"BRAM estimation mismatch: FINN={manual_bram}, Brainsmith={auto_bram}"
        )

        # Verify correctness (pure combinational logic, no storage)
        assert auto_bram == 0, (
            f"DuplicateStreams BRAM estimate should be 0 (combinational fanout), "
            f"got {auto_bram}. Check for accidental buffering."
        )

    def test_dsp_estimation_value(self):
        """Validate DSP estimation is 0 (routing only, no arithmetic).

        DuplicateStreams performs no computation - it's pure routing/fanout.
        No DSP blocks should be required.

        This test validates:
        - FINN and Brainsmith both estimate 0 DSPs
        - Confirms no accidental arithmetic in implementation
        """
        manual_op, _ = self.setup_manual_op()
        auto_op, _ = self.setup_auto_op()

        fpgapart = "xcve2802"  # Representative FPGA part
        manual_dsp = manual_op.dsp_estimation(fpgapart)
        auto_dsp = auto_op.dsp_estimation(fpgapart)

        # Verify parity
        assert manual_dsp == auto_dsp, (
            f"DSP estimation mismatch: FINN={manual_dsp}, Brainsmith={auto_dsp}"
        )

        # Verify correctness (pure routing, no arithmetic)
        assert auto_dsp == 0, (
            f"DuplicateStreams DSP estimate should be 0 (routing only), "
            f"got {auto_dsp}. Check for accidental arithmetic operations."
        )
