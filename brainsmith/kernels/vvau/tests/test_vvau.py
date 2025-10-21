############################################################################
# Copyright (C) 2025, Advanced Micro Devices, Inc.
# All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Migration to KernelOp by Microsoft Corporation
############################################################################

import pytest
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.datatype import DataType
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes

from brainsmith.kernels.vvau import VectorVectorActivation, VectorVectorActivation_hls


class TestVVAUInference:
    """Test VectorVectorActivation inference from MatMul with sparsity."""

    def test_can_infer_from_matmul_with_sparsity(self):
        """Test can_infer_from() correctly identifies MatMul with sparsity annotation."""
        # Create a simple MatMul node with sparsity annotation
        # For a depthwise conv with k_h=3, k_w=3, channels=4:
        # Weight matrix shape: (k_h*k_w*channels, channels) = (36, 4)
        channels = 4
        k_h, k_w = 3, 3
        dim_h, dim_w = 28, 28

        # Input shape after Im2Col: (1, 28, 28, 36)
        # Output shape: (1, 28, 28, 4)

        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, dim_h, dim_w, k_h * k_w * channels]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, dim_h, dim_w, channels]
        )

        # Create sparse depthwise weight matrix
        W_sparse = np.zeros((k_h * k_w * channels, channels), dtype=np.float32)
        for ch in range(channels):
            # Fill diagonal elements (depthwise structure)
            for i in range(k_h * k_w):
                W_sparse[ch * k_h * k_w + i, ch] = np.random.rand()

        # Create MatMul node
        node = helper.make_node(
            "MatMul",
            inputs=["input", "weights"],
            outputs=["output"],
            name="MatMul_0"
        )

        # Create graph and model
        graph_proto = helper.make_graph(
            [node], "test_vvau", [input_tensor], [output_tensor]
        )
        model_proto = helper.make_model(graph_proto, producer_name="brainsmith_test")
        model_proto.opset_import[0].version = 11

        model = ModelWrapper(model_proto)
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Set datatypes
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["INT8"])

        # Set weight initializer and sparsity annotation
        model.set_initializer("weights", W_sparse)
        model.set_tensor_datatype("weights", DataType["INT8"])
        model.set_tensor_sparsity("weights", {"dw": {"kernel_shape": [k_h, k_w]}})

        # Test can_infer_from
        assert VectorVectorActivation.can_infer_from(node, model)

    def test_cannot_infer_from_regular_matmul(self):
        """Test can_infer_from() rejects MatMul without sparsity annotation."""
        input_tensor = helper.make_tensor_value_info(
            "input", TensorProto.FLOAT, [1, 10]
        )
        output_tensor = helper.make_tensor_value_info(
            "output", TensorProto.FLOAT, [1, 5]
        )

        node = helper.make_node(
            "MatMul",
            inputs=["input", "weights"],
            outputs=["output"],
            name="MatMul_0"
        )

        graph_proto = helper.make_graph(
            [node], "test_model", [input_tensor], [output_tensor]
        )
        model_proto = helper.make_model(graph_proto, producer_name="test")
        model_proto.opset_import[0].version = 11

        model = ModelWrapper(model_proto)
        model.set_tensor_datatype("input", DataType["INT8"])
        model.set_tensor_datatype("output", DataType["INT8"])

        # No sparsity annotation
        W = np.random.rand(10, 5).astype(np.float32)
        model.set_initializer("weights", W)
        model.set_tensor_datatype("weights", DataType["INT8"])

        # Should not be able to infer
        assert not VectorVectorActivation.can_infer_from(node, model)


class TestVVAUShapes:
    """Test VectorVectorActivation shape methods."""

    @pytest.fixture
    def vvau_node(self):
        """Create a VVAU node for testing."""
        channels = 16
        k_h, k_w = 3, 3
        dim_h, dim_w = 28, 28

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[dim_h, dim_w],
            Channels=channels,
            Kernel=[k_h, k_w],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        return node

    def test_normal_input_shape(self, vvau_node):
        """Test get_normal_input_shape() returns correct shape."""
        op = VectorVectorActivation(vvau_node)

        # Expected: (1, dim_h, dim_w, channels * k_h * k_w)
        expected_shape = (1, 28, 28, 16 * 3 * 3)
        assert op.get_normal_input_shape(0) == expected_shape

    def test_normal_output_shape(self, vvau_node):
        """Test get_normal_output_shape() returns correct shape."""
        op = VectorVectorActivation(vvau_node)

        # Expected: (1, dim_h, dim_w, channels)
        expected_shape = (1, 28, 28, 16)
        assert op.get_normal_output_shape() == expected_shape

    def test_folded_input_shape(self, vvau_node):
        """Test get_folded_input_shape() returns correct folded shape."""
        op = VectorVectorActivation(vvau_node)

        # With PE=4, SIMD=3, Channels=16, Kernel=3x3:
        # sf = (k_h * k_w) // SIMD = 9 // 3 = 3
        # nf = Channels // PE = 16 // 4 = 4
        # Expected: (1, dim_h, dim_w, sf*nf, SIMD*PE)
        #         = (1, 28, 28, 12, 12)
        expected_shape = (1, 28, 28, 12, 12)
        assert op.get_folded_input_shape(0) == expected_shape

    def test_folded_output_shape(self, vvau_node):
        """Test get_folded_output_shape() returns correct folded shape."""
        op = VectorVectorActivation(vvau_node)

        # Expected: (1, dim_h, dim_w, nf, PE)
        #         = (1, 28, 28, 4, 4)
        expected_shape = (1, 28, 28, 4, 4)
        assert op.get_folded_output_shape() == expected_shape


class TestVVAUBackends:
    """Test VectorVectorActivation backend instantiation."""

    def test_hls_backend_instantiation(self):
        """Test VectorVectorActivation_hls can be instantiated."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=8,
            SIMD=3,
            Dim=[28, 28],
            Channels=32,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_hls(node)
        assert op is not None
        assert op.get_nodeattr("PE") == 8
        assert op.get_nodeattr("SIMD") == 3
        assert op.get_nodeattr("Channels") == 32

    def test_base_kernel_instantiation(self):
        """Test VectorVectorActivation base class can be instantiated."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=16,
            SIMD=1,
            Dim=[14, 14],
            Channels=64,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)
        assert op is not None
        assert op.get_nodeattr("PE") == 16


class TestVVAUMemoryCalculations:
    """Test VVAU memory calculations."""

    def test_calc_wmem(self):
        """Test calc_wmem() returns correct weight memory depth."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)

        # WMEM = (k_h * k_w * Channels // PE) // SIMD
        #      = (3 * 3 * 16 // 4) // 3
        #      = (144 // 4) // 3
        #      = 36 // 3
        #      = 12
        expected_wmem = 12
        assert op.calc_wmem() == expected_wmem

    def test_calc_tmem_no_activation(self):
        """Test calc_tmem() returns 0 when no_activation=1."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=1,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)
        assert op.calc_tmem() == 0

    def test_calc_tmem_with_activation(self):
        """Test calc_tmem() returns correct value when no_activation=0."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights", "thresholds"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=1,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=0,
        )

        op = VectorVectorActivation(node)

        # TMEM = Channels // PE = 16 // 4 = 4
        assert op.calc_tmem() == 4


class TestVVAUStreamWidths:
    """Test VVAU stream width calculations."""

    def test_instream_width_data(self):
        """Test get_instream_width(0) for data input."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)

        # Stream width = input_bits * SIMD * PE
        #              = 8 * 3 * 4 = 96
        assert op.get_instream_width(0) == 96

    def test_outstream_width(self):
        """Test get_outstream_width() for output."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=8,
            SIMD=1,
            Dim=[14, 14],
            Channels=32,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)

        # Stream width = output_bits * PE
        #              = 8 * 8 = 64
        assert op.get_outstream_width() == 64


class TestVVAUDatatypes:
    """Test VVAU datatype methods."""

    def test_get_input_datatype(self):
        """Test get_input_datatype() returns correct datatypes."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=1,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT4",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation(node)

        assert op.get_input_datatype(0) == DataType["INT8"]
        assert op.get_input_datatype(1) == DataType["INT4"]

    def test_get_output_datatype(self):
        """Test get_output_datatype() returns correct datatype."""
        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=1,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="UINT4",
            no_activation=1,
        )

        op = VectorVectorActivation(node)

        assert op.get_output_datatype() == DataType["UINT4"]


class TestVVAURTLBackend:
    """Test VectorVectorActivation_rtl backend."""

    def test_rtl_backend_instantiation(self):
        """Test VectorVectorActivation_rtl can be instantiated."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=8,
            SIMD=3,
            Dim=[28, 28],
            Channels=32,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_rtl(node)
        assert op is not None
        assert op.get_nodeattr("PE") == 8
        assert op.get_nodeattr("SIMD") == 3

    def test_rtl_dsp_estimation(self):
        """Test DSP estimation for RTL backend."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=9,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_rtl(node)

        # DSP estimate = PE * ceil(SIMD / 3)
        # = 4 * ceil(9 / 3) = 4 * 3 = 12
        assert op.dsp_estimation("xcve2802") == 12

    def test_rtl_lut_estimation(self):
        """Test LUT estimation for RTL backend (should be 0)."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_rtl(node)
        assert op.lut_estimation() == 0

    def test_rtl_resolve_segment_len(self):
        """Test pipeline segment length calculation."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=9,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_rtl(node)

        # With clk = 5.0 ns:
        # critical_path_dsps = floor((5.0 - 0.741) / 0.605 + 1) = floor(8.04) = 8
        # max_chain_len = ceil(9 / 3) = 3
        # segment_len = min(8, 3) = 3
        assert op._resolve_segment_len(5.0) == 3

        # With clk = 2.0 ns (tighter):
        # critical_path_dsps = floor((2.0 - 0.741) / 0.605 + 1) = floor(3.08) = 3
        # max_chain_len = 3
        # segment_len = min(3, 3) = 3
        assert op._resolve_segment_len(2.0) == 3

    def test_rtl_resolve_dsp_version(self):
        """Test DSP version selection (should be 3 for Versal)."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
            res_type="dsp",
        )

        op = VectorVectorActivation_rtl(node)

        # Versal device should return version 3 (DSP58)
        assert op._resolve_dsp_version("xcve2802") == 3

    def test_rtl_file_list(self):
        """Test get_rtl_file_list() returns correct files."""
        from brainsmith.kernels.vvau import VectorVectorActivation_rtl

        node = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        op = VectorVectorActivation_rtl(node)
        op.set_nodeattr("gen_top_module", "test_vvau_rtl")

        # Get file list (relative paths)
        files = op.get_rtl_file_list(abspath=False)

        # Should contain wrapper and RTL library files
        assert "test_vvau_rtl_wrapper.v" in files
        assert any("mvu_pkg.sv" in f for f in files)
        assert any("mvu_vvu_axi.sv" in f for f in files)
        assert any("mvu.sv" in f for f in files)


class TestVVAUComparison:
    """Compare HLS and RTL backends."""

    def test_both_backends_have_same_interface(self):
        """Test HLS and RTL backends have same basic interface."""
        from brainsmith.kernels.vvau import VectorVectorActivation_hls, VectorVectorActivation_rtl

        node_hls = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        node_rtl = helper.make_node(
            "VectorVectorActivation",
            inputs=["input", "weights"],
            outputs=["output"],
            domain="brainsmith.kernels",
            PE=4,
            SIMD=3,
            Dim=[28, 28],
            Channels=16,
            Kernel=[3, 3],
            input_dtype="INT8",
            weight_dtype="INT8",
            output_dtype="INT8",
            no_activation=1,
        )

        hls_op = VectorVectorActivation_hls(node_hls)
        rtl_op = VectorVectorActivation_rtl(node_rtl)

        # Both should have same basic properties
        assert hls_op.get_nodeattr("PE") == rtl_op.get_nodeattr("PE")
        assert hls_op.get_nodeattr("SIMD") == rtl_op.get_nodeattr("SIMD")
        assert hls_op.get_nodeattr("Channels") == rtl_op.get_nodeattr("Channels")

        # Both should compute same shapes
        assert hls_op.get_normal_input_shape() == rtl_op.get_normal_input_shape()
        assert hls_op.get_normal_output_shape() == rtl_op.get_normal_output_shape()
        assert hls_op.get_folded_input_shape() == rtl_op.get_folded_input_shape()
        assert hls_op.get_folded_output_shape() == rtl_op.get_folded_output_shape()

        # Both should compute same stream widths
        assert hls_op.get_instream_width(0) == rtl_op.get_instream_width(0)
        assert hls_op.get_outstream_width() == rtl_op.get_outstream_width()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
