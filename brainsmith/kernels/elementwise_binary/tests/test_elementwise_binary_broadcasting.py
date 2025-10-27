############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################
"""Execution tests for ElementwiseBinaryOp broadcasting support (Phase 2).

Tests various broadcasting patterns in HLS code generation and execution:
- Scalar broadcasting: [1,64,64,128] + [1] → [1,64,64,128]
- Channel broadcasting: [1,64,64,128] + [128] → [1,64,64,128]
- Spatial broadcasting: [1,64,64,128] + [1,1,1,128] → [1,64,64,128]
- Bidirectional broadcasting: [1,64,1,128] + [1,1,64,1] → [1,64,64,128]
"""

import pytest
import numpy as np
from onnx import helper, TensorProto
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.util.basic import qonnx_make_model, gen_finn_dt_tensor
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.core.datatype import DataType
from finn.util.basic import getHWCustomOp

from brainsmith.transforms.infer_kernel_list import InferKernelList
from brainsmith.kernels.elementwise_binary.tests.test_elementwise_binary_parity import (
    SpecializeElementwiseBinaryToHLS
)


class TestElementwiseBinaryBroadcasting:
    """Test broadcasting patterns for ElementwiseBinaryOp.

    Covers Phase 2 functionality:
    - Dynamic + dynamic inputs with broadcasting
    - Various broadcast patterns (scalar, channel, spatial, bidirectional)
    - Correct HLS code generation for buffering and conditional reads
    """

    # =========================================================================
    # Test Model Creation Helpers
    # =========================================================================

    def make_broadcast_test_model(
        self,
        lhs_shape,
        rhs_shape,
        operation="Add",
        idt=DataType["INT8"]
    ):
        """Create ONNX model with broadcasting inputs.

        Args:
            lhs_shape: Shape of left-hand side input (dynamic)
            rhs_shape: Shape of right-hand side input (dynamic with broadcast)
            operation: ONNX operation name (Add, Mul, etc.)
            idt: Input/output datatype

        Returns:
            Tuple of (ModelWrapper, node_name)
        """
        node_name = f"{operation}_broadcast_test"

        # Compute output shape (numpy broadcast semantics)
        output_shape = tuple(np.broadcast_shapes(lhs_shape, rhs_shape))

        # Create input tensor infos
        lhs_input = helper.make_tensor_value_info("lhs", TensorProto.FLOAT, lhs_shape)
        rhs_input = helper.make_tensor_value_info("rhs", TensorProto.FLOAT, rhs_shape)
        output = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

        # Create ONNX node
        node = helper.make_node(
            operation, ["lhs", "rhs"], ["output"], name=node_name
        )

        # Build graph and model
        graph = helper.make_graph(
            nodes=[node],
            name="broadcast_test",
            inputs=[lhs_input, rhs_input],
            outputs=[output],
        )
        model = ModelWrapper(qonnx_make_model(graph, producer_name="broadcast-test"))

        # Set datatypes
        model.set_tensor_datatype("lhs", idt)
        model.set_tensor_datatype("rhs", idt)
        model.set_tensor_datatype("output", idt)

        return model, node_name

    def infer_to_hls_backend(self, model):
        """Apply transformations to get ElementwiseBinaryOp_hls instance.

        Args:
            model: ONNX ModelWrapper

        Returns:
            Tuple of (ElementwiseBinaryOp_hls instance, transformed model)
        """
        # Standard inference
        model = model.transform(InferShapes())
        model = model.transform(InferDataTypes())

        # Infer to ElementwiseBinaryOp
        model = model.transform(InferKernelList())

        # Specialize to HLS backend
        model = model.transform(SpecializeElementwiseBinaryToHLS())

        # Get HLS node
        assert len(model.graph.node) == 1, f"Expected 1 node, got {len(model.graph.node)}"
        hls_node = model.graph.node[0]

        assert hls_node.op_type == "ElementwiseBinaryOp_hls", (
            f"Expected ElementwiseBinaryOp_hls, got {hls_node.op_type}"
        )

        op = getHWCustomOp(hls_node, model)
        return op, model

    # =========================================================================
    # Broadcasting Pattern Tests
    # =========================================================================

    def test_scalar_broadcast(self):
        """Test scalar broadcasting: [1,64,64,128] + [1] → [1,64,64,128]"""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (1,)

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        # Verify pattern detection
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify broadcasting info cached correctly
        bcast_info = op._get_broadcast_info("rhs")
        assert bcast_info is not None
        assert bcast_info.has_broadcast is True
        assert bcast_info.lhs_shape == lhs_shape
        assert bcast_info.rhs_shape == rhs_shape  # Original shape, not padded
        assert bcast_info.output_shape == lhs_shape

    def test_channel_broadcast(self):
        """Test channel broadcasting: [1,64,64,128] + [128] → [1,64,64,128]"""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (128,)

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        # Verify pattern detection
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify broadcasting info
        bcast_info = op._get_broadcast_info("rhs")
        assert bcast_info is not None
        assert bcast_info.has_broadcast is True
        assert bcast_info.rhs_shape == rhs_shape  # Original shape
        assert bcast_info.broadcast_dims_rhs == (1, 2)  # Broadcasts in H, W (after padding)

    def test_spatial_broadcast(self):
        """Test spatial broadcasting: [1,64,64,128] + [1,1,1,128] → [1,64,64,128]"""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (1, 1, 1, 128)

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        # Verify pattern detection
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify broadcasting info
        bcast_info = op._get_broadcast_info("rhs")
        assert bcast_info is not None
        assert bcast_info.has_broadcast is True
        assert bcast_info.broadcast_dims_rhs == (1, 2)  # Broadcasts in H, W

    def test_bidirectional_broadcast(self):
        """Test bidirectional: [1,64,1,128] + [1,1,64,1] → [1,64,64,128]"""
        lhs_shape = (1, 64, 1, 128)
        rhs_shape = (1, 1, 64, 1)

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        # Verify pattern detection
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify broadcasting info (same object for both sides)
        bcast_info = op._get_broadcast_info("lhs")

        assert bcast_info is not None
        assert bcast_info.has_broadcast is True

        # LHS broadcasts in W (dim 2), RHS broadcasts in H and C (dims 1, 3)
        assert bcast_info.broadcast_dims_lhs == (2,)
        assert bcast_info.broadcast_dims_rhs == (1, 3)

    @pytest.mark.skip(reason="AddStreams kernel handles identical-shape dynamic+dynamic case")
    def test_no_broadcast_dynamic_dynamic(self):
        """Test dynamic+dynamic without broadcasting: [1,64,64,128] + [1,64,64,128]

        Note: This case is correctly inferred as AddStreams by InferKernelList,
        not ElementwiseBinaryOp. AddStreams is specialized for identical-shape
        dynamic tensor addition without broadcasting.
        """
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (1, 64, 64, 128)

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        # Verify pattern detection
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify NO broadcasting
        bcast_info = op._get_broadcast_info("lhs")
        assert bcast_info is not None
        assert bcast_info.has_broadcast is False

    # =========================================================================
    # HLS Code Generation Tests
    # =========================================================================
    # Note: Low-level HLS helper methods are thoroughly tested in
    # test_elementwise_binary_hls_helpers.py with mocked design_point.
    # These tests focus on end-to-end pattern detection and broadcasting info.

    def test_read_condition_with_broadcast(self):
        """Test that broadcast patterns generate correct read conditions."""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (1, 1, 1, 128)  # Spatial broadcast

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        loop_counters = ("i0", "i1", "i2")

        # LHS: no broadcast, always read
        lhs_cond = op._get_read_condition("lhs", loop_counters)
        assert lhs_cond == "true"

        # RHS: broadcast in H,W → conditional read
        rhs_cond = op._get_read_condition("rhs", loop_counters)
        assert rhs_cond != "true"
        assert "i1 == 0" in rhs_cond  # Only read when H index is 0
        assert "i2 == 0" in rhs_cond  # Only read when W index is 0

    def test_indexing_expression_with_broadcast(self):
        """Test that broadcast patterns generate correct indexing."""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (128,)  # Channel broadcast

        model, node_name = self.make_broadcast_test_model(lhs_shape, rhs_shape)
        op, model = self.infer_to_hls_backend(model)

        loop_counters = ("i0", "i1", "i2")

        # LHS: no broadcast, regular indexing
        lhs_index = op._get_indexing_expression("lhs", loop_counters, "pe")
        assert lhs_index == "[i0][i1][i2][pe]"

        # RHS: broadcast → fixed indices for broadcast dims
        rhs_index = op._get_indexing_expression("rhs", loop_counters, "pe")
        assert "[0]" in rhs_index  # Broadcast dimensions use 0
        assert "[pe]" in rhs_index  # PE dimension not broadcast

    # =========================================================================
    # Multiple Operations
    # =========================================================================

    @pytest.mark.parametrize("operation", ["Add", "Sub", "Mul"])
    def test_broadcast_multiple_ops(self, operation):
        """Test broadcasting works for multiple binary operations."""
        lhs_shape = (1, 64, 64, 128)
        rhs_shape = (128,)

        model, node_name = self.make_broadcast_test_model(
            lhs_shape, rhs_shape, operation=operation
        )
        op, model = self.infer_to_hls_backend(model)

        # Verify correct operation and pattern
        assert op.get_nodeattr("func") == operation
        assert op.get_nodeattr("input_pattern") == "dynamic_dynamic"

        # Verify broadcasting detected
        bcast_info = op._get_broadcast_info("rhs")
        assert bcast_info is not None
        assert bcast_info.has_broadcast is True
