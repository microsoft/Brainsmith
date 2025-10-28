############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################
"""Unit tests for ElementwiseBinaryOp HLS helper methods (Phase 2)."""

import pytest
import numpy as np
from unittest.mock import Mock, MagicMock
from qonnx.core.datatype import DataType

from brainsmith.kernels.elementwise_binary.elementwise_binary_hls import ElementwiseBinaryOp_hls
from brainsmith.dataflow.broadcast_helpers import BroadcastInfo


class TestNeedsStreamingInterface:
    """Tests for _needs_streaming_interface() method."""

    def test_dynamic_static_pattern_lhs(self):
        """LHS should be streaming in dynamic_static pattern."""
        # Create mock kernel with dynamic_static pattern
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel.get_nodeattr = Mock(return_value="dynamic_static")

        # Call method directly
        result = ElementwiseBinaryOp_hls._needs_streaming_interface(mock_kernel, "lhs")

        assert result is True

    def test_dynamic_static_pattern_rhs(self):
        """RHS should NOT be streaming in dynamic_static pattern."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel.get_nodeattr = Mock(return_value="dynamic_static")

        result = ElementwiseBinaryOp_hls._needs_streaming_interface(mock_kernel, "rhs")

        assert result is False

    def test_dynamic_dynamic_pattern_lhs(self):
        """LHS should be streaming in dynamic_dynamic pattern."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel.get_nodeattr = Mock(return_value="dynamic_dynamic")

        result = ElementwiseBinaryOp_hls._needs_streaming_interface(mock_kernel, "lhs")

        assert result is True

    def test_dynamic_dynamic_pattern_rhs(self):
        """RHS should be streaming in dynamic_dynamic pattern."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel.get_nodeattr = Mock(return_value="dynamic_dynamic")

        result = ElementwiseBinaryOp_hls._needs_streaming_interface(mock_kernel, "rhs")

        assert result is True

    def test_unknown_pattern_raises_error(self):
        """Unknown pattern should raise ValueError."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel.get_nodeattr = Mock(return_value="unknown_pattern")

        with pytest.raises(ValueError, match="Unknown input_pattern"):
            ElementwiseBinaryOp_hls._needs_streaming_interface(mock_kernel, "lhs")


class TestGetBufferDeclaration:
    """Tests for _get_buffer_declaration() method."""

    def test_static_input_no_buffer(self):
        """Static inputs should return None."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._needs_streaming_interface = Mock(return_value=False)

        result = ElementwiseBinaryOp_hls._get_buffer_declaration(mock_kernel, "rhs", pe=64)

        assert result is None

    def test_streaming_no_broadcast_simple_buffer(self):
        """Streaming without broadcast should use simple output-shaped buffer."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._needs_streaming_interface = Mock(return_value=True)
        mock_kernel._get_broadcast_info = Mock(return_value=None)

        # Mock design_point
        mock_output = Mock()
        mock_output.tensor_shape = (1, 64, 64, 128)
        mock_kernel.design_point = Mock()
        mock_kernel.design_point.outputs = {"output": mock_output}

        result = ElementwiseBinaryOp_hls._get_buffer_declaration(mock_kernel, "lhs", pe=64)

        # Should create buffer shaped [1][64][64][2] (128/64=2)
        assert result is not None
        assert "LhsType lhs[1][64][64][2];" in result.declaration
        assert "#pragma HLS ARRAY_PARTITION variable=lhs complete dim=4" == result.partition_pragma

    def test_streaming_with_broadcast_uses_broadcast_info(self):
        """Streaming with broadcast should use BroadcastInfo.get_buffer_shape()."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._needs_streaming_interface = Mock(return_value=True)

        # Mock broadcast info
        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = True
        mock_bcast_info.get_buffer_shape = Mock(return_value=(1, 1, 1, 2))  # Broadcast shape
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        result = ElementwiseBinaryOp_hls._get_buffer_declaration(mock_kernel, "rhs", pe=64)

        # Should use broadcast buffer shape
        assert result is not None
        assert "RhsType rhs[1][1][1][2];" in result.declaration
        assert "#pragma HLS ARRAY_PARTITION variable=rhs complete dim=4" == result.partition_pragma
        mock_bcast_info.get_buffer_shape.assert_called_once_with("rhs", 64)


class TestGetReadCondition:
    """Tests for _get_read_condition() method."""

    def test_no_broadcast_returns_true(self):
        """No broadcasting should return 'true' (always read)."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._get_broadcast_info = Mock(return_value=None)

        result = ElementwiseBinaryOp_hls._get_read_condition(
            mock_kernel, "lhs", ("i0", "i1", "i2")
        )

        assert result == "true"

    def test_no_broadcast_flag_returns_true(self):
        """BroadcastInfo without has_broadcast should return 'true'."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)

        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = False
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        result = ElementwiseBinaryOp_hls._get_read_condition(
            mock_kernel, "lhs", ("i0", "i1", "i2")
        )

        assert result == "true"

    def test_with_broadcast_returns_condition(self):
        """Broadcasting should return conditional expression."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)

        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = True
        mock_bcast_info.should_read_new_value = Mock(return_value="i1 == 0 && i2 == 0")
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        result = ElementwiseBinaryOp_hls._get_read_condition(
            mock_kernel, "rhs", ("i0", "i1", "i2")
        )

        assert result == "i1 == 0 && i2 == 0"
        mock_bcast_info.should_read_new_value.assert_called_once_with("rhs", ("i0", "i1", "i2"))

    def test_none_from_should_read_fallback_to_true(self):
        """If should_read_new_value returns None, should fallback to 'true'."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)

        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = True
        mock_bcast_info.should_read_new_value = Mock(return_value=None)
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        result = ElementwiseBinaryOp_hls._get_read_condition(
            mock_kernel, "lhs", ("i0", "i1")
        )

        assert result == "true"


class TestGetIndexingExpression:
    """Tests for _get_indexing_expression() method."""

    def test_no_broadcast_simple_indexing(self):
        """No broadcasting should use simple sequential indexing."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._get_broadcast_info = Mock(return_value=None)

        # Mock output shape
        mock_output = Mock()
        mock_output.tensor_shape = (1, 64, 64, 128)
        mock_kernel.design_point = Mock()
        mock_kernel.design_point.outputs = {"output": mock_output}

        result = ElementwiseBinaryOp_hls._get_indexing_expression(
            mock_kernel, "lhs", ("i0", "i1", "i2"), "pe"
        )

        # Should use all counters + PE
        assert result == "[i0][i1][i2][pe]"

    def test_no_broadcast_flag_simple_indexing(self):
        """BroadcastInfo without has_broadcast should use simple indexing."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)

        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = False
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        mock_output = Mock()
        mock_output.tensor_shape = (1, 64, 64, 128)
        mock_kernel.design_point = Mock()
        mock_kernel.design_point.outputs = {"output": mock_output}

        result = ElementwiseBinaryOp_hls._get_indexing_expression(
            mock_kernel, "rhs", ("i0", "i1", "i2"), "pe"
        )

        assert result == "[i0][i1][i2][pe]"

    def test_with_broadcast_uses_broadcast_info(self):
        """Broadcasting should use BroadcastInfo.get_index_expression()."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)

        mock_bcast_info = Mock(spec=BroadcastInfo)
        mock_bcast_info.has_broadcast = True
        mock_bcast_info.get_index_expression = Mock(return_value="[0][0][i2][pe]")
        mock_kernel._get_broadcast_info = Mock(return_value=mock_bcast_info)

        result = ElementwiseBinaryOp_hls._get_indexing_expression(
            mock_kernel, "rhs", ("i0", "i1", "i2"), "pe"
        )

        assert result == "[0][0][i2][pe]"
        mock_bcast_info.get_index_expression.assert_called_once_with("rhs", ("i0", "i1", "i2"), "pe")

    def test_custom_pe_variable(self):
        """Should support custom PE variable name."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._get_broadcast_info = Mock(return_value=None)

        mock_output = Mock()
        mock_output.tensor_shape = (1, 64, 128)
        mock_kernel.design_point = Mock()
        mock_kernel.design_point.outputs = {"output": mock_output}

        result = ElementwiseBinaryOp_hls._get_indexing_expression(
            mock_kernel, "lhs", ("i0", "i1"), "my_pe"
        )

        assert result == "[i0][i1][my_pe]"


class TestGetBroadcastInfoCaching:
    """Tests for _get_broadcast_info() caching behavior."""

    def test_caches_broadcast_info(self):
        """Should cache BroadcastInfo to avoid recomputation."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._needs_streaming_interface = Mock(side_effect=lambda x: True)

        # Mock design_point with both LHS and RHS shapes (new implementation requires both)
        mock_lhs_input = Mock()
        mock_lhs_input.tensor_shape = (1, 64, 64, 128)
        mock_rhs_input = Mock()
        mock_rhs_input.tensor_shape = (128,)

        mock_kernel.design_point = Mock()
        mock_kernel.design_point.inputs = {"lhs": mock_lhs_input, "rhs": mock_rhs_input}

        # First call - should compute
        result1 = ElementwiseBinaryOp_hls._get_broadcast_info(mock_kernel, "rhs")

        # Second call - should return cached value
        result2 = ElementwiseBinaryOp_hls._get_broadcast_info(mock_kernel, "rhs")

        # Should be same object (cached)
        assert result1 is result2
        assert isinstance(result1, BroadcastInfo)

    def test_returns_none_for_static_inputs(self):
        """Should return None for static inputs."""
        mock_kernel = Mock(spec=ElementwiseBinaryOp_hls)
        mock_kernel._needs_streaming_interface = Mock(return_value=False)

        result = ElementwiseBinaryOp_hls._get_broadcast_info(mock_kernel, "rhs")

        assert result is None
