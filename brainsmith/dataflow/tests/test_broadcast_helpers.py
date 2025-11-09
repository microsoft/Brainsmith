############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
############################################################################
"""Tests for broadcasting helpers."""

import pytest
import numpy as np
from brainsmith.dataflow.broadcast_helpers import BroadcastInfo, compute_broadcast_info


class TestBroadcastInfoCompute:
    """Tests for BroadcastInfo.compute() method."""

    def test_identical_shapes_no_broadcast(self):
        """Identical shapes should not broadcast."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == ()
        assert info.broadcast_dims_rhs == ()
        assert not info.has_broadcast
        assert info.is_compatible_shapes
        assert not info.broadcast_last_axis

    def test_scalar_broadcast_to_4d(self):
        """Scalar (rank-1) broadcasting to 4D tensor."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == ()
        # RHS pads to (1,1,1,128); broadcasts in dims 1,2 where LHS has 64,64
        assert info.broadcast_dims_rhs == (1, 2)
        assert info.has_broadcast
        assert not info.broadcast_last_axis_rhs  # Last dim matches (128)
        assert not info.broadcast_last_axis

    def test_scalar_size_1_broadcast(self):
        """Size-1 scalar broadcasting."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1,))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == ()
        # RHS pads to (1,1,1,1); broadcasts in dims 1,2,3 where LHS has 64,64,128
        assert info.broadcast_dims_rhs == (1, 2, 3)
        assert info.has_broadcast
        assert info.broadcast_last_axis_rhs  # Last dim is 1 → 128
        assert info.broadcast_last_axis

    def test_batch_dimension_broadcast_lhs(self):
        """LHS broadcasts in batch dimension."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (8, 64, 64, 128))

        assert info.output_shape == (8, 64, 64, 128)
        assert info.broadcast_dims_lhs == (0,)  # LHS batch 1 → 8
        assert info.broadcast_dims_rhs == ()
        assert info.has_broadcast
        assert not info.broadcast_last_axis

    def test_spatial_dimension_broadcast_rhs(self):
        """RHS broadcasts in spatial dimensions."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 1, 1, 128))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == ()
        assert info.broadcast_dims_rhs == (1, 2)  # Spatial dims broadcast
        assert info.has_broadcast
        assert not info.broadcast_last_axis

    def test_channel_dimension_broadcast_lhs(self):
        """LHS broadcasts in channel dimension (last axis)."""
        info = BroadcastInfo.compute((1, 64, 64, 1), (1, 64, 64, 128))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == (3,)  # Last dim broadcasts
        assert info.broadcast_dims_rhs == ()
        assert info.has_broadcast
        assert info.broadcast_last_axis_lhs
        assert info.broadcast_last_axis

    def test_bidirectional_broadcast(self):
        """Both inputs broadcast different dimensions."""
        info = BroadcastInfo.compute((1, 1, 64, 128), (8, 64, 1, 128))

        assert info.output_shape == (8, 64, 64, 128)
        assert info.broadcast_dims_lhs == (0, 1)  # LHS broadcasts batch, height
        assert info.broadcast_dims_rhs == (2,)     # RHS broadcasts width
        assert info.has_broadcast
        assert not info.broadcast_last_axis

    def test_incompatible_shapes_error(self):
        """Incompatible shapes should raise ValueError."""
        with pytest.raises(ValueError, match="not broadcastable"):
            BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 32, 128))

    def test_rank_mismatch_2d_to_4d(self):
        """2D broadcasting to 4D."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (64, 128))

        assert info.output_shape == (1, 64, 64, 128)
        assert info.broadcast_dims_lhs == ()
        # RHS pads to (1, 1, 64, 128); only dim 1 broadcasts (1→64)
        assert info.broadcast_dims_rhs == (1,)
        assert info.has_broadcast

    def test_convenience_function(self):
        """Test compute_broadcast_info convenience function."""
        info = compute_broadcast_info((1, 64, 64, 128), (128,))

        assert isinstance(info, BroadcastInfo)
        assert info.output_shape == (1, 64, 64, 128)


class TestBroadcastInfoBufferShape:
    """Tests for BroadcastInfo.get_buffer_shape() method."""

    def test_no_broadcast_buffer_shape(self):
        """Buffer shape with no broadcasting."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))

        lhs_buffer = info.get_buffer_shape("lhs", pe=64)
        rhs_buffer = info.get_buffer_shape("rhs", pe=64)

        assert lhs_buffer == (1, 64, 64, 2)  # 128/64 = 2
        assert rhs_buffer == (1, 64, 64, 2)

    def test_scalar_broadcast_buffer_shape(self):
        """Buffer shape for scalar broadcast."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        rhs_buffer = info.get_buffer_shape("rhs", pe=64)

        # RHS is (128,) but broadcasts to (1, 64, 64, 128)
        # Buffer pads to output rank with broadcast dims as 1: (1, 1, 1, 128/64) = (1, 1, 1, 2)
        assert rhs_buffer == (1, 1, 1, 2)

    def test_spatial_broadcast_buffer_shape(self):
        """Buffer shape for spatial broadcast."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 1, 1, 128))

        rhs_buffer = info.get_buffer_shape("rhs", pe=64)

        # RHS spatial dims are 1 (broadcast), buffer uses 1
        assert rhs_buffer == (1, 1, 1, 2)

    def test_channel_broadcast_buffer_shape(self):
        """Buffer shape for channel broadcast."""
        info = BroadcastInfo.compute((1, 64, 64, 1), (1, 64, 64, 128))

        lhs_buffer = info.get_buffer_shape("lhs", pe=1)

        # LHS channel is 1 (broadcast)
        assert lhs_buffer == (1, 64, 64, 1)

    def test_buffer_shape_invalid_input_name(self):
        """Invalid input_name should raise ValueError."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        with pytest.raises(ValueError, match="must be 'lhs' or 'rhs'"):
            info.get_buffer_shape("invalid", pe=64)


class TestBroadcastInfoNeedsBuffer:
    """Tests for BroadcastInfo.needs_buffer() method."""

    def test_lhs_never_needs_buffer(self):
        """LHS (streaming) never needs buffer."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        assert not info.needs_buffer("lhs")

        info = BroadcastInfo.compute((1, 64, 64, 1), (1, 64, 64, 128))
        assert not info.needs_buffer("lhs")

    def test_rhs_scalar_with_broadcast_needs_buffer(self):
        """RHS scalar with broadcasting needs buffer."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        # Rank-1 (128,) broadcasts to (1, 64, 64, 128)
        # Broadcasting occurs, so buffer is needed to reuse values
        assert info.needs_buffer("rhs")

    def test_rhs_multidimensional_needs_buffer(self):
        """RHS multi-dimensional needs buffer."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))
        assert info.needs_buffer("rhs")

    def test_rhs_broadcast_needs_buffer(self):
        """RHS with broadcasting needs buffer."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 1, 1, 128))
        assert info.needs_buffer("rhs")

    def test_needs_buffer_invalid_input_name(self):
        """Invalid input_name should raise ValueError."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        with pytest.raises(ValueError, match="must be 'lhs' or 'rhs'"):
            info.needs_buffer("invalid")


class TestBroadcastInfoReadCondition:
    """Tests for BroadcastInfo.should_read_new_value() method."""

    def test_no_broadcast_always_read(self):
        """No broadcasting should return None (always read)."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))

        condition = info.should_read_new_value("rhs", ("rep", "spatial", "c", "pe"))
        assert condition is None

    def test_scalar_broadcast_conditional_read(self):
        """Scalar broadcast from rank-1 should read conditionally."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        # RHS (128,) pads to (1, 1, 1, 128) and broadcasts in dims 1, 2
        # (where LHS has 64, 64 and RHS has 1, 1)
        condition = info.should_read_new_value("rhs", ("rep", "spatial", "c", "pe"))

        # RHS broadcasts in dims 1, 2 (spatial, c channels)
        assert condition == "spatial == 0 && c == 0"

    def test_spatial_broadcast_read_condition(self):
        """Spatial broadcast should read when spatial=0."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 1, 1, 128))

        condition = info.should_read_new_value("rhs", ("rep", "spatial_h", "spatial_w", "c"))

        # RHS broadcasts in dims 1, 2 (spatial)
        assert condition == "spatial_h == 0 && spatial_w == 0"

    def test_batch_broadcast_read_condition(self):
        """Batch broadcast should read when rep=0."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (8, 64, 64, 128))

        condition = info.should_read_new_value("lhs", ("rep", "spatial", "c", "pe"))

        # LHS broadcasts in dim 0 (batch)
        assert condition == "rep == 0"

    def test_bidirectional_broadcast_read_condition(self):
        """Bidirectional broadcast should have multiple conditions."""
        info = BroadcastInfo.compute((1, 1, 64, 128), (8, 64, 1, 128))

        condition_lhs = info.should_read_new_value("lhs", ("rep", "h", "w", "c"))
        condition_rhs = info.should_read_new_value("rhs", ("rep", "h", "w", "c"))

        # LHS broadcasts in dims 0, 1
        assert condition_lhs == "rep == 0 && h == 0"

        # RHS broadcasts in dim 2
        assert condition_rhs == "w == 0"

    def test_read_condition_invalid_input_name(self):
        """Invalid input_name should raise ValueError."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        with pytest.raises(ValueError, match="must be 'lhs' or 'rhs'"):
            info.should_read_new_value("invalid", ("rep", "spatial", "c"))


class TestBroadcastInfoIndexExpression:
    """Tests for BroadcastInfo.get_index_expression() method."""

    def test_no_broadcast_index(self):
        """No broadcasting uses direct indexing."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))

        expr = info.get_index_expression("rhs", ("rep", "spatial", "c"), "pe")
        assert expr == "[rep][spatial][c][pe]"

    def test_scalar_broadcast_index(self):
        """Scalar broadcast only indexes by PE."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        expr = info.get_index_expression("rhs", ("rep", "spatial", "c"), "pe")
        # RHS is rank-1, broadcasts in first 3 dims (use 0), last dim uses pe
        assert expr == "[0][0][0][pe]"

    def test_spatial_broadcast_index(self):
        """Spatial broadcast uses 0 for broadcast dims."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 1, 1, 128))

        expr = info.get_index_expression("rhs", ("rep", "h", "w"), "pe")
        # RHS broadcasts in dims 1, 2 (spatial)
        assert expr == "[rep][0][0][pe]"

    def test_batch_broadcast_index(self):
        """Batch broadcast uses 0 for batch dim."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (8, 64, 64, 128))

        expr = info.get_index_expression("lhs", ("rep", "spatial", "c"), "pe")
        # LHS broadcasts in dim 0 (batch)
        assert expr == "[0][spatial][c][pe]"

    def test_channel_broadcast_index(self):
        """Channel broadcast uses 0 for channel dim."""
        info = BroadcastInfo.compute((1, 64, 64, 1), (1, 64, 64, 128))

        expr = info.get_index_expression("lhs", ("rep", "h", "w"), "pe")
        # LHS broadcasts in dim 3 (channel)
        assert expr == "[rep][h][w][0]"

    def test_rank_mismatch_index(self):
        """Rank mismatch uses 0 for implicit and broadcast dims."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (64, 128))

        expr = info.get_index_expression("rhs", ("rep", "h", "w"), "pe")
        # RHS (64, 128) pads to (1, 1, 64, 128)
        # dim 0: implicit → 0, dim 1: broadcasts → 0, dim 2: matches → w, dim 3: last → pe
        assert expr == "[0][0][w][pe]"

    def test_custom_pe_variable(self):
        """Custom PE variable name."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))

        expr = info.get_index_expression("rhs", ("rep", "spatial", "c"), "my_pe")
        assert expr == "[rep][spatial][c][my_pe]"

    def test_index_expression_invalid_input_name(self):
        """Invalid input_name should raise ValueError."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        with pytest.raises(ValueError, match="must be 'lhs' or 'rhs'"):
            info.get_index_expression("invalid", ("rep", "spatial", "c"))


class TestBroadcastInfoProperties:
    """Tests for BroadcastInfo convenience properties."""

    def test_has_broadcast_property(self):
        """Test has_broadcast property."""
        no_broadcast = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))
        assert not no_broadcast.has_broadcast

        with_broadcast = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        assert with_broadcast.has_broadcast

    def test_is_compatible_shapes_property(self):
        """Test is_compatible_shapes property."""
        compatible = BroadcastInfo.compute((1, 64, 64, 128), (1, 64, 64, 128))
        assert compatible.is_compatible_shapes

        incompatible = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        assert not incompatible.is_compatible_shapes

    def test_broadcast_last_axis_property(self):
        """Test broadcast_last_axis property."""
        no_last_axis = BroadcastInfo.compute((1, 64, 64, 128), (128,))
        assert not no_last_axis.broadcast_last_axis

        lhs_last_axis = BroadcastInfo.compute((1, 64, 64, 1), (1, 64, 64, 128))
        assert lhs_last_axis.broadcast_last_axis

        rhs_last_axis = BroadcastInfo.compute((1, 64, 64, 128), (1,))
        assert rhs_last_axis.broadcast_last_axis

    def test_immutability(self):
        """BroadcastInfo should be immutable (frozen dataclass)."""
        info = BroadcastInfo.compute((1, 64, 64, 128), (128,))

        with pytest.raises(AttributeError):
            info.output_shape = (1, 1, 1, 1)  # Should fail, frozen dataclass


class TestBroadcastInfoRealWorldPatterns:
    """Tests for real-world broadcasting patterns from FINN."""

    def test_finn_add_bias_pattern(self):
        """FINN add bias: activation + per-channel bias."""
        info = BroadcastInfo.compute((1, 56, 56, 256), (256,))

        assert info.output_shape == (1, 56, 56, 256)
        assert not info.broadcast_last_axis
        assert info.has_broadcast

    def test_finn_mul_scale_pattern(self):
        """FINN mul scale: activation * per-channel scale."""
        info = BroadcastInfo.compute((1, 28, 28, 512), (512,))

        assert info.output_shape == (1, 28, 28, 512)
        assert not info.broadcast_last_axis

    def test_finn_residual_add_pattern(self):
        """FINN residual: same-shape addition."""
        info = BroadcastInfo.compute((1, 56, 56, 64), (1, 56, 56, 64))

        assert info.is_compatible_shapes
        assert not info.has_broadcast

    def test_finn_batch_scale_pattern(self):
        """FINN batch-wise scale: batch-specific scaling."""
        info = BroadcastInfo.compute((8, 64, 64, 128), (8, 1, 1, 128))

        assert info.output_shape == (8, 64, 64, 128)
        assert info.broadcast_dims_rhs == (1, 2)
        assert not info.broadcast_last_axis
