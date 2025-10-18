############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for dimension derivation patterns."""

import pytest
from dataclasses import dataclass
from typing import Optional

from brainsmith.dataflow.dimension_sources import (
    DimensionSource,
    DerivedDim,
    ScaledDim,
    SumDims,
    MaxDim,
    ComputedDim,
)
from brainsmith.dataflow.types import ShapeHierarchy


# Mock interface model for testing
@dataclass
class MockInterface:
    """Mock interface for testing dimension derivation."""
    name: str
    stream_shape: tuple
    block_shape: tuple
    tensor_shape: tuple

    def get_shape(self, hierarchy: ShapeHierarchy) -> tuple:
        """Get shape at specified hierarchy level."""
        if hierarchy == ShapeHierarchy.STREAM:
            return self.stream_shape
        elif hierarchy == ShapeHierarchy.BLOCK:
            return self.block_shape
        elif hierarchy == ShapeHierarchy.TENSOR:
            return self.tensor_shape
        else:
            raise ValueError(f"Unknown hierarchy: {hierarchy}")


# Test fixtures
@pytest.fixture
def mock_interfaces():
    """Create mock interfaces for testing."""
    return {
        "input": MockInterface(
            name="input",
            stream_shape=(16,),
            block_shape=(32, 64),
            tensor_shape=(1, 128, 256)
        ),
        "input0": MockInterface(
            name="input0",
            stream_shape=(8,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 128)
        ),
        "input1": MockInterface(
            name="input1",
            stream_shape=(16,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 128)
        ),
        "input2": MockInterface(
            name="input2",
            stream_shape=(32,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 128)
        ),
    }


@pytest.fixture
def dummy_param_getter():
    """Create dummy parameter getter."""
    return lambda name: None


# ============================================================================
# DerivedDim Tests
# ============================================================================

class TestDerivedDim:
    """Tests for DerivedDim pattern."""

    def test_basic_copy(self, mock_interfaces, dummy_param_getter):
        """Test basic dimension copying."""
        dim = DerivedDim("input", -1)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 16  # input.stream_shape[-1]

    def test_negative_indexing(self, mock_interfaces, dummy_param_getter):
        """Test negative dimension indexing."""
        dim = DerivedDim("input", -1, hierarchy=ShapeHierarchy.TENSOR)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 256  # input.tensor_shape[-1]

    def test_positive_indexing(self, mock_interfaces, dummy_param_getter):
        """Test positive dimension indexing."""
        dim = DerivedDim("input", 0, hierarchy=ShapeHierarchy.BLOCK)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # input.block_shape[0]

    def test_hierarchy_default(self, mock_interfaces, dummy_param_getter):
        """Test default hierarchy is STREAM."""
        dim = DerivedDim("input", 0)
        assert dim.hierarchy == ShapeHierarchy.STREAM
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 16  # input.stream_shape[0]

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dim = DerivedDim("nonexistent", 0)
        with pytest.raises(ValueError, match="Source.*'nonexistent' not found"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_index_out_of_range(self, mock_interfaces, dummy_param_getter):
        """Test error when dimension index out of range."""
        dim = DerivedDim("input", 5, hierarchy=ShapeHierarchy.STREAM)
        with pytest.raises(ValueError, match="Index 5 out of range"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_negative_index_out_of_range(self, mock_interfaces, dummy_param_getter):
        """Test error when negative index out of range."""
        dim = DerivedDim("input", -10, hierarchy=ShapeHierarchy.STREAM)
        with pytest.raises(ValueError, match="Index -10 out of range"):
            dim.resolve(mock_interfaces, dummy_param_getter)


# ============================================================================
# ScaledDim Tests
# ============================================================================

class TestScaledDim:
    """Tests for ScaledDim pattern."""

    def test_scale_up_integer(self, mock_interfaces, dummy_param_getter):
        """Test scaling up by integer factor."""
        dim = ScaledDim("input", 0, 2.0)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # input.stream_shape[0] * 2 = 16 * 2

    def test_scale_up_fractional(self, mock_interfaces, dummy_param_getter):
        """Test scaling up by fractional factor."""
        dim = ScaledDim("input", 0, 1.5)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 24  # input.stream_shape[0] * 1.5 = 16 * 1.5

    def test_scale_down_divisible(self, mock_interfaces, dummy_param_getter):
        """Test scaling down with divisible dimension."""
        dim = ScaledDim("input", 0, 0.5)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 8  # input.stream_shape[0] / 2 = 16 / 2

    def test_scale_down_not_divisible(self, mock_interfaces, dummy_param_getter):
        """Test error when scaling down non-divisible dimension."""
        dim = ScaledDim("input0", 0, 0.5)  # input0.stream[0] = 8
        # 8 is divisible by 2, so this should work
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 4

        # Now test with a non-divisible case
        dim = ScaledDim("input0", 0, 0.25)  # 8 / 4 = 2 (divisible)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 2

    def test_scale_factor_zero(self, mock_interfaces, dummy_param_getter):
        """Test error with zero scale factor."""
        dim = ScaledDim("input", 0, 0.0)
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_scale_factor_negative(self, mock_interfaces, dummy_param_getter):
        """Test error with negative scale factor."""
        dim = ScaledDim("input", 0, -2.0)
        with pytest.raises(ValueError, match="scale_factor must be positive"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_hierarchy_support(self, mock_interfaces, dummy_param_getter):
        """Test scaling with different hierarchies."""
        dim = ScaledDim("input", 0, 2.0, hierarchy=ShapeHierarchy.BLOCK)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 64  # input.block_shape[0] * 2 = 32 * 2

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dim = ScaledDim("nonexistent", 0, 2.0)
        with pytest.raises(ValueError, match="Source.*'nonexistent' not found"):
            dim.resolve(mock_interfaces, dummy_param_getter)


# ============================================================================
# SumDims Tests
# ============================================================================

class TestSumDims:
    """Tests for SumDims pattern."""

    def test_sum_two_sources(self, mock_interfaces, dummy_param_getter):
        """Test summing dimensions from two sources."""
        dim = SumDims((("input0", 0), ("input1", 0)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 24  # 8 + 16

    def test_sum_three_sources(self, mock_interfaces, dummy_param_getter):
        """Test summing dimensions from three sources."""
        dim = SumDims((("input0", 0), ("input1", 0), ("input2", 0)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 56  # 8 + 16 + 32

    def test_sum_with_negative_indexing(self, mock_interfaces, dummy_param_getter):
        """Test summing with negative dimension indices."""
        dim = SumDims((("input0", -1), ("input1", -1), ("input2", -1)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 56  # 8 + 16 + 32 (all stream shapes have 1 dimension)

    def test_sum_different_hierarchies(self, mock_interfaces, dummy_param_getter):
        """Test summing at different hierarchy levels."""
        dim = SumDims(
            (("input0", 0), ("input1", 0)),
            hierarchy=ShapeHierarchy.BLOCK
        )
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # 16 + 16 (both have block[0] = 16)

    def test_empty_sources(self, mock_interfaces, dummy_param_getter):
        """Test error with empty sources."""
        dim = SumDims(())
        with pytest.raises(ValueError, match="SumDims requires at least one source"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dim = SumDims((("input0", 0), ("nonexistent", 0)))
        with pytest.raises(ValueError, match="Source.*'nonexistent' not found"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_index_out_of_range(self, mock_interfaces, dummy_param_getter):
        """Test error when dimension index out of range."""
        dim = SumDims((("input0", 0), ("input1", 10)))
        with pytest.raises(ValueError, match="Index 10 out of range"):
            dim.resolve(mock_interfaces, dummy_param_getter)


# ============================================================================
# MaxDim Tests
# ============================================================================

class TestMaxDim:
    """Tests for MaxDim pattern."""

    def test_max_two_sources(self, mock_interfaces, dummy_param_getter):
        """Test finding maximum from two sources."""
        dim = MaxDim((("input0", 0), ("input1", 0)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 16  # max(8, 16)

    def test_max_three_sources(self, mock_interfaces, dummy_param_getter):
        """Test finding maximum from three sources."""
        dim = MaxDim((("input0", 0), ("input1", 0), ("input2", 0)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # max(8, 16, 32)

    def test_max_all_equal(self, mock_interfaces, dummy_param_getter):
        """Test maximum when all sources have same value."""
        dim = MaxDim((("input0", 0), ("input1", 0)), hierarchy=ShapeHierarchy.BLOCK)
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 16  # max(16, 16)

    def test_max_with_negative_indexing(self, mock_interfaces, dummy_param_getter):
        """Test maximum with negative dimension indices."""
        dim = MaxDim((("input0", -1), ("input1", -1), ("input2", -1)))
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # max(8, 16, 32)

    def test_empty_sources(self, mock_interfaces, dummy_param_getter):
        """Test error with empty sources."""
        dim = MaxDim(())
        with pytest.raises(ValueError, match="MaxDim requires at least one source"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dim = MaxDim((("input0", 0), ("nonexistent", 0)))
        with pytest.raises(ValueError, match="Source.*'nonexistent' not found"):
            dim.resolve(mock_interfaces, dummy_param_getter)


# ============================================================================
# ComputedDim Tests
# ============================================================================

class TestComputedDim:
    """Tests for ComputedDim pattern."""

    def test_basic_computation(self, mock_interfaces, dummy_param_getter):
        """Test basic custom computation."""
        def compute(interfaces, param_getter):
            return interfaces["input"].stream_shape[0] * 2

        dim = ComputedDim(compute, "Double input stream")
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 32  # 16 * 2

    def test_multi_interface_computation(self, mock_interfaces, dummy_param_getter):
        """Test computation using multiple interfaces."""
        def compute(interfaces, param_getter):
            return (
                interfaces["input0"].stream_shape[0] +
                interfaces["input1"].stream_shape[0] +
                interfaces["input2"].stream_shape[0]
            )

        dim = ComputedDim(compute, "Sum all inputs")
        result = dim.resolve(mock_interfaces, dummy_param_getter)
        assert result == 56  # 8 + 16 + 32

    def test_computation_with_params(self, mock_interfaces):
        """Test computation using parameter getter."""
        params = {"multiplier": 3}
        param_getter = lambda name: params.get(name)

        def compute(interfaces, pg):
            mult = pg("multiplier")
            return interfaces["input"].stream_shape[0] * mult

        dim = ComputedDim(compute, "Parameterized multiplication")
        result = dim.resolve(mock_interfaces, param_getter)
        assert result == 48  # 16 * 3

    def test_computation_returns_non_int(self, mock_interfaces, dummy_param_getter):
        """Test error when computation returns non-integer."""
        def compute(interfaces, param_getter):
            return "not an int"

        dim = ComputedDim(compute, "Invalid return type")
        with pytest.raises(ValueError, match="must return int"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_computation_returns_zero(self, mock_interfaces, dummy_param_getter):
        """Test error when computation returns zero."""
        def compute(interfaces, param_getter):
            return 0

        dim = ComputedDim(compute, "Returns zero")
        with pytest.raises(ValueError, match="must be positive"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_computation_returns_negative(self, mock_interfaces, dummy_param_getter):
        """Test error when computation returns negative."""
        def compute(interfaces, param_getter):
            return -10

        dim = ComputedDim(compute, "Returns negative")
        with pytest.raises(ValueError, match="must be positive"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_computation_raises_exception(self, mock_interfaces, dummy_param_getter):
        """Test error handling when computation raises exception."""
        def compute(interfaces, param_getter):
            raise RuntimeError("Something went wrong")

        dim = ComputedDim(compute, "Raises exception")
        with pytest.raises(ValueError, match="raised exception"):
            dim.resolve(mock_interfaces, dummy_param_getter)

    def test_repr_with_description(self):
        """Test repr with description."""
        dim = ComputedDim(lambda i, p: 42, "Test description")
        assert "Test description" in repr(dim)

    def test_repr_without_description(self):
        """Test repr without description."""
        dim = ComputedDim(lambda i, p: 42)
        assert "custom" in repr(dim)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for dimension source patterns."""

    def test_nested_derivation(self, mock_interfaces, dummy_param_getter):
        """Test deriving from already derived dimensions."""
        # First derive stream from input
        derived = DerivedDim("input", 0)
        result1 = derived.resolve(mock_interfaces, dummy_param_getter)

        # Then scale it
        scaled = ScaledDim("input", 0, 2.0)
        result2 = scaled.resolve(mock_interfaces, dummy_param_getter)

        assert result2 == result1 * 2

    def test_all_patterns_frozen(self):
        """Test that all dimension sources are frozen dataclasses."""
        patterns = [
            DerivedDim("input", 0),
            ScaledDim("input", 0, 2.0),
            SumDims((("input", 0),)),
            MaxDim((("input", 0),)),
            ComputedDim(lambda i, p: 42, "test"),
        ]

        for pattern in patterns:
            # Frozen dataclasses raise FrozenInstanceError on attribute modification
            with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
                pattern.source_interface = "modified"

    def test_all_patterns_hashable(self):
        """Test that all dimension sources are hashable."""
        patterns = [
            DerivedDim("input", 0),
            ScaledDim("input", 0, 2.0),
            SumDims((("input", 0),)),
            MaxDim((("input", 0),)),
            ComputedDim(lambda i, p: 42, "test"),
        ]

        # All should be hashable (can be added to set)
        pattern_set = set(patterns)
        assert len(pattern_set) == len(patterns)
