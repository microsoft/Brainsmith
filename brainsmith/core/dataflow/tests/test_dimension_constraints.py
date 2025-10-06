############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Tests for dimension constraint system.

Tests validate constraints against stream_shape (streaming parallelism dimensions)
not tensor_shape (logical tensor dimensions).
"""

import pytest
from brainsmith.core.dataflow.dimension_constraints import (
    DivisibleConstraint,
    MinValueConstraint,
    MaxValueConstraint,
    EqualityConstraint,
    DivisibleByDimensionConstraint,
    ScaledEqualityConstraint,
)
from brainsmith.core.dataflow.models import InputModel, OutputModel
from qonnx.core.datatype import DataType


def make_nodeattr_getter(attrs):
    """Create a nodeattr getter function from a dict."""
    def getter(name):
        if name not in attrs:
            raise AttributeError(f"Attribute '{name}' not found")
        return attrs[name]
    return getter


def test_divisible_constraint_pass():
    """Test divisible constraint passes when stream dimensions are valid."""
    # Create test input with stream_shape=(1, 8)
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 8),  # stream[1] = 8 (elements per cycle)
        datatype=DataType["INT8"]
    )

    # Test: stream[1] % 8 == 0 (8 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, 8)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[1] % SIMD == 0 (8 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, "SIMD")
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({"SIMD": 8}))
    assert error is None


def test_divisible_constraint_fail():
    """Test divisible constraint fails when stream dimensions are invalid."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 7),  # 7 not divisible by 8
        datatype=DataType["INT8"]
    )

    # Test: stream[1] % 8 != 0 (7 % 8 != 0) ✗
    constraint = DivisibleConstraint("input", 1, 8)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "divisible" in error.lower()


def test_min_value_constraint():
    """Test min value constraint on stream dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    # Test: stream[0] >= 10 (16 >= 10) ✓
    constraint = MinValueConstraint("input", 0, 10)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[1] >= 10 (8 >= 10) ✗
    constraint = MinValueConstraint("input", 1, 10)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "must be >=" in error


def test_max_value_constraint():
    """Test max value constraint on stream dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    # Test: stream[1] <= 10 (8 <= 10) ✓
    constraint = MaxValueConstraint("input", 1, 10)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[0] <= 10 (16 <= 10) ✗
    constraint = MaxValueConstraint("input", 0, 10)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "must be <=" in error


def test_range_constraint_via_composition():
    """Test range constraint via Min + Max composition on stream dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    # Test: 10 <= stream[0] <= 20 (10 <= 16 <= 20) ✓
    min_constraint = MinValueConstraint("input", 0, 10)
    max_constraint = MaxValueConstraint("input", 0, 20)

    min_error = min_constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    max_error = max_constraint.check_interface("input", input_model, make_nodeattr_getter({}))

    assert min_error is None
    assert max_error is None

    # Test: 5 <= stream[1] <= 7 (5 <= 8 <= 7) - Should pass min but fail max
    min_constraint = MinValueConstraint("input", 1, 5)
    max_constraint = MaxValueConstraint("input", 1, 7)

    min_error = min_constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    max_error = max_constraint.check_interface("input", input_model, make_nodeattr_getter({}))

    assert min_error is None  # 8 >= 5
    assert max_error is not None  # 8 <= 7 fails


def test_equality_constraint():
    """Test equality constraint between stream dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(128, 64),
        block_shape=(16, 8),  # Same stream shape as input
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: input.stream[0] == output.stream[0] (16 == 16) ✓
    constraint = EqualityConstraint("input", 0, "output", 0)
    error = constraint.check_relationship(interfaces)
    assert error is None

    # Test: input.stream[1] == output.stream[0] (8 == 16) ✗
    constraint = EqualityConstraint("input", 1, "output", 0)
    error = constraint.check_relationship(interfaces)
    assert error is not None
    assert "must equal" in error


def test_divisible_by_dimension_constraint():
    """Test divisibility constraint between stream dimensions."""
    tensor_model = InputModel(
        name="tensor",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    block_model = InputModel(
        name="block",
        tensor_shape=(16, 8),
        block_shape=(16, 8),
        stream_shape=(4, 2),  # stream[0] = 4, stream[1] = 2
        datatype=DataType["INT8"]
    )

    interfaces = {"tensor": tensor_model, "block": block_model}

    # Test: tensor.stream[0] % block.stream[0] == 0 (16 % 4 == 0) ✓
    constraint = DivisibleByDimensionConstraint("tensor", 0, "block", 0)
    error = constraint.check_relationship(interfaces)
    assert error is None

    # Test: tensor.stream[1] % block.stream[0] == 0 (8 % 4 == 0) ✓
    constraint = DivisibleByDimensionConstraint("tensor", 1, "block", 0)
    error = constraint.check_relationship(interfaces)
    assert error is None


def test_scaled_equality_constraint():
    """Test scaled equality constraint on stream dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(8, 4),  # stream[0] = 8, stream[1] = 4
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(256, 128),
        block_shape=(16, 8),  # stream[0] = 16 (2x input), stream[1] = 8 (2x input)
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: output.stream[0] == input.stream[0] * 2 (16 == 8 * 2) ✓
    constraint = ScaledEqualityConstraint("input", 0, "output", 0, 2)
    error = constraint.check_relationship(interfaces)
    assert error is None

    # Test: output.stream[1] == input.stream[1] * 4 (8 == 4 * 4) ✗
    constraint = ScaledEqualityConstraint("input", 1, "output", 1, 4)
    error = constraint.check_relationship(interfaces)
    assert error is not None


def test_constraint_with_nodeattr_reference():
    """Test constraint referencing nodeattr from nodeattr_getter."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),  # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    nodeattrs = {
        "MIN_SIZE": 10,
        "MAX_SIZE": 20,
        "DIVISOR": 8,
    }
    nodeattr_getter = make_nodeattr_getter(nodeattrs)

    # Test: stream[0] >= MIN_SIZE (16 >= 10) ✓
    constraint = MinValueConstraint("input", 0, "MIN_SIZE")
    error = constraint.check_interface("input", input_model, nodeattr_getter)
    assert error is None

    # Test: stream[0] <= MAX_SIZE (16 <= 20) ✓
    constraint = MaxValueConstraint("input", 0, "MAX_SIZE")
    error = constraint.check_interface("input", input_model, nodeattr_getter)
    assert error is None

    # Test: stream[1] % DIVISOR == 0 (8 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, "DIVISOR")
    error = constraint.check_interface("input", input_model, nodeattr_getter)
    assert error is None


def test_cross_interface_constraints_not_atomic():
    """Test that cross-interface constraints return None for check_interface()."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    # Cross-interface constraints should return None for atomic validation
    constraint = EqualityConstraint("input", 0, "output", 0)
    error = constraint.check_interface("input", input_model, make_nodeattr_getter({}))
    assert error is None  # Cannot validate without both interfaces


def test_atomic_constraints_not_cross_interface():
    """Test that atomic constraints return None for check_relationship()."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model}

    # Atomic constraints should return None for cross-interface validation
    constraint = DivisibleConstraint("input", 1, 8)
    error = constraint.check_relationship(interfaces)
    assert error is None  # Not a cross-interface constraint


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
