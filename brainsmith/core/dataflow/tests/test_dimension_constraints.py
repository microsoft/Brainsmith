############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Tests for dimension constraint system.
"""

import pytest
from brainsmith.core.dataflow.dimension_constraints import (
    DivisibleConstraint,
    MinValueConstraint,
    MaxValueConstraint,
    RangeConstraint,
    PowerOfTwoConstraint,
    EqualityConstraint,
    DivisibleByDimensionConstraint,
    ScaledEqualityConstraint,
)
from brainsmith.core.dataflow.models import InputModel, OutputModel
from qonnx.core.datatype import DataType


def test_divisible_constraint_pass():
    """Test divisible constraint passes when valid."""
    # Create test input
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 8),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model, "SIMD": 8}

    # Test: input[1] % 8 == 0 (64 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, 8)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[1] % SIMD == 0 (64 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, "SIMD")
    result = constraint.validate_with_context(context)
    assert result.is_valid


def test_divisible_constraint_fail():
    """Test divisible constraint fails when invalid."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 63),  # 63 not divisible by 8
        block_shape=(128, 63),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model}

    # Test: input[1] % 8 != 0 (63 % 8 != 0) ✗
    constraint = DivisibleConstraint("input", 1, 8)
    result = constraint.validate_with_context(context)
    assert not result.is_valid
    assert len(result.violations) == 1
    assert "divisible" in result.violations[0].message.lower()


def test_min_value_constraint():
    """Test min value constraint."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model}

    # Test: input[0] >= 100 (128 >= 100) ✓
    constraint = MinValueConstraint("input", 0, 100)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[1] >= 100 (64 >= 100) ✗
    constraint = MinValueConstraint("input", 1, 100)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_max_value_constraint():
    """Test max value constraint."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model}

    # Test: input[1] <= 100 (64 <= 100) ✓
    constraint = MaxValueConstraint("input", 1, 100)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[0] <= 100 (128 <= 100) ✗
    constraint = MaxValueConstraint("input", 0, 100)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_range_constraint():
    """Test range constraint."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model}

    # Test: 100 <= input[0] <= 200 (100 <= 128 <= 200) ✓
    constraint = RangeConstraint("input", 0, 100, 200)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: 10 <= input[1] <= 50 (10 <= 64 <= 50) ✗
    constraint = RangeConstraint("input", 1, 10, 50)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_power_of_two_constraint():
    """Test power of 2 constraint."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 65),  # 128 is 2^7, 65 is not power of 2
        block_shape=(128, 65),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model}

    # Test: input[0] == 2^n (128 == 2^7) ✓
    constraint = PowerOfTwoConstraint("input", 0)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[1] == 2^n (65 != 2^n) ✗
    constraint = PowerOfTwoConstraint("input", 1)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_equality_constraint():
    """Test equality constraint between interfaces."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(128, 64),  # Same as input
        block_shape=(128, 64),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model, "output": output_model}

    # Test: input[0] == output[0] (128 == 128) ✓
    constraint = EqualityConstraint("input", 0, "output", 0)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[1] == output[0] (64 == 128) ✗
    constraint = EqualityConstraint("input", 1, "output", 0)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_divisible_by_dimension_constraint():
    """Test divisibility constraint between dimensions."""
    tensor_model = InputModel(
        name="tensor",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    block_model = InputModel(
        name="block",
        tensor_shape=(16, 8),
        block_shape=(16, 8),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {"tensor": tensor_model, "block": block_model}

    # Test: tensor[0] % block[0] == 0 (128 % 16 == 0) ✓
    constraint = DivisibleByDimensionConstraint("tensor", 0, "block", 0)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: tensor[1] % block[0] == 0 (64 % 16 == 0) ✓
    constraint = DivisibleByDimensionConstraint("tensor", 1, "block", 0)
    result = constraint.validate_with_context(context)
    assert result.is_valid


def test_scaled_equality_constraint():
    """Test scaled equality constraint."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(256, 128),  # 2x input
        block_shape=(256, 128),
        datatype=DataType["INT8"]
    )

    context = {"input": input_model, "output": output_model}

    # Test: output[0] == input[0] * 2 (256 == 128 * 2) ✓
    constraint = ScaledEqualityConstraint("output", 0, "input", 0, 2)
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: output[1] == input[1] * 4 (128 == 64 * 4) ✗
    constraint = ScaledEqualityConstraint("output", 1, "input", 1, 4)
    result = constraint.validate_with_context(context)
    assert not result.is_valid


def test_constraint_with_nodeattr_reference():
    """Test constraint referencing nodeattr from context."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 1),
        datatype=DataType["INT8"]
    )

    context = {
        "input": input_model,
        "MIN_SIZE": 100,
        "MAX_SIZE": 200,
        "DIVISOR": 8,
    }

    # Test: input[0] >= MIN_SIZE (128 >= 100) ✓
    constraint = MinValueConstraint("input", 0, "MIN_SIZE")
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[0] <= MAX_SIZE (128 <= 200) ✓
    constraint = MaxValueConstraint("input", 0, "MAX_SIZE")
    result = constraint.validate_with_context(context)
    assert result.is_valid

    # Test: input[1] % DIVISOR == 0 (64 % 8 == 0) ✓
    constraint = DivisibleConstraint("input", 1, "DIVISOR")
    result = constraint.validate_with_context(context)
    assert result.is_valid


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
