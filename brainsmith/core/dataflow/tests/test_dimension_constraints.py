############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Tests for unified constraint system.

Tests validate constraints against stream_shape (streaming parallelism dimensions)
not tensor_shape (logical tensor dimensions).
"""

import pytest
from brainsmith.core.dataflow import (
    ShapeHierarchy,
    DatatypeConstraint,
    DimensionDivisible,
    DimensionMinValue,
    DimensionMaxValue,
    DimensionEquality,
    DimensionDivisibleBy,
    DimensionScaled,
    InputModel,
    OutputModel,
    KernelModel,
)
from qonnx.core.datatype import DataType


def make_nodeattr_getter(attrs):
    """Create a nodeattr getter function from a dict."""
    def getter(name):
        if name not in attrs:
            raise AttributeError(f"Attribute '{name}' not found")
        return attrs[name]
    return getter


def test_datatype_constraint_pass():
    """Test datatype constraint passes when datatype matches."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 8),
        datatype=DataType["INT8"]
    )

    # Test: INT, 4-16 bits (INT8 matches)
    constraint = DatatypeConstraint("input", "INT", 4, 16)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None


def test_datatype_constraint_fail():
    """Test datatype constraint fails when datatype doesn't match."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 8),
        datatype=DataType["UINT8"]
    )

    # Test: INT constraint (UINT8 doesn't match)
    constraint = DatatypeConstraint("input", "INT", 4, 16)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "INT" in error


def test_divisible_constraint_pass():
    """Test divisible constraint passes when stream dimensions are valid."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),
        stream_shape=(1, 8),  # stream[1] = 8 (elements per cycle)
        datatype=DataType["INT8"]
    )

    # Test: stream[1] % 8 == 0 (8 % 8 == 0) ✓
    constraint = DimensionDivisible("input", 1, 8)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[1] % SIMD == 0 (8 % 8 == 0) ✓
    constraint = DimensionDivisible("input", 1, "SIMD")
    error = constraint.check(input_model, make_nodeattr_getter({"SIMD": 8}))
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
    constraint = DimensionDivisible("input", 1, 8)
    error = constraint.check(input_model, make_nodeattr_getter({}))
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
    constraint = DimensionMinValue("input", 0, 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[1] >= 10 (8 >= 10) ✗
    constraint = DimensionMinValue("input", 1, 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
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
    constraint = DimensionMaxValue("input", 1, 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[0] <= 10 (16 <= 10) ✗
    constraint = DimensionMaxValue("input", 0, 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
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
    min_constraint = DimensionMinValue("input", 0, 10)
    max_constraint = DimensionMaxValue("input", 0, 20)

    min_error = min_constraint.check(input_model, make_nodeattr_getter({}))
    max_error = max_constraint.check(input_model, make_nodeattr_getter({}))

    assert min_error is None
    assert max_error is None

    # Test: 5 <= stream[1] <= 7 (5 <= 8 <= 7) - Should pass min but fail max
    min_constraint = DimensionMinValue("input", 1, 5)
    max_constraint = DimensionMaxValue("input", 1, 7)

    min_error = min_constraint.check(input_model, make_nodeattr_getter({}))
    max_error = max_constraint.check(input_model, make_nodeattr_getter({}))

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
        block_shape=(16, 8),
        stream_shape=(16, 8),  # Same stream shape as input
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: input.stream[0] == output.stream[0] (16 == 16) ✓
    constraint = DimensionEquality("input", 0, "output", 0)
    error = constraint.check(interfaces)
    assert error is None

    # Test: input.stream[1] == output.stream[0] (8 == 16) ✗
    constraint = DimensionEquality("input", 1, "output", 0)
    error = constraint.check(interfaces)
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
    constraint = DimensionDivisibleBy("tensor", 0, "block", 0)
    error = constraint.check(interfaces)
    assert error is None

    # Test: tensor.stream[1] % block.stream[0] == 0 (8 % 4 == 0) ✓
    constraint = DimensionDivisibleBy("tensor", 1, "block", 0)
    error = constraint.check(interfaces)
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
        block_shape=(16, 8),
        stream_shape=(16, 8),  # stream[0] = 16 (2x input), stream[1] = 8 (2x input)
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: output.stream[0] == input.stream[0] * 2 (16 == 8 * 2) ✓
    constraint = DimensionScaled("input", 0, "output", 0, 2)
    error = constraint.check(interfaces)
    assert error is None

    # Test: output.stream[1] == input.stream[1] * 4 (8 == 4 * 4) ✗
    constraint = DimensionScaled("input", 1, "output", 1, 4)
    error = constraint.check(interfaces)
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
    constraint = DimensionMinValue("input", 0, "MIN_SIZE")
    error = constraint.check(input_model, nodeattr_getter)
    assert error is None

    # Test: stream[0] <= MAX_SIZE (16 <= 20) ✓
    constraint = DimensionMaxValue("input", 0, "MAX_SIZE")
    error = constraint.check(input_model, nodeattr_getter)
    assert error is None

    # Test: stream[1] % DIVISOR == 0 (8 % 8 == 0) ✓
    constraint = DimensionDivisible("input", 1, "DIVISOR")
    error = constraint.check(input_model, nodeattr_getter)
    assert error is None


# =============================================================================
# Shape Target Tests (Block and Tensor)
# =============================================================================

def test_block_dimension_divisible():
    """Test divisible constraint on block_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),  # block[1] = 64
        stream_shape=(1, 8),
        datatype=DataType["INT8"]
    )

    # Test: block[1] % 16 == 0 (64 % 16 == 0) ✓
    constraint = DimensionDivisible("input", 1, 16, ShapeHierarchy.BLOCK)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: block[1] % 32 == 0 (64 % 32 == 0) ✓
    constraint = DimensionDivisible("input", 1, 32, ShapeHierarchy.BLOCK)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: block[1] % 100 == 0 (64 % 100 != 0) ✗
    constraint = DimensionDivisible("input", 1, 100, ShapeHierarchy.BLOCK)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "block[1]" in error
    assert "divisible" in error.lower()


def test_tensor_dimension_min_value():
    """Test min value constraint on tensor_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),  # tensor[0] = 128, tensor[1] = 64
        block_shape=(128, 64),
        stream_shape=(1, 8),
        datatype=DataType["INT8"]
    )

    # Test: tensor[0] >= 100 (128 >= 100) ✓
    constraint = DimensionMinValue("input", 0, 100, ShapeHierarchy.TENSOR)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: tensor[1] >= 100 (64 >= 100) ✗
    constraint = DimensionMinValue("input", 1, 100, ShapeHierarchy.TENSOR)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "tensor[1]" in error
    assert "must be >=" in error


def test_block_dimension_max_value():
    """Test max value constraint on block_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),  # block[0] = 128, block[1] = 64
        stream_shape=(1, 8),
        datatype=DataType["INT8"]
    )

    # Test: block[1] <= 100 (64 <= 100) ✓
    constraint = DimensionMaxValue("input", 1, 100, ShapeHierarchy.BLOCK)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: block[0] <= 100 (128 <= 100) ✗
    constraint = DimensionMaxValue("input", 0, 100, ShapeHierarchy.BLOCK)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "block[0]" in error
    assert "must be <=" in error


def test_block_dimension_equality():
    """Test equality constraint on block_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(128, 64),  # block[0] = 128, block[1] = 64
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(128, 64),
        block_shape=(128, 32),  # block[0] = 128, block[1] = 32
        stream_shape=(16, 4),
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: input.block[0] == output.block[0] (128 == 128) ✓
    constraint = DimensionEquality("input", 0, "output", 0, ShapeHierarchy.BLOCK)
    error = constraint.check(interfaces)
    assert error is None

    # Test: input.block[1] == output.block[1] (64 == 32) ✗
    constraint = DimensionEquality("input", 1, "output", 1, ShapeHierarchy.BLOCK)
    error = constraint.check(interfaces)
    assert error is not None
    assert "block[1]" in error
    assert "must equal" in error


def test_tensor_dimension_divisible_by():
    """Test divisibility constraint on tensor_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),  # tensor[0] = 128, tensor[1] = 64
        block_shape=(128, 64),
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(256, 128),  # tensor[0] = 256, tensor[1] = 128
        block_shape=(128, 64),
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: output.tensor[0] % input.tensor[0] == 0 (256 % 128 == 0) ✓
    constraint = DimensionDivisibleBy("output", 0, "input", 0, ShapeHierarchy.TENSOR)
    error = constraint.check(interfaces)
    assert error is None

    # Test: output.tensor[1] % input.tensor[1] == 0 (128 % 64 == 0) ✓
    constraint = DimensionDivisibleBy("output", 1, "input", 1, ShapeHierarchy.TENSOR)
    error = constraint.check(interfaces)
    assert error is None


def test_block_dimension_scaled():
    """Test scaled equality constraint on block_shape."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64),
        block_shape=(64, 32),  # block[0] = 64, block[1] = 32
        stream_shape=(8, 4),
        datatype=DataType["INT8"]
    )

    output_model = OutputModel(
        name="output",
        tensor_shape=(256, 128),
        block_shape=(128, 64),  # block[0] = 128 (2x input), block[1] = 64 (2x input)
        stream_shape=(16, 8),
        datatype=DataType["INT8"]
    )

    interfaces = {"input": input_model, "output": output_model}

    # Test: output.block[0] == input.block[0] * 2 (128 == 64 * 2) ✓
    constraint = DimensionScaled("input", 0, "output", 0, 2, ShapeHierarchy.BLOCK)
    error = constraint.check(interfaces)
    assert error is None

    # Test: output.block[1] == input.block[1] * 2 (64 == 32 * 2) ✓
    constraint = DimensionScaled("input", 1, "output", 1, 2, ShapeHierarchy.BLOCK)
    error = constraint.check(interfaces)
    assert error is None

    # Test: output.block[0] == input.block[0] * 3 (128 == 64 * 3) ✗
    constraint = DimensionScaled("input", 0, "output", 0, 3, ShapeHierarchy.BLOCK)
    error = constraint.check(interfaces)
    assert error is not None
    assert "block[0]" in error


def test_mixed_shape_targets():
    """Test using different shape targets on same model."""
    model = InputModel(
        name="test",
        tensor_shape=(256, 128),  # tensor[0] = 256, tensor[1] = 128
        block_shape=(128, 64),    # block[0] = 128, block[1] = 64
        stream_shape=(16, 8),     # stream[0] = 16, stream[1] = 8
        datatype=DataType["INT8"]
    )

    # All dimensions at each level should be divisible by 8
    tensor_constraint = DimensionDivisible("test", 1, 8, ShapeHierarchy.TENSOR)
    block_constraint = DimensionDivisible("test", 1, 8, ShapeHierarchy.BLOCK)
    stream_constraint = DimensionDivisible("test", 1, 8, ShapeHierarchy.STREAM)

    # tensor[1] = 128 % 8 == 0 ✓
    error = tensor_constraint.check(model, make_nodeattr_getter({}))
    assert error is None

    # block[1] = 64 % 8 == 0 ✓
    error = block_constraint.check(model, make_nodeattr_getter({}))
    assert error is None

    # stream[1] = 8 % 8 == 0 ✓
    error = stream_constraint.check(model, make_nodeattr_getter({}))
    assert error is None


def test_multi_index_divisible_pass():
    """Test divisibility constraint on multiple dimensions (all pass)."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64, 32),
        block_shape=(128, 64, 32),
        stream_shape=(16, 8, 16),  # All divisible by 8
        datatype=DataType["INT8"]
    )

    # Test: stream[0,1,2] all % 8 == 0 ✓
    constraint = DimensionDivisible("input", [0, 1, 2], 8)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None


def test_multi_index_divisible_fail():
    """Test divisibility constraint on multiple dimensions (some fail)."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64, 32),
        block_shape=(128, 64, 32),
        stream_shape=(17, 8, 15),  # 17 and 15 not divisible by 8
        datatype=DataType["INT8"]
    )

    # Test: stream[0,1,2] % 8 - should fail for dims 0 and 2
    constraint = DimensionDivisible("input", [0, 1, 2], 8)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "stream[0]" in error  # First failure
    assert "stream[2]" in error  # Second failure
    assert "17" in error  # Value of dim 0
    assert "15" in error  # Value of dim 2


def test_multi_index_min_value():
    """Test minimum value constraint on multiple dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64, 32),
        block_shape=(128, 64, 32),
        stream_shape=(16, 8, 4),
        datatype=DataType["INT8"]
    )

    # Test: stream[0,1] all >= 8 (16 >= 8 ✓, 8 >= 8 ✓)
    constraint = DimensionMinValue("input", [0, 1], 8)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[1,2] >= 10 (8 >= 10 ✗, 4 >= 10 ✗)
    constraint = DimensionMinValue("input", [1, 2], 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "stream[1]" in error
    assert "stream[2]" in error


def test_multi_index_max_value():
    """Test maximum value constraint on multiple dimensions."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64, 32),
        block_shape=(128, 64, 32),
        stream_shape=(16, 8, 4),
        datatype=DataType["INT8"]
    )

    # Test: stream[0,1,2] all <= 20 ✓
    constraint = DimensionMaxValue("input", [0, 1, 2], 20)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is None

    # Test: stream[0,1] <= 10 (16 <= 10 ✗, 8 <= 10 ✓)
    constraint = DimensionMaxValue("input", [0, 1], 10)
    error = constraint.check(input_model, make_nodeattr_getter({}))
    assert error is not None
    assert "stream[0]" in error
    assert "16" in error


def test_multi_index_with_nodeattr():
    """Test multi-index constraint with nodeattr reference."""
    input_model = InputModel(
        name="input",
        tensor_shape=(128, 64, 32),
        block_shape=(128, 64, 32),
        stream_shape=(16, 8, 16),
        datatype=DataType["INT8"]
    )

    # Test: stream[0,1,2] all % SIMD == 0, SIMD=8
    constraint = DimensionDivisible("input", [0, 1, 2], "SIMD")
    error = constraint.check(input_model, make_nodeattr_getter({"SIMD": 8}))
    assert error is None


def test_multi_index_describe():
    """Test describe() method with multi-index."""
    constraint = DimensionDivisible("input", [0, 1, 2], 8)
    desc = constraint.describe()
    assert "stream[0,1,2]" in desc
    assert "% 8 == 0" in desc

    constraint = DimensionMinValue("output", [1, 3], "MIN", ShapeHierarchy.BLOCK)
    desc = constraint.describe()
    assert "block[1,3]" in desc
    assert ">= MIN" in desc


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
