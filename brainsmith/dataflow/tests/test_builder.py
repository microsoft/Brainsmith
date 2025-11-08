############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for DesignSpaceBuilder helper methods (two-phase construction)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith._internal.math import divisors
from brainsmith.dataflow.builder import DesignSpaceBuilder
from brainsmith.dataflow.dse_models import InterfaceDesignSpace


# =============================================================================
# Test divisors() Helper
# =============================================================================

def test_divisors_small_numbers():
    """Test divisors() with small known values."""
    # Test 12: divisors are 1, 2, 3, 4, 6, 12
    divisors_12 = divisors(12)
    assert divisors_12 == {1, 2, 3, 4, 6, 12}

    # Test 1: only divisor is 1
    divisors_1 = divisors(1)
    assert divisors_1 == {1}

    # Test 2: divisors are 1, 2
    divisors_2 = divisors(2)
    assert divisors_2 == {1, 2}


def test_divisors_prime_numbers():
    """Test divisors() with prime numbers."""
    # Prime numbers have exactly 2 divisors: 1 and themselves
    divisors_7 = divisors(7)
    assert divisors_7 == {1, 7}

    divisors_13 = divisors(13)
    assert divisors_13 == {1, 13}


def test_divisors_perfect_square():
    """Test divisors() with perfect squares."""
    # 16 = 4^2: divisors are 1, 2, 4, 8, 16
    divisors_16 = divisors(16)
    assert divisors_16 == {1, 2, 4, 8, 16}

    # 64 = 8^2: divisors are 1, 2, 4, 8, 16, 32, 64
    divisors_64 = divisors(64)
    assert divisors_64 == {1, 2, 4, 8, 16, 32, 64}


def test_divisors_768():
    """Test divisors(768) returns correct 18 divisors."""
    divisors_768 = divisors(768)

    # 768 = 2^8 * 3 has 18 divisors
    expected = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}
    assert divisors_768 == expected
    assert len(divisors_768) == 18


def test_divisors_powers_of_two():
    """Test divisors() with powers of 2."""
    # 2^n has n+1 divisors: 1, 2, 4, 8, ..., 2^n
    divisors_8 = divisors(8)
    assert divisors_8 == {1, 2, 4, 8}
    assert len(divisors_8) == 4

    divisors_32 = divisors(32)
    assert divisors_32 == {1, 2, 4, 8, 16, 32}
    assert len(divisors_32) == 6


def test_divisors_zero_raises_error():
    """Test divisors(0) raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        divisors(0)


def test_divisors_negative_raises_error():
    """Test divisors() with negative number raises ValueError."""
    with pytest.raises(ValueError, match="must be positive"):
        divisors(-10)


# =============================================================================
# Test _compute_parameter_ranges() Helper
# =============================================================================

def test_compute_parameter_ranges_single_parameter():
    """Test _compute_parameter_ranges() with single parallelization parameter."""
    builder = DesignSpaceBuilder()

    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=[1, "SIMD"],  # Match dimensions: literal 1, param SIMD
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [])

    assert "SIMD" in valid_ranges
    # Should be divisors of 768 (second dimension)
    assert valid_ranges["SIMD"] == {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}
    assert len(valid_ranges["SIMD"]) == 18


def test_compute_parameter_ranges_multi_parameter():
    """Test _compute_parameter_ranges() with multiple parameters."""
    builder = DesignSpaceBuilder()

    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 64),
        block_shape=(768, 64),
        stream_tiling=["MW", "MH"],
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [])

    assert "MW" in valid_ranges
    assert "MH" in valid_ranges

    # MW should be divisors of 768
    assert len(valid_ranges["MW"]) == 18
    assert 64 in valid_ranges["MW"]

    # MH should be divisors of 64
    assert valid_ranges["MH"] == {1, 2, 4, 8, 16, 32, 64}
    assert len(valid_ranges["MH"]) == 7


def test_compute_parameter_ranges_cross_interface():
    """Test _compute_parameter_ranges() with same param in multiple interfaces."""
    builder = DesignSpaceBuilder()

    # Input and output both use SIMD
    # Input block: 768, Output block: 384
    # Valid SIMD must divide both → gcd(768, 384) = 384
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=[1, "SIMD"],  # Match dimensions
        datatype=DataType["INT8"],
    )

    output_inv = InterfaceDesignSpace(
        name="output",
        tensor_shape=(1, 384),
        block_shape=(1, 384),
        stream_tiling=[1, "SIMD"],  # Match dimensions
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [output_inv])

    assert "SIMD" in valid_ranges
    # Should be divisors of gcd(768, 384) = 384
    expected_divisors = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 384}
    assert valid_ranges["SIMD"] == expected_divisors


def test_compute_parameter_ranges_no_stream_tiling():
    """Test _compute_parameter_ranges() with None stream_tiling."""
    builder = DesignSpaceBuilder()

    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=None,  # No stream tiling
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [])

    # No parallelization parameters
    assert valid_ranges == {}


def test_compute_parameter_ranges_literal_values():
    """Test _compute_parameter_ranges() ignores literal values in stream_tiling."""
    builder = DesignSpaceBuilder()

    # stream_tiling with literal int (not a parameter)
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 64),
        block_shape=(768, 64),
        stream_tiling=[1, "PE"],  # 1 is literal, PE is parameter
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [])

    # Should only have PE (not 1)
    assert "PE" in valid_ranges
    assert 1 not in valid_ranges  # 1 was a literal, not param name
    assert valid_ranges["PE"] == {1, 2, 4, 8, 16, 32, 64}


def test_compute_parameter_ranges_empty_inputs_outputs():
    """Test _compute_parameter_ranges() with no interfaces."""
    builder = DesignSpaceBuilder()

    valid_ranges = builder._compute_parameter_ranges([], [])

    assert valid_ranges == {}


def test_compute_parameter_ranges_gcd_different_dimensions():
    """Test _compute_parameter_ranges() GCD logic with different dimensions."""
    builder = DesignSpaceBuilder()

    # SIMD appears in two different dimensions: 768 and 96
    # Valid SIMD = divisors(gcd(768, 96)) = divisors(96)
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 96),
        block_shape=(768, 96),
        stream_tiling=["SIMD", "SIMD"],  # Same param for both dims
        datatype=DataType["INT8"],
    )

    valid_ranges = builder._compute_parameter_ranges([input_inv], [])

    assert "SIMD" in valid_ranges
    # gcd(768, 96) = 96
    # divisors(96) = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96}
    expected_divisors = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 96}
    assert valid_ranges["SIMD"] == expected_divisors


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
