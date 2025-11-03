############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Tests for OrderedDimension class."""

import pytest

from brainsmith.dataflow.ordered_dimension import OrderedDimension


# ============================================================================
# Construction and Validation Tests
# ============================================================================

def test_ordered_dimension_basic_construction():
    """Test basic OrderedDimension construction."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))

    assert dim.name == "SIMD"
    assert dim.values == (1, 2, 4, 8, 16)
    assert dim.default is None


def test_ordered_dimension_with_default():
    """Test OrderedDimension with explicit default."""
    dim = OrderedDimension("depth", (128, 256, 512, 1024), default=256)

    assert dim.name == "depth"
    assert dim.default == 256
    assert dim.get_default() == 256


def test_ordered_dimension_default_is_min():
    """Test that get_default() returns min when no default specified."""
    dim = OrderedDimension("PE", (1, 2, 4, 8, 16, 32))

    assert dim.default is None
    assert dim.get_default() == 1  # Should return minimum


def test_ordered_dimension_validates_empty():
    """Test that empty values raises ValueError."""
    with pytest.raises(ValueError, match="empty values"):
        OrderedDimension("test", ())


def test_ordered_dimension_validates_unsorted():
    """Test that unsorted values raises ValueError."""
    with pytest.raises(ValueError, match="must be sorted ascending"):
        OrderedDimension("test", (8, 4, 2, 1))  # Descending


def test_ordered_dimension_validates_duplicates():
    """Test that duplicate values raises ValueError."""
    with pytest.raises(ValueError, match="duplicate values"):
        OrderedDimension("test", (1, 2, 4, 4, 8))


def test_ordered_dimension_validates_invalid_default():
    """Test that default not in values raises ValueError."""
    with pytest.raises(ValueError, match="Default value .* not in dimension"):
        OrderedDimension("test", (1, 2, 4, 8), default=3)


def test_ordered_dimension_converts_list_to_tuple():
    """Test that list values are converted to tuple."""
    dim = OrderedDimension("test", [1, 2, 4, 8])  # Pass list

    assert isinstance(dim.values, tuple)
    assert dim.values == (1, 2, 4, 8)


# ============================================================================
# Positional Access Tests
# ============================================================================

def test_min_max():
    """Test min() and max() methods."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16, 32, 64))

    assert dim.min() == 1
    assert dim.max() == 64


def test_at_index_positive():
    """Test at_index() with positive indices."""
    dim = OrderedDimension("PE", (1, 2, 4, 8, 16))

    assert dim.at_index(0) == 1
    assert dim.at_index(2) == 4
    assert dim.at_index(4) == 16


def test_at_index_negative():
    """Test at_index() with negative indices."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))

    assert dim.at_index(-1) == 16
    assert dim.at_index(-2) == 8
    assert dim.at_index(-5) == 1


def test_at_index_out_of_range():
    """Test that at_index() raises IndexError for out of range."""
    dim = OrderedDimension("test", (1, 2, 4))

    with pytest.raises(IndexError, match="out of range"):
        dim.at_index(3)

    with pytest.raises(IndexError, match="out of range"):
        dim.at_index(-4)


def test_index_of():
    """Test index_of() method."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))

    assert dim.index_of(1) == 0
    assert dim.index_of(4) == 2
    assert dim.index_of(16) == 4


def test_index_of_not_found():
    """Test that index_of() raises ValueError for value not in dimension."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="Value .* not in dimension"):
        dim.index_of(3)


# ============================================================================
# Navigation Tests
# ============================================================================

def test_step_up():
    """Test step_up() navigation."""
    dim = OrderedDimension("PE", (1, 2, 4, 8, 16, 32, 64))

    assert dim.step_up(1, 1) == 2
    assert dim.step_up(4, 1) == 8
    assert dim.step_up(4, 2) == 16
    assert dim.step_up(8, 3) == 64


def test_step_up_clamping():
    """Test that step_up() clamps at maximum."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))

    # Step beyond max
    assert dim.step_up(8, 10) == 16  # Clamped at max
    assert dim.step_up(16, 5) == 16  # Already at max


def test_step_up_zero_steps():
    """Test step_up() with n=0."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    assert dim.step_up(4, 0) == 4  # No change


def test_step_up_negative_n():
    """Test that step_up() raises ValueError for negative n."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="requires n >= 0"):
        dim.step_up(4, -1)


def test_step_up_invalid_current():
    """Test that step_up() raises ValueError for invalid current value."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="not in dimension"):
        dim.step_up(3, 1)  # 3 is not in dimension


def test_step_down():
    """Test step_down() navigation."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16, 32, 64))

    assert dim.step_down(64, 1) == 32
    assert dim.step_down(16, 1) == 8
    assert dim.step_down(16, 2) == 4
    assert dim.step_down(32, 4) == 2


def test_step_down_clamping():
    """Test that step_down() clamps at minimum."""
    dim = OrderedDimension("PE", (1, 2, 4, 8, 16))

    # Step beyond min
    assert dim.step_down(4, 10) == 1  # Clamped at min
    assert dim.step_down(1, 5) == 1   # Already at min


def test_step_down_zero_steps():
    """Test step_down() with n=0."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    assert dim.step_down(4, 0) == 4  # No change


def test_step_down_negative_n():
    """Test that step_down() raises ValueError for negative n."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="requires n >= 0"):
        dim.step_down(4, -1)


# ============================================================================
# Percentage-Based Access Tests
# ============================================================================

def test_at_percentage_extremes():
    """Test at_percentage() at 0.0 and 1.0."""
    dim = OrderedDimension("PE", (1, 2, 4, 8, 16))

    assert dim.at_percentage(0.0) == 1   # Min
    assert dim.at_percentage(1.0) == 16  # Max


def test_at_percentage_middle():
    """Test at_percentage() at 0.5 (middle)."""
    dim = OrderedDimension("test", (1, 2, 4, 8, 16))  # 5 values, indices 0-4

    # 0.5 * 4 = 2.0 → index 2 → value 4
    assert dim.at_percentage(0.5, rounding='natural') == 4


def test_at_percentage_quartiles():
    """Test at_percentage() at quartile points."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16))  # 5 values

    assert dim.at_percentage(0.0) == 1    # 0.0 * 4 = 0
    assert dim.at_percentage(0.25) == 2   # 0.25 * 4 = 1.0 → round(1.0) = 1
    assert dim.at_percentage(0.5) == 4    # 0.5 * 4 = 2.0 → round(2.0) = 2
    assert dim.at_percentage(0.75) == 8   # 0.75 * 4 = 3.0 → round(3.0) = 3
    assert dim.at_percentage(1.0) == 16   # 1.0 * 4 = 4


def test_at_percentage_rounding_natural():
    """Test at_percentage() with natural (round) rounding."""
    dim = OrderedDimension("test", (10, 20, 30, 40, 50))  # 5 values, indices 0-4

    # Test values that round to different indices
    assert dim.at_percentage(0.1, rounding='natural') == 10  # 0.1*4=0.4 → round=0
    assert dim.at_percentage(0.3, rounding='natural') == 20  # 0.3*4=1.2 → round=1
    assert dim.at_percentage(0.4, rounding='natural') == 30  # 0.4*4=1.6 → round=2
    assert dim.at_percentage(0.6, rounding='natural') == 30  # 0.6*4=2.4 → round=2
    assert dim.at_percentage(0.7, rounding='natural') == 40  # 0.7*4=2.8 → round=3


def test_at_percentage_rounding_down():
    """Test at_percentage() with floor (down) rounding."""
    dim = OrderedDimension("test", (10, 20, 30, 40, 50))  # 5 values

    assert dim.at_percentage(0.1, rounding='down') == 10  # 0.1*4=0.4 → floor=0
    assert dim.at_percentage(0.3, rounding='down') == 20  # 0.3*4=1.2 → floor=1
    assert dim.at_percentage(0.7, rounding='down') == 30  # 0.7*4=2.8 → floor=2
    assert dim.at_percentage(0.9, rounding='down') == 40  # 0.9*4=3.6 → floor=3


def test_at_percentage_rounding_up():
    """Test at_percentage() with ceil (up) rounding."""
    dim = OrderedDimension("test", (10, 20, 30, 40, 50))  # 5 values

    assert dim.at_percentage(0.1, rounding='up') == 20  # 0.1*4=0.4 → ceil=1
    assert dim.at_percentage(0.3, rounding='up') == 30  # 0.3*4=1.2 → ceil=2
    assert dim.at_percentage(0.7, rounding='up') == 40  # 0.7*4=2.8 → ceil=3
    assert dim.at_percentage(0.9, rounding='up') == 50  # 0.9*4=3.6 → ceil=4


def test_at_percentage_exact_values():
    """Test at_percentage() with exact fractional indices."""
    dim = OrderedDimension("test", (10, 20, 30, 40))  # 4 values, indices 0-3

    # Exact index values (should give same result for all rounding modes)
    assert dim.at_percentage(0.0) == 10   # 0.0*3=0
    assert dim.at_percentage(1/3) == 20   # 0.333*3=1.0
    assert dim.at_percentage(2/3) == 30   # 0.666*3=2.0
    assert dim.at_percentage(1.0) == 40   # 1.0*3=3


def test_at_percentage_invalid_percentage():
    """Test that at_percentage() raises ValueError for out of range percentage."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        dim.at_percentage(-0.1)

    with pytest.raises(ValueError, match="must be in \\[0.0, 1.0\\]"):
        dim.at_percentage(1.1)


def test_at_percentage_invalid_rounding():
    """Test that at_percentage() raises ValueError for invalid rounding mode."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    with pytest.raises(ValueError, match="Invalid rounding mode"):
        dim.at_percentage(0.5, rounding='invalid')


def test_at_percentage_single_value():
    """Test at_percentage() with single-value dimension."""
    dim = OrderedDimension("test", (42,))

    # All percentages should return the only value
    assert dim.at_percentage(0.0) == 42
    assert dim.at_percentage(0.5) == 42
    assert dim.at_percentage(1.0) == 42


# ============================================================================
# Iteration Tests
# ============================================================================

def test_len():
    """Test __len__() method."""
    dim = OrderedDimension("test", (1, 2, 4, 8, 16))

    assert len(dim) == 5


def test_iteration():
    """Test __iter__() method."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    values = list(dim)
    assert values == [1, 2, 4, 8]


def test_contains():
    """Test __contains__() method (in operator)."""
    dim = OrderedDimension("test", (1, 2, 4, 8, 16))

    assert 1 in dim
    assert 4 in dim
    assert 16 in dim
    assert 3 not in dim
    assert 32 not in dim


def test_validate():
    """Test validate() method."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    assert dim.validate(1) is True
    assert dim.validate(4) is True
    assert dim.validate(8) is True
    assert dim.validate(3) is False
    assert dim.validate(16) is False


# ============================================================================
# Display Tests
# ============================================================================

def test_repr_short():
    """Test __repr__() for short value lists."""
    dim = OrderedDimension("PE", (1, 2, 4, 8))

    # Short lists show all values
    assert repr(dim) == "OrderedDimension('PE', (1, 2, 4, 8))"


def test_repr_long():
    """Test __repr__() for long value lists."""
    dim = OrderedDimension("SIMD", (1, 2, 4, 8, 16, 32, 64, 128))

    # Long lists show first, second, and last
    assert "OrderedDimension('SIMD', (1, 2, ..., 128))" == repr(dim)


def test_repr_with_default():
    """Test __repr__() with explicit default."""
    dim = OrderedDimension("depth", (128, 256, 512), default=256)

    assert "default=256" in repr(dim)


# ============================================================================
# Edge Cases
# ============================================================================

def test_single_value_dimension():
    """Test dimension with single value."""
    dim = OrderedDimension("test", (42,))

    assert dim.min() == 42
    assert dim.max() == 42
    assert dim.at_index(0) == 42
    assert dim.index_of(42) == 0
    assert dim.step_up(42, 10) == 42  # Clamped
    assert dim.step_down(42, 10) == 42  # Clamped
    assert len(dim) == 1


def test_two_value_dimension():
    """Test dimension with two values."""
    dim = OrderedDimension("test", (1, 10))

    assert dim.min() == 1
    assert dim.max() == 10
    assert dim.step_up(1, 1) == 10
    assert dim.step_down(10, 1) == 1
    assert dim.at_percentage(0.0) == 1
    assert dim.at_percentage(0.5, rounding='down') == 1   # 0.5*1=0.5 → floor=0
    assert dim.at_percentage(0.5, rounding='up') == 10    # 0.5*1=0.5 → ceil=1
    assert dim.at_percentage(1.0) == 10


def test_large_dimension():
    """Test dimension with many values (divisors of 768)."""
    divisors_768 = (1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768)
    dim = OrderedDimension("SIMD", divisors_768)

    assert len(dim) == 18
    assert dim.min() == 1
    assert dim.max() == 768
    # 0.5 * 17 = 8.5 → round(8.5) = 8 → values[8] = 24
    assert dim.at_percentage(0.5) == 24  # Middle value (index 8 of 0-17)
    assert 64 in dim
    assert dim.step_up(32, 2) == 64


def test_immutability():
    """Test that OrderedDimension is immutable (frozen dataclass)."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    # Attempt to modify should raise FrozenInstanceError
    with pytest.raises(Exception):  # FrozenInstanceError or AttributeError
        dim.name = "modified"

    with pytest.raises(Exception):
        dim.values = (1, 2, 3)


def test_values_tuple_immutable():
    """Test that values tuple is immutable."""
    dim = OrderedDimension("test", (1, 2, 4, 8))

    # Attempt to modify tuple should raise TypeError
    with pytest.raises(TypeError):
        dim.values[0] = 10
