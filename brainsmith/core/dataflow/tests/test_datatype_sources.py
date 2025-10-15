############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for datatype derivation patterns."""

import pytest
from dataclasses import dataclass

from qonnx.core.datatype import DataType
from brainsmith.core.dataflow.datatype_sources import (
    DatatypeSource,
    DerivedDatatype,
    WidenedDatatype,
    UnionDatatype,
    ComputedDatatype,
)


# Mock interface model for testing
@dataclass
class MockInterface:
    """Mock interface for testing datatype derivation."""
    name: str
    datatype: DataType


# Test fixtures
@pytest.fixture
def mock_interfaces():
    """Create mock interfaces for testing."""
    return {
        "input": MockInterface("input", DataType["INT8"]),
        "input0": MockInterface("input0", DataType["UINT4"]),
        "input1": MockInterface("input1", DataType["INT4"]),
        "input2": MockInterface("input2", DataType["INT8"]),
        "weight": MockInterface("weight", DataType["INT4"]),
    }


@pytest.fixture
def dummy_param_getter():
    """Create dummy parameter getter."""
    return lambda name: None


# ============================================================================
# DerivedDatatype Tests
# ============================================================================

class TestDerivedDatatype:
    """Tests for DerivedDatatype pattern."""

    def test_basic_copy(self, mock_interfaces, dummy_param_getter):
        """Test basic datatype copying."""
        dt = DerivedDatatype("input")
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT8"]

    def test_copy_different_types(self, mock_interfaces, dummy_param_getter):
        """Test copying various datatypes."""
        test_cases = [
            ("input", DataType["INT8"]),
            ("input0", DataType["UINT4"]),
            ("input1", DataType["INT4"]),
            ("weight", DataType["INT4"]),
        ]

        for interface_name, expected_dt in test_cases:
            dt = DerivedDatatype(interface_name)
            result = dt.resolve(mock_interfaces, dummy_param_getter)
            assert result == expected_dt

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dt = DerivedDatatype("nonexistent")
        with pytest.raises(ValueError, match="Source interface 'nonexistent' not found"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_interface_no_datatype(self, dummy_param_getter):
        """Test error when interface has no datatype attribute."""
        interfaces = {"input": object()}  # Object without datatype attribute
        dt = DerivedDatatype("input")
        with pytest.raises(ValueError, match="does not have datatype attribute"):
            dt.resolve(interfaces, dummy_param_getter)


# ============================================================================
# WidenedDatatype Tests
# ============================================================================

class TestWidenedDatatype:
    """Tests for WidenedDatatype pattern."""

    def test_widen_signed(self, mock_interfaces, dummy_param_getter):
        """Test widening signed datatype."""
        dt = WidenedDatatype("input", extra_bits=1)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # INT8 + 1 bit = INT9
        assert result == DataType["INT9"]

    def test_widen_unsigned(self, mock_interfaces, dummy_param_getter):
        """Test widening unsigned datatype."""
        dt = WidenedDatatype("input0", extra_bits=2)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # UINT4 + 2 bits = UINT6
        assert result == DataType["UINT6"]

    def test_widen_multiple_bits(self, mock_interfaces, dummy_param_getter):
        """Test widening by multiple bits."""
        dt = WidenedDatatype("input", extra_bits=4)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # INT8 + 4 bits = INT12
        assert result == DataType["INT12"]

    def test_widen_zero_bits(self, mock_interfaces, dummy_param_getter):
        """Test widening by zero bits (should be identical)."""
        dt = WidenedDatatype("input", extra_bits=0)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # INT8 + 0 bits = INT8
        assert result == DataType["INT8"]

    def test_negative_extra_bits(self, mock_interfaces, dummy_param_getter):
        """Test error with negative extra bits."""
        dt = WidenedDatatype("input", extra_bits=-1)
        with pytest.raises(ValueError, match="extra_bits must be non-negative"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dt = WidenedDatatype("nonexistent", extra_bits=1)
        with pytest.raises(ValueError, match="Source interface 'nonexistent' not found"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_preserves_signedness(self, mock_interfaces, dummy_param_getter):
        """Test that widening preserves signedness."""
        # Signed
        dt_signed = WidenedDatatype("input", extra_bits=2)
        result_signed = dt_signed.resolve(mock_interfaces, dummy_param_getter)
        assert result_signed.signed()

        # Unsigned
        dt_unsigned = WidenedDatatype("input0", extra_bits=2)
        result_unsigned = dt_unsigned.resolve(mock_interfaces, dummy_param_getter)
        assert not result_unsigned.signed()


# ============================================================================
# UnionDatatype Tests
# ============================================================================

class TestUnionDatatype:
    """Tests for UnionDatatype pattern."""

    def test_union_all_unsigned(self, mock_interfaces, dummy_param_getter):
        """Test union of unsigned datatypes."""
        # UINT4: [0, 15]
        dt = UnionDatatype(("input0",))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["UINT4"]

    def test_union_all_signed_same_width(self, mock_interfaces, dummy_param_getter):
        """Test union of signed datatypes with same width."""
        # INT4: [-8, 7] and INT4: [-8, 7]
        dt = UnionDatatype(("input1", "weight"))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # Union of [-8, 7] and [-8, 7] should be INT4
        assert result == DataType["INT4"]

    def test_union_mixed_signed_unsigned(self, mock_interfaces, dummy_param_getter):
        """Test union of mixed signed/unsigned datatypes."""
        # UINT4: [0, 15] and INT4: [-8, 7]
        dt = UnionDatatype(("input0", "input1"))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # Union of [0, 15] and [-8, 7] = [-8, 15]
        # Needs 5 bits signed (covers -16 to 15)
        assert result == DataType["INT5"]

    def test_union_different_widths(self, mock_interfaces, dummy_param_getter):
        """Test union of datatypes with different widths."""
        # INT4: [-8, 7] and INT8: [-128, 127]
        dt = UnionDatatype(("input1", "input2"))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # Union should be INT8 (widest signed type)
        assert result == DataType["INT8"]

    def test_union_three_inputs(self, mock_interfaces, dummy_param_getter):
        """Test union of three inputs."""
        # UINT4: [0, 15], INT4: [-8, 7], INT8: [-128, 127]
        dt = UnionDatatype(("input0", "input1", "input2"))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # Union of [0, 15], [-8, 7], [-128, 127] = [-128, 127]
        assert result == DataType["INT8"]

    def test_empty_sources(self, mock_interfaces, dummy_param_getter):
        """Test error with empty sources."""
        dt = UnionDatatype(())
        with pytest.raises(ValueError, match="requires at least one source interface"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        dt = UnionDatatype(("input0", "nonexistent"))
        with pytest.raises(ValueError, match="Source interface 'nonexistent' not found"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_union_single_value_unsigned(self, dummy_param_getter):
        """Test union with single zero value (edge case)."""
        # Create interface with zero max
        interfaces = {"input": MockInterface("input", DataType["UINT1"])}
        dt = UnionDatatype(("input",))
        result = dt.resolve(interfaces, dummy_param_getter)
        # UINT1 can represent [0, 1], so result should be UINT1
        assert result == DataType["UINT1"]


# ============================================================================
# ComputedDatatype Tests
# ============================================================================

class TestComputedDatatype:
    """Tests for ComputedDatatype pattern."""

    def test_basic_computation(self, mock_interfaces, dummy_param_getter):
        """Test basic custom computation."""
        def compute(interfaces, param_getter):
            # Return same as input
            return interfaces["input"].datatype

        dt = ComputedDatatype(compute, "Copy input datatype")
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT8"]

    def test_multi_interface_computation(self, mock_interfaces, dummy_param_getter):
        """Test computation using multiple interfaces."""
        def compute(interfaces, param_getter):
            # Compute accumulator datatype for multiply-accumulate
            input_dt = interfaces["input"].datatype
            weight_dt = interfaces["weight"].datatype
            # Simple logic: return widest
            if input_dt.bitwidth() > weight_dt.bitwidth():
                return input_dt
            else:
                return weight_dt

        dt = ComputedDatatype(compute, "MAC accumulator")
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT8"]  # Wider of INT8 and INT4

    def test_computation_with_params(self, mock_interfaces):
        """Test computation using parameter getter."""
        params = {"use_wide": True}
        param_getter = lambda name: params.get(name)

        def compute(interfaces, pg):
            if pg("use_wide"):
                return DataType["INT32"]
            else:
                return DataType["INT8"]

        dt = ComputedDatatype(compute, "Parameterized width")
        result = dt.resolve(mock_interfaces, param_getter)
        assert result == DataType["INT32"]

    def test_computation_returns_non_datatype(self, mock_interfaces, dummy_param_getter):
        """Test error when computation returns non-DataType."""
        def compute(interfaces, param_getter):
            return "not a datatype"

        dt = ComputedDatatype(compute, "Invalid return type")
        with pytest.raises(ValueError, match="must return DataType"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_computation_raises_exception(self, mock_interfaces, dummy_param_getter):
        """Test error handling when computation raises exception."""
        def compute(interfaces, param_getter):
            raise RuntimeError("Something went wrong")

        dt = ComputedDatatype(compute, "Raises exception")
        with pytest.raises(ValueError, match="raised exception"):
            dt.resolve(mock_interfaces, dummy_param_getter)

    def test_repr_with_description(self):
        """Test repr with description."""
        dt = ComputedDatatype(lambda i, p: DataType["INT8"], "Test description")
        assert "Test description" in repr(dt)

    def test_repr_without_description(self):
        """Test repr without description."""
        dt = ComputedDatatype(lambda i, p: DataType["INT8"])
        assert "custom" in repr(dt)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for datatype source patterns."""

    def test_all_patterns_frozen(self):
        """Test that all datatype sources are frozen dataclasses."""
        patterns = [
            DerivedDatatype("input"),
            WidenedDatatype("input", 2),
            UnionDatatype(("input",)),
            ComputedDatatype(lambda i, p: DataType["INT8"], "test"),
        ]

        for pattern in patterns:
            # Frozen dataclasses raise FrozenInstanceError on attribute modification
            with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
                pattern.source_interface = "modified"

    def test_all_patterns_hashable(self):
        """Test that all datatype sources are hashable."""
        patterns = [
            DerivedDatatype("input"),
            WidenedDatatype("input", 2),
            UnionDatatype(("input",)),
            ComputedDatatype(lambda i, p: DataType["INT8"], "test"),
        ]

        # All should be hashable (can be added to set)
        pattern_set = set(patterns)
        assert len(pattern_set) == len(patterns)

    def test_realistic_concat_scenario(self, mock_interfaces, dummy_param_getter):
        """Test realistic Concat scenario with UnionDatatype."""
        # Concat with inputs: UINT4, INT4, INT8
        # Expected output: INT8 (union of all ranges)
        dt = UnionDatatype(("input0", "input1", "input2"))
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT8"]

    def test_realistic_elementwise_add_scenario(self, mock_interfaces, dummy_param_getter):
        """Test realistic ElementwiseAdd scenario with WidenedDatatype."""
        # ElementwiseAdd needs 1 extra bit for overflow
        dt = WidenedDatatype("input", extra_bits=1)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT9"]

    def test_realistic_layernorm_scenario(self, mock_interfaces, dummy_param_getter):
        """Test realistic LayerNorm scenario with DerivedDatatype."""
        # LayerNorm output has same datatype as input
        dt = DerivedDatatype("input")
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        assert result == DataType["INT8"]


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_union_with_zero_range(self, dummy_param_getter):
        """Test union computation with minimum range."""
        interfaces = {
            "input0": MockInterface("input0", DataType["UINT1"]),  # [0, 1]
            "input1": MockInterface("input1", DataType["UINT1"]),  # [0, 1]
        }
        dt = UnionDatatype(("input0", "input1"))
        result = dt.resolve(interfaces, dummy_param_getter)
        assert result == DataType["UINT1"]

    def test_union_wide_range(self, dummy_param_getter):
        """Test union with very wide ranges."""
        interfaces = {
            "input0": MockInterface("input0", DataType["INT32"]),
            "input1": MockInterface("input1", DataType["INT32"]),
        }
        dt = UnionDatatype(("input0", "input1"))
        result = dt.resolve(interfaces, dummy_param_getter)
        assert result == DataType["INT32"]

    def test_widen_to_large_bitwidth(self, mock_interfaces, dummy_param_getter):
        """Test widening to large bitwidth."""
        dt = WidenedDatatype("input", extra_bits=24)
        result = dt.resolve(mock_interfaces, dummy_param_getter)
        # INT8 + 24 = INT32
        assert result == DataType["INT32"]
