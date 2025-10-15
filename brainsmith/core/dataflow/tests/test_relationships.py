############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for cross-interface relationship validation."""

import pytest
from dataclasses import dataclass
from typing import Optional

from qonnx.core.datatype import DataType
from brainsmith.core.dataflow.relationships import (
    InterfaceRelationship,
    DatatypesEqual,
    DimensionsEqual,
    CustomRelationship,
)
from brainsmith.core.dataflow.types import ShapeHierarchy


# Mock interface and kernel model for testing
@dataclass
class MockInterface:
    """Mock interface for testing relationship validation."""
    name: str
    datatype: DataType
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


class MockKernelModel:
    """Mock kernel model for testing relationship validation."""

    def __init__(self, interfaces: dict):
        self._interfaces = interfaces

    def get_interface(self, name: str):
        """Get interface by name."""
        if name not in self._interfaces:
            raise KeyError(f"Interface '{name}' not found")
        return self._interfaces[name]


# Test fixtures
@pytest.fixture
def mock_interfaces():
    """Create mock interfaces for testing."""
    return {
        "input": MockInterface(
            name="input",
            datatype=DataType["INT8"],
            stream_shape=(16,),
            block_shape=(32, 64),
            tensor_shape=(1, 128, 256)
        ),
        "input0": MockInterface(
            name="input0",
            datatype=DataType["INT8"],
            stream_shape=(16,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 128)
        ),
        "input1": MockInterface(
            name="input1",
            datatype=DataType["INT8"],
            stream_shape=(16,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 128)
        ),
        "input2": MockInterface(
            name="input2",
            datatype=DataType["INT4"],
            stream_shape=(32,),
            block_shape=(16, 32),
            tensor_shape=(1, 64, 256)
        ),
        "output": MockInterface(
            name="output",
            datatype=DataType["INT8"],
            stream_shape=(16,),
            block_shape=(32, 64),
            tensor_shape=(1, 128, 256)
        ),
    }


@pytest.fixture
def dummy_param_getter():
    """Create dummy parameter getter."""
    return lambda name: None


# ============================================================================
# DatatypesEqual Tests
# ============================================================================

class TestDatatypesEqual:
    """Tests for DatatypesEqual relationship."""

    def test_two_equal_datatypes(self, mock_interfaces, dummy_param_getter):
        """Test validation with two equal datatypes."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input0", "input1"))
        result = rel.check(model, dummy_param_getter)
        assert result is None  # No error

    def test_three_equal_datatypes(self, mock_interfaces, dummy_param_getter):
        """Test validation with three equal datatypes."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input", "input0", "input1", "output"))
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_two_unequal_datatypes(self, mock_interfaces, dummy_param_getter):
        """Test error with two unequal datatypes."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input0", "input2"))
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "INT8" in result
        assert "INT4" in result
        assert "mismatch" in result.lower()

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input0", "nonexistent"))
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "not found" in result.lower()

    def test_less_than_two_interfaces(self, mock_interfaces, dummy_param_getter):
        """Test error with less than two interfaces."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input0",))
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "at least 2" in result.lower()

    def test_input_output_match(self, mock_interfaces, dummy_param_getter):
        """Test validation between input and output."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(("input", "output"))
        result = rel.check(model, dummy_param_getter)
        assert result is None


# ============================================================================
# DimensionsEqual Tests
# ============================================================================

class TestDimensionsEqual:
    """Tests for DimensionsEqual relationship."""

    # --- Single dimension index tests ---

    def test_single_dim_equal(self, mock_interfaces, dummy_param_getter):
        """Test single dimension equality."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input1"), dim_index=0, hierarchy=ShapeHierarchy.STREAM)
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_single_dim_unequal(self, mock_interfaces, dummy_param_getter):
        """Test single dimension inequality."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input2"), dim_index=0, hierarchy=ShapeHierarchy.STREAM)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "mismatch" in result.lower()

    def test_negative_indexing(self, mock_interfaces, dummy_param_getter):
        """Test negative dimension indexing."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input1"), dim_index=-1, hierarchy=ShapeHierarchy.STREAM)
        result = rel.check(model, dummy_param_getter)
        assert result is None

    # --- Per-interface index tests ---

    def test_per_interface_indices_match(self, mock_interfaces, dummy_param_getter):
        """Test per-interface dimension indices that match."""
        model = MockKernelModel(mock_interfaces)
        # input.tensor[-1] = 256, input2.tensor[-1] = 256
        rel = DimensionsEqual(("input", "input2"), dim_index=(-1, -1), hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_per_interface_indices_mismatch(self, mock_interfaces, dummy_param_getter):
        """Test per-interface dimension indices that don't match."""
        model = MockKernelModel(mock_interfaces)
        # input0.tensor[-1] = 128, input2.tensor[-1] = 256
        rel = DimensionsEqual(("input0", "input2"), dim_index=(-1, -1), hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "mismatch" in result.lower()

    def test_per_interface_wrong_count(self, mock_interfaces, dummy_param_getter):
        """Test error when per-interface indices count doesn't match."""
        # Should raise during initialization
        with pytest.raises(ValueError, match="length must match"):
            DimensionsEqual(("input0", "input1", "input2"), dim_index=(0, 1))

    # --- Slice tests ---

    def test_slice_all_match(self, mock_interfaces, dummy_param_getter):
        """Test slice where all dimensions match."""
        model = MockKernelModel(mock_interfaces)
        # Both have block[0:2] = (16, 32)
        rel = DimensionsEqual(("input0", "input1", "input2"), dim_index=slice(0, 2), hierarchy=ShapeHierarchy.BLOCK)
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_slice_mismatch(self, mock_interfaces, dummy_param_getter):
        """Test slice where dimensions don't match."""
        model = MockKernelModel(mock_interfaces)
        # input0.tensor[0:-1] = (1, 64), input2.tensor[0:-1] = (1, 64) - should match
        rel = DimensionsEqual(("input0", "input2"), dim_index=slice(0, -1), hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        assert result is None

        # input.tensor[0:-1] = (1, 128), input0.tensor[0:-1] = (1, 64) - should not match
        rel2 = DimensionsEqual(("input", "input0"), dim_index=slice(0, -1), hierarchy=ShapeHierarchy.TENSOR)
        result2 = rel2.check(model, dummy_param_getter)
        assert result2 is not None

    # --- None (full shape) tests ---

    def test_full_shape_match(self, mock_interfaces, dummy_param_getter):
        """Test full shape equality."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input1"), dim_index=None, hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_full_shape_mismatch(self, mock_interfaces, dummy_param_getter):
        """Test full shape inequality."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input", "input0"), dim_index=None, hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "mismatch" in result.lower()

    # --- Hierarchy tests ---

    def test_different_hierarchies(self, mock_interfaces, dummy_param_getter):
        """Test dimensions at different hierarchy levels."""
        model = MockKernelModel(mock_interfaces)

        # Stream level
        rel_stream = DimensionsEqual(("input0", "input1"), dim_index=0, hierarchy=ShapeHierarchy.STREAM)
        assert rel_stream.check(model, dummy_param_getter) is None

        # Block level
        rel_block = DimensionsEqual(("input0", "input1", "input2"), dim_index=slice(None), hierarchy=ShapeHierarchy.BLOCK)
        assert rel_block.check(model, dummy_param_getter) is None

        # Tensor level
        rel_tensor = DimensionsEqual(("input0", "input1"), dim_index=None, hierarchy=ShapeHierarchy.TENSOR)
        assert rel_tensor.check(model, dummy_param_getter) is None

    def test_default_hierarchy(self, mock_interfaces, dummy_param_getter):
        """Test default hierarchy is TENSOR."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input1"), dim_index=0)
        assert rel.hierarchy == ShapeHierarchy.TENSOR

    # --- Error cases ---

    def test_interface_not_found(self, mock_interfaces, dummy_param_getter):
        """Test error when interface not found."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "nonexistent"), dim_index=0)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "not found" in result.lower()

    def test_less_than_two_interfaces(self, mock_interfaces, dummy_param_getter):
        """Test error with less than two interfaces."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0",), dim_index=0)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "at least 2" in result.lower()

    def test_index_out_of_range(self, mock_interfaces, dummy_param_getter):
        """Test error when dimension index out of range."""
        model = MockKernelModel(mock_interfaces)
        rel = DimensionsEqual(("input0", "input1"), dim_index=10, hierarchy=ShapeHierarchy.STREAM)
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "out of range" in result.lower()


# ============================================================================
# CustomRelationship Tests
# ============================================================================

class TestCustomRelationship:
    """Tests for CustomRelationship."""

    def test_basic_validation_pass(self, mock_interfaces, dummy_param_getter):
        """Test basic custom validation that passes."""
        def check(model, pg):
            # Always pass
            return None

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check, "Always pass")
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_basic_validation_fail(self, mock_interfaces, dummy_param_getter):
        """Test basic custom validation that fails."""
        def check(model, pg):
            return "Validation failed"

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check, "Always fail")
        result = rel.check(model, dummy_param_getter)
        assert result == "Validation failed"

    def test_multi_interface_check(self, mock_interfaces, dummy_param_getter):
        """Test custom validation using multiple interfaces."""
        def check_matmul_dims(model, pg):
            input_shape = model.get_interface("input").tensor_shape
            output_shape = model.get_interface("output").tensor_shape
            if input_shape != output_shape:
                return f"Shape mismatch: {input_shape} != {output_shape}"
            return None

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check_matmul_dims, "MatMul dims")
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_with_param_getter(self, mock_interfaces):
        """Test custom validation using parameter getter."""
        params = {"check_datatypes": True}
        param_getter = lambda name: params.get(name)

        def check(model, pg):
            if pg("check_datatypes"):
                input_dt = model.get_interface("input0").datatype
                output_dt = model.get_interface("output").datatype
                if input_dt != output_dt:
                    return f"Datatype mismatch: {input_dt} != {output_dt}"
            return None

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check, "Parameterized check")
        result = rel.check(model, param_getter)
        assert result is None

    def test_check_raises_exception(self, mock_interfaces, dummy_param_getter):
        """Test error handling when check function raises exception."""
        def check(model, pg):
            raise RuntimeError("Something went wrong")

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check, "Raises exception")
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "raised exception" in result.lower()

    def test_check_returns_non_string(self, mock_interfaces, dummy_param_getter):
        """Test error when check returns non-string."""
        def check(model, pg):
            return 42  # Invalid return type

        model = MockKernelModel(mock_interfaces)
        rel = CustomRelationship(check, "Invalid return")
        result = rel.check(model, dummy_param_getter)
        assert result is not None
        assert "must return None or str" in result

    def test_repr_with_description(self):
        """Test repr with description."""
        rel = CustomRelationship(lambda m, p: None, "Test description")
        assert "Test description" in repr(rel)

    def test_repr_without_description(self):
        """Test repr without description."""
        rel = CustomRelationship(lambda m, p: None)
        assert "custom" in repr(rel)


# ============================================================================
# Integration Tests
# ============================================================================

class TestIntegration:
    """Integration tests for relationship validation."""

    def test_multiple_relationships(self, mock_interfaces, dummy_param_getter):
        """Test multiple relationships on same model."""
        model = MockKernelModel(mock_interfaces)

        # Check all pass
        rel1 = DatatypesEqual(("input0", "input1"))
        rel2 = DimensionsEqual(("input0", "input1"), dim_index=None, hierarchy=ShapeHierarchy.TENSOR)
        rel3 = CustomRelationship(lambda m, p: None, "Always pass")

        assert rel1.check(model, dummy_param_getter) is None
        assert rel2.check(model, dummy_param_getter) is None
        assert rel3.check(model, dummy_param_getter) is None

    def test_realistic_elementwise_add(self, mock_interfaces, dummy_param_getter):
        """Test realistic ElementwiseAdd validation."""
        model = MockKernelModel(mock_interfaces)

        # ElementwiseAdd: both inputs must have same datatype and shape
        rel_dt = DatatypesEqual(("input0", "input1"))
        rel_shape = DimensionsEqual(("input0", "input1"), dim_index=None, hierarchy=ShapeHierarchy.TENSOR)

        assert rel_dt.check(model, dummy_param_getter) is None
        assert rel_shape.check(model, dummy_param_getter) is None

    def test_realistic_concat(self, mock_interfaces, dummy_param_getter):
        """Test realistic Concat validation."""
        model = MockKernelModel(mock_interfaces)

        # Concat: all inputs must have same spatial dimensions (all but last)
        rel = DimensionsEqual(
            ("input0", "input1"),
            dim_index=slice(0, -1),
            hierarchy=ShapeHierarchy.TENSOR
        )
        result = rel.check(model, dummy_param_getter)
        assert result is None

    def test_all_patterns_frozen(self):
        """Test that all relationships are frozen dataclasses."""
        patterns = [
            DatatypesEqual(("input0", "input1")),
            DimensionsEqual(("input0", "input1"), dim_index=0),
            CustomRelationship(lambda m, p: None, "test"),
        ]

        for pattern in patterns:
            # Frozen dataclasses raise FrozenInstanceError on attribute modification
            with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
                pattern.interface_names = ("modified",)

    def test_all_patterns_hashable(self):
        """Test that all relationships are hashable."""
        patterns = [
            DatatypesEqual(("input0", "input1")),
            DimensionsEqual(("input0", "input1"), dim_index=0),
            CustomRelationship(lambda m, p: None, "test"),
        ]

        # All should be hashable (can be added to set)
        pattern_set = set(patterns)
        assert len(pattern_set) == len(patterns)


# ============================================================================
# Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_empty_tuple_interfaces(self, mock_interfaces, dummy_param_getter):
        """Test with empty interface tuple."""
        model = MockKernelModel(mock_interfaces)
        rel = DatatypesEqual(())
        result = rel.check(model, dummy_param_getter)
        assert result is not None

    def test_slice_empty_result(self, mock_interfaces, dummy_param_getter):
        """Test slice that results in empty tuple."""
        model = MockKernelModel(mock_interfaces)
        # Slice that could result in empty (e.g., slice(10, 20) on 3D tensor)
        rel = DimensionsEqual(("input0", "input1"), dim_index=slice(10, 20), hierarchy=ShapeHierarchy.TENSOR)
        result = rel.check(model, dummy_param_getter)
        # Empty slices should match (both empty)
        assert result is None

    def test_very_long_interface_list(self, dummy_param_getter):
        """Test with many interfaces."""
        # Create many interfaces with same properties
        interfaces = {
            f"input{i}": MockInterface(
                name=f"input{i}",
                datatype=DataType["INT8"],
                stream_shape=(16,),
                block_shape=(16, 32),
                tensor_shape=(1, 64, 128)
            )
            for i in range(10)
        }
        model = MockKernelModel(interfaces)

        interface_names = tuple(f"input{i}" for i in range(10))
        rel = DatatypesEqual(interface_names)
        result = rel.check(model, dummy_param_getter)
        assert result is None
