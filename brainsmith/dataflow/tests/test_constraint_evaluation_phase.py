############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Tests for Constraint.evaluation_phase property (Phase 4).

Tests that the evaluation_phase property correctly classifies constraints
as 'structural' or 'parametric' for two-phase kernel construction.
"""

import pytest
from brainsmith.dataflow.constraints import (
    Constraint,
    DatatypeInteger,
    DatatypeFloat,
    DatatypeInRange,
    DatatypesEqual,
    ShapesEqual,
    DimensionDivisible,
    DimensionInRange,
    DimensionEquals,
    IsDynamic,
    IsStatic,
    HasLayout,
    NodeAttributeEquals,
    CustomConstraint,
)
from brainsmith.dataflow.types import ShapeHierarchy


# ====================================================================
# Test 1: Datatype Constraints (No Hierarchy)
# ====================================================================

def test_datatype_integer_is_invariant():
    """Test that DatatypeInteger is classified as structural."""
    constraint = DatatypeInteger(("input0",))
    assert constraint.evaluation_phase == 'structural'


def test_datatype_float_is_invariant():
    """Test that DatatypeFloat is classified as structural."""
    constraint = DatatypeFloat(("input0",))
    assert constraint.evaluation_phase == 'structural'


def test_datatype_in_range_is_invariant():
    """Test that DatatypeInRange is classified as structural."""
    constraint = DatatypeInRange("input", "INT", 4, 8)
    assert constraint.evaluation_phase == 'structural'


def test_datatypes_equal_is_invariant():
    """Test that DatatypesEqual is classified as structural."""
    constraint = DatatypesEqual(("input0", "input1"))
    assert constraint.evaluation_phase == 'structural'


# ====================================================================
# Test 2: Shape Constraints with Different Hierarchies
# ====================================================================

def test_shapes_equal_tensor_is_invariant():
    """Test that ShapesEqual with TENSOR hierarchy is structural."""
    constraint = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.TENSOR)
    assert constraint.evaluation_phase == 'structural'


def test_shapes_equal_block_is_invariant():
    """Test that ShapesEqual with BLOCK hierarchy is structural."""
    constraint = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.BLOCK)
    assert constraint.evaluation_phase == 'structural'


def test_shapes_equal_stream_is_variant():
    """Test that ShapesEqual with STREAM hierarchy is parametric."""
    constraint = ShapesEqual(("input0", "output0"), hierarchy=ShapeHierarchy.STREAM)
    assert constraint.evaluation_phase == 'parametric'


def test_dimension_divisible_tensor_is_invariant():
    """Test that DimensionDivisible with TENSOR hierarchy is structural."""
    constraint = DimensionDivisible("input", 0, 8, hierarchy=ShapeHierarchy.TENSOR)
    assert constraint.evaluation_phase == 'structural'


def test_dimension_divisible_block_is_invariant():
    """Test that DimensionDivisible with BLOCK hierarchy is structural."""
    constraint = DimensionDivisible("input", 0, 8, hierarchy=ShapeHierarchy.BLOCK)
    assert constraint.evaluation_phase == 'structural'


def test_dimension_divisible_stream_is_variant():
    """Test that DimensionDivisible with STREAM hierarchy is parametric."""
    constraint = DimensionDivisible("input", 0, 8, hierarchy=ShapeHierarchy.STREAM)
    assert constraint.evaluation_phase == 'parametric'


def test_dimension_in_range_tensor_is_invariant():
    """Test that DimensionInRange with TENSOR hierarchy is structural."""
    constraint = DimensionInRange("input", 0, 1, 1024, hierarchy=ShapeHierarchy.TENSOR)
    assert constraint.evaluation_phase == 'structural'


def test_dimension_in_range_stream_is_variant():
    """Test that DimensionInRange with STREAM hierarchy is parametric."""
    constraint = DimensionInRange("input", 0, 1, 1024, hierarchy=ShapeHierarchy.STREAM)
    assert constraint.evaluation_phase == 'parametric'


def test_dimension_equals_tensor_is_invariant():
    """Test that DimensionEquals with TENSOR hierarchy is structural."""
    constraint = DimensionEquals("input", 0, 1, hierarchy=ShapeHierarchy.TENSOR)
    assert constraint.evaluation_phase == 'structural'


def test_dimension_equals_stream_is_variant():
    """Test that DimensionEquals with STREAM hierarchy is parametric."""
    constraint = DimensionEquals("input", 0, 1, hierarchy=ShapeHierarchy.STREAM)
    assert constraint.evaluation_phase == 'parametric'


# ====================================================================
# Test 3: ONNX-Specific Constraints
# ====================================================================

def test_is_dynamic_is_invariant():
    """Test that IsDynamic is classified as structural."""
    constraint = IsDynamic(("input0",))
    assert constraint.evaluation_phase == 'structural'


def test_is_static_is_invariant():
    """Test that IsStatic is classified as structural."""
    constraint = IsStatic(("weight",))
    assert constraint.evaluation_phase == 'structural'


def test_has_layout_is_invariant():
    """Test that HasLayout is classified as structural."""
    constraint = HasLayout("input", "NHWC")
    assert constraint.evaluation_phase == 'structural'


def test_node_attribute_equals_is_invariant():
    """Test that NodeAttributeEquals is classified as structural."""
    constraint = NodeAttributeEquals("axis", -1)
    assert constraint.evaluation_phase == 'structural'


# ====================================================================
# Test 4: Custom Constraint
# ====================================================================

def test_custom_constraint_is_invariant():
    """Test that Custom constraint is classified as structural by default."""
    def check_fn(ctx):
        return None

    constraint = CustomConstraint(check_fn, "Test custom constraint")
    assert constraint.evaluation_phase == 'structural'


# ====================================================================
# Test 5: Backward Compatibility
# ====================================================================

def test_all_constraints_have_evaluation_phase():
    """Test that all constraint types have evaluation_phase property."""
    # Test all concrete constraint classes
    constraints_to_test = [
        DatatypeInteger(("input0",)),
        DatatypeFloat(("input0",)),
        DatatypeInRange("input", "INT", 4, 8),
        DatatypesEqual(("input0", "input1")),
        ShapesEqual(("input0", "output0")),
        DimensionDivisible("input", 0, 8),
        DimensionInRange("input", 0, 1, 1024),
        DimensionEquals("input", 0, 1),
        IsDynamic(("input0",)),
        IsStatic(("weight",)),
        HasLayout("input", "NHWC"),
        NodeAttributeEquals("axis", -1),
        CustomConstraint(lambda ctx: None, "Test"),
    ]

    for constraint in constraints_to_test:
        # All constraints should have evaluation_phase property
        assert hasattr(constraint, 'evaluation_phase'), \
            f"{constraint.__class__.__name__} missing evaluation_phase property"

        # All should return either 'invariant' or 'variant'
        phase = constraint.evaluation_phase
        assert phase in ['structural', 'parametric'], \
            f"{constraint.__class__.__name__}.evaluation_phase returned {phase}"


def test_default_hierarchy_constraints_are_invariant():
    """Test that constraints with default hierarchy (STREAM) follow heuristic."""
    # Default hierarchy is STREAM for dimension constraints
    # But the property should still work
    c1 = DimensionDivisible("input", 0, 8)  # Default: STREAM
    assert c1.hierarchy == ShapeHierarchy.STREAM
    assert c1.evaluation_phase == 'parametric'

    c2 = DimensionInRange("input", 0, 1, 1024)  # Default: STREAM
    assert c2.hierarchy == ShapeHierarchy.STREAM
    assert c2.evaluation_phase == 'parametric'

    c3 = DimensionEquals("input", 0, 1)  # Default: STREAM
    assert c3.hierarchy == ShapeHierarchy.STREAM
    assert c3.evaluation_phase == 'parametric'


# ====================================================================
# Test 7: Edge Cases
# ====================================================================

def test_shapes_equal_default_hierarchy():
    """Test ShapesEqual with default hierarchy (TENSOR)."""
    constraint = ShapesEqual(("input0", "output0"))  # Default: TENSOR
    assert constraint.hierarchy == ShapeHierarchy.TENSOR
    assert constraint.evaluation_phase == 'structural'


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
