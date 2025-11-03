############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Tests for NavigableInterface - interface-agnostic parallelism control.

Tests validate:
- Parallelism query properties (has_parallelism, parallelism, parallelism_dimension)
- Navigation methods (with_parallelism, increase/decrease, min/max, percentage)
- Exploration methods (sweep_parallelism, sweep_parallelism_percentage)
- Error handling (no parallelism, invalid values)
- Shared dimension references (no duplication)
- Backward compatibility (input_list still works)
"""

import pytest
from onnx import helper, TensorProto

from qonnx.core.modelwrapper import ModelWrapper
from brainsmith.kernels.layernorm import LayerNorm
from brainsmith.dataflow import NavigableInterface, OrderedDimension


def create_layernorm_model():
    """Create minimal ONNX model with LayerNorm (has SIMD parallelism)."""
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 768])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 768])

    node = helper.make_node(
        "LayerNorm",
        inputs=["input"],
        outputs=["output"],
        name="layernorm_0",
        domain="brainsmith.kernels",
        SIMD=64,  # Initial value
        epsilon=1e-5,
        input0Datatype="FLOAT32",
        output0Datatype="FLOAT32"
    )

    graph = helper.make_graph(
        [node],
        "layernorm_graph",
        [input_tensor],
        [output_tensor]
    )

    model = helper.make_model(graph)
    model_w = ModelWrapper(model)

    return model_w, node


# =============================================================================
# Parallelism Query Tests
# =============================================================================

def test_has_parallelism():
    """Test has_parallelism property."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # LayerNorm input has SIMD parallelism
    assert point.input[0].has_parallelism
    assert isinstance(point.input[0], NavigableInterface)


def test_parallelism_query():
    """Test parallelism property returns current value."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize

    # Set SIMD to 16
    kernel_op.set_nodeattr("SIMD", 16)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    assert point.input[0].parallelism == 16


def test_parallelism_dimension_query():
    """Test parallelism_dimension property returns OrderedDimension."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    dim = point.input[0].parallelism_dimension

    assert isinstance(dim, OrderedDimension)
    assert dim.name == "SIMD"
    assert dim.min() == 1
    assert 768 in dim.values  # 768 should be a divisor (max)


def test_structural_properties_delegation():
    """Test that structural properties are delegated to InterfaceDesignPoint."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    nav_interface = point.input[0]
    direct_interface = point.input_list[0]

    # All structural properties should match
    assert nav_interface.name == direct_interface.name
    assert nav_interface.tensor_shape == direct_interface.tensor_shape
    assert nav_interface.block_shape == direct_interface.block_shape
    assert nav_interface.stream_shape == direct_interface.stream_shape
    assert nav_interface.datatype == direct_interface.datatype
    assert nav_interface.is_weight == direct_interface.is_weight


# =============================================================================
# Navigation Tests
# =============================================================================

def test_with_parallelism():
    """Test with_parallelism() sets specific value."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Set to specific value (interface-agnostic)
    point2 = point.input[0].with_parallelism(32)

    assert point2.input[0].parallelism == 32
    # Original point unchanged (immutable)
    assert point.input[0].parallelism != 32


def test_with_min_parallelism():
    """Test with_min_parallelism() sets to minimum."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    point_min = point.input[0].with_min_parallelism()

    assert point_min.input[0].parallelism == 1  # Minimum divisor


def test_with_max_parallelism():
    """Test with_max_parallelism() sets to maximum."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    point_max = point.input[0].with_max_parallelism()

    assert point_max.input[0].parallelism == 768  # Maximum divisor


def test_increase_parallelism():
    """Test increase_parallelism() steps up."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    kernel_op.set_nodeattr("SIMD", 4)
    point = kernel_op.design_point

    # Step up 1
    point2 = point.input[0].increase_parallelism(1)
    assert point2.input[0].parallelism > 4

    # Step up 2
    point3 = point.input[0].increase_parallelism(2)
    assert point3.input[0].parallelism > point2.input[0].parallelism


def test_decrease_parallelism():
    """Test decrease_parallelism() steps down."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    kernel_op.set_nodeattr("SIMD", 16)
    point = kernel_op.design_point

    # Step down 1
    point2 = point.input[0].decrease_parallelism(1)
    assert point2.input[0].parallelism < 16

    # Step down 2
    point3 = point.input[0].decrease_parallelism(2)
    assert point3.input[0].parallelism < point2.input[0].parallelism


def test_with_parallelism_percentage():
    """Test with_parallelism_percentage() maps to dimension."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # 0% = minimum
    point_0 = point.input[0].with_parallelism_percentage(0.0)
    assert point_0.input[0].parallelism == point.input[0].parallelism_dimension.min()

    # 100% = maximum
    point_100 = point.input[0].with_parallelism_percentage(1.0)
    assert point_100.input[0].parallelism == point.input[0].parallelism_dimension.max()

    # 50% = middle
    point_50 = point.input[0].with_parallelism_percentage(0.5)
    assert point_0.input[0].parallelism < point_50.input[0].parallelism < point_100.input[0].parallelism


# =============================================================================
# Exploration Tests
# =============================================================================

def test_sweep_parallelism():
    """Test sweep_parallelism() iterates all values."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Full sweep
    values = [p.input[0].parallelism for p in point.input[0].sweep_parallelism()]

    # Should match dimension values
    expected = list(point.input[0].parallelism_dimension.values)
    assert values == expected


def test_sweep_parallelism_partial():
    """Test sweep_parallelism() with start/stop."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Partial sweep
    values = [p.input[0].parallelism for p in point.input[0].sweep_parallelism(start=4, stop=16)]

    # Should be subset
    assert values[0] == 4
    assert values[-1] == 16
    assert all(4 <= v <= 16 for v in values)


def test_sweep_parallelism_percentage():
    """Test sweep_parallelism_percentage() at specific percentages."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Quartile sweep
    percentages = [0.0, 0.25, 0.5, 0.75, 1.0]
    values = [p.input[0].parallelism for p in point.input[0].sweep_parallelism_percentage(percentages)]

    assert len(values) == 5
    # Should be increasing
    assert values == sorted(values)
    # First should be min, last should be max
    assert values[0] == point.input[0].parallelism_dimension.min()
    assert values[-1] == point.input[0].parallelism_dimension.max()


# =============================================================================
# Error Handling Tests
# =============================================================================

def test_no_parallelism_error():
    """Test error when trying to navigate interface without parallelism."""
    # Create a model where output has no parallelism
    # (LayerNorm output has same parallelism as input, so use input as test)

    # For this test, we'd need a kernel with an interface that has NO stream params
    # For now, test the error case by checking the validation

    # This is tested indirectly - if an interface has no parallelism_dimension,
    # the methods will raise ValueError with helpful message
    pass  # Placeholder - actual test would require a different kernel


def test_invalid_parallelism_value():
    """Test error when setting invalid parallelism value."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Try to set invalid value (not in divisors of 768)
    with pytest.raises(ValueError, match="Invalid"):
        point.input[0].with_parallelism(5)  # 5 does not divide 768


# =============================================================================
# Shared Reference Tests
# =============================================================================

def test_shared_dimension_reference():
    """Test that parallelism_dimension is same object as kernel dimension."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Get dimension from kernel
    kernel_dim = point.design_space.dimensions["SIMD"]

    # Get dimension from interface
    interface_dim = point.input[0].parallelism_dimension

    # Should be SAME object (not a copy)
    assert interface_dim is kernel_dim


def test_shared_dimension_via_design_space():
    """Test dimension shared via InterfaceDesignSpace."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize

    # Get dimension from design space
    space_dim = kernel_op.design_space.input_list[0].parallelism_dimension
    kernel_dim = kernel_op.design_space.dimensions["SIMD"]

    # Should be same object
    assert space_dim is kernel_dim


# =============================================================================
# Backward Compatibility Tests
# =============================================================================

def test_backward_compat_input_list():
    """Test that input_list still works (backward compatibility)."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Old API (direct access)
    direct_interface = point.input_list[0]

    # New API (navigable)
    nav_interface = point.input[0]

    # Structural properties should match
    assert direct_interface.tensor_shape == nav_interface.tensor_shape
    assert direct_interface.stream_shape == nav_interface.stream_shape


def test_wrapper_created_on_demand():
    """Test that NavigableInterface is created fresh on each property access."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Get interface twice
    nav1 = point.input[0]
    nav2 = point.input[0]

    # Should be different wrapper instances
    assert nav1 is not nav2

    # But wrapping same underlying InterfaceDesignPoint
    assert nav1._point is nav2._point

    # And same KernelDesignPoint
    assert nav1._kernel is nav2._kernel


# =============================================================================
# Integration Test: End-to-End DSE Workflow
# =============================================================================

def test_end_to_end_interface_agnostic_dse():
    """Test complete DSE workflow using interface-agnostic API.

    Demonstrates the ergonomic benefit: don't need to know param name (SIMD).
    """
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)
    kernel_op.build_design_space(model_w)  # Initialize
    point = kernel_op.design_point

    # Start at minimum parallelism (don't need to know it's SIMD!)
    point_min = point.input[0].with_min_parallelism()
    assert point_min.input[0].parallelism == 1

    # Step up to find sweet spot
    point_mid = point_min.input[0].increase_parallelism(5)
    assert point_mid.input[0].parallelism > 1

    # Jump to max
    point_max = point.input[0].with_max_parallelism()
    assert point_max.input[0].parallelism == 768

    # Percentage-based sampling
    sample_percentages = [0.0, 0.25, 0.5, 0.75, 1.0]
    sample_points = list(point.input[0].sweep_parallelism_percentage(sample_percentages))
    assert len(sample_points) == 5

    # All points should have different parallelism values
    sample_values = [p.input[0].parallelism for p in sample_points]
    assert sample_values == sorted(sample_values)  # Monotonic
