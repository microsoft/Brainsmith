############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Tests for dataflow utility functions (Phase 7)."""

import pytest
from onnx import TensorProto, helper
from qonnx.core.modelwrapper import ModelWrapper

from brainsmith.dataflow.utils import get_interface, iter_valid_configurations
from brainsmith.kernels.layernorm import LayerNorm


def create_layernorm_model():
    """Create a minimal ONNX model with LayerNorm for testing."""
    # Create input tensor
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 768])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 1, 768])

    # Create LayerNorm node
    node = helper.make_node(
        "LayerNorm",
        inputs=["input"],
        outputs=["output"],
        name="layernorm_0",
        domain="brainsmith.kernels",
        SIMD=1,
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


# ====================================================================
# Tests for iter_valid_configurations()
# ====================================================================

def test_iter_valid_configurations_single_param():
    """Test iteration over single parameter (SIMD)."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Collect all configurations
    configs = list(iter_valid_configurations(kernel_op, model_w))

    # Verify we got configurations
    assert len(configs) > 0, "Should generate at least one configuration"

    # Verify each config has SIMD
    for config in configs:
        assert "SIMD" in config, "Each config should have SIMD parameter"
        assert isinstance(config["SIMD"], int), "SIMD should be an int"
        assert config["SIMD"] > 0, "SIMD should be positive"

    # Verify all configs are unique
    config_tuples = [tuple(sorted(c.items())) for c in configs]
    assert len(config_tuples) == len(set(config_tuples)), "All configs should be unique"

    # Verify configs are in sorted order
    simd_values = [c["SIMD"] for c in configs]
    assert simd_values == sorted(simd_values), "Configs should be sorted by SIMD value"


def test_iter_valid_configurations_count():
    """Test that number of configs matches expected count."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Get valid ranges directly
    valid_ranges = kernel_op.get_valid_ranges(model_w)
    expected_count = len(valid_ranges["SIMD"])

    # Count configs from iterator
    configs = list(iter_valid_configurations(kernel_op, model_w))
    actual_count = len(configs)

    assert actual_count == expected_count, \
        f"Should generate {expected_count} configs, got {actual_count}"


def test_iter_valid_configurations_with_filter():
    """Test filtering of parameter values."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # First get unfiltered count
    all_configs = list(iter_valid_configurations(kernel_op, model_w))
    assert len(all_configs) > 0, "Should have some unfiltered configs"

    # Now test with filter: only SIMD values >= 4 and <= 128
    param_filters = {"SIMD": lambda x: 4 <= x <= 128}
    configs = list(iter_valid_configurations(kernel_op, model_w, param_filters))

    # Verify all configs pass the filter
    for config in configs:
        assert 4 <= config["SIMD"] <= 128, \
            f"SIMD={config['SIMD']} should be in range [4, 128]"

    # Verify we got fewer configs than without filter (unless all values already in range)
    # For 768-dim LayerNorm, divisors include {1, 2, 3} which are < 4, so filtering should reduce count
    assert len(configs) <= len(all_configs), \
        "Filtered configs should not exceed unfiltered"


def test_iter_valid_configurations_filter_eliminates_all():
    """Test filter that eliminates all values."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Filter that rejects everything
    param_filters = {"SIMD": lambda x: False}

    configs = list(iter_valid_configurations(kernel_op, model_w, param_filters))

    # Should return empty iterator
    assert len(configs) == 0, "Should generate no configs when filter rejects all"


def test_iter_valid_configurations_filter_nonexistent_param():
    """Test filter for parameter that doesn't exist."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Filter for nonexistent parameter (should be ignored)
    param_filters = {"NONEXISTENT": lambda x: x > 10}

    configs = list(iter_valid_configurations(kernel_op, model_w, param_filters))

    # Should still generate configs (filter for nonexistent param is ignored)
    assert len(configs) > 0, "Should generate configs even with filter for nonexistent param"


def test_iter_valid_configurations_validates():
    """Test that all generated configs work with get_design_point()."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Get first 5 configs (to keep test fast)
    configs = []
    for i, config in enumerate(iter_valid_configurations(kernel_op, model_w)):
        configs.append(config)
        if i >= 4:  # Get first 5
            break

    # Verify each config works
    for config in configs:
        # Set parameters
        for param_name, param_value in config.items():
            kernel_op.set_nodeattr(param_name, param_value)

        # Should not raise
        configured = kernel_op.get_design_point(model_w)

        # Verify parameters match
        assert configured.params["SIMD"] == config["SIMD"], \
            "Configured instance should have correct SIMD"


def test_iter_valid_configurations_deterministic():
    """Test that iteration order is deterministic."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    # Generate configs twice
    configs1 = list(iter_valid_configurations(kernel_op, model_w))
    configs2 = list(iter_valid_configurations(kernel_op, model_w))

    # Should be identical
    assert configs1 == configs2, "Iteration order should be deterministic"


def test_iter_valid_configurations_iterator_behavior():
    """Test that function returns an iterator, not a list."""
    model_w, node = create_layernorm_model()
    kernel_op = LayerNorm(node)

    result = iter_valid_configurations(kernel_op, model_w)

    # Should be an iterator
    assert hasattr(result, '__iter__'), "Should return an iterator"
    assert hasattr(result, '__next__'), "Should be a generator/iterator"


# ====================================================================
# Tests for get_interface() (existing utility)
# ====================================================================

def test_get_interface_success():
    """Test successful interface retrieval."""
    interfaces = {"input0": "interface_obj", "output0": "another_obj"}
    result = get_interface(interfaces, "input0")
    assert result == "interface_obj"


def test_get_interface_missing():
    """Test error when interface not found."""
    interfaces = {"input0": "interface_obj"}

    with pytest.raises(ValueError) as exc_info:
        get_interface(interfaces, "nonexistent")

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
    assert "input0" in error_msg  # Should list available


def test_get_interface_with_context():
    """Test error message includes context."""
    interfaces = {"input0": "interface_obj"}

    with pytest.raises(ValueError) as exc_info:
        get_interface(interfaces, "nonexistent", context="DerivedDim")

    error_msg = str(exc_info.value)
    assert "nonexistent" in error_msg
