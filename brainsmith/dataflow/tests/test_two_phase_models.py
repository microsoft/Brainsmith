############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for two-phase model construction (configure() method)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.dse_models import (
    InterfaceDesignSpace,
    KernelDesignSpace,
    KernelDesignPoint,
)
from brainsmith.dataflow.types import ShapeHierarchy
from brainsmith.dataflow.validation import ValidationError


# =============================================================================
# Test KernelDesignSpace.configure()
# =============================================================================

def test_configure_valid_params():
    """Test configure() with valid parallelization parameters."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_inv = InterfaceDesignSpace(
        name="output",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8, 16, 32, 64}},
    )

    # Configure with valid SIMD value
    configured = invariant_model.configure({"SIMD": 64})

    assert isinstance(configured, KernelDesignPoint)
    assert configured.params == {"SIMD": 64}
    assert len(configured.inputs) == 1
    assert len(configured.outputs) == 1
    assert configured.inputs[0].stream_shape == (64,)
    assert configured.outputs[0].stream_shape == (64,)


def test_configure_invalid_param_value():
    """Test configure() with invalid parameter value (not in valid set)."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8}},  # 5 is NOT valid
    )

    # Try to configure with invalid value
    with pytest.raises(ValueError, match="Invalid value for SIMD=5"):
        invariant_model.configure({"SIMD": 5})


def test_configure_unknown_param():
    """Test configure() with unknown parameter name."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2}},
    )

    # Try to configure with unknown param
    with pytest.raises(ValueError, match="Unknown parallelization parameter: 'PE'"):
        invariant_model.configure({"SIMD": 1, "PE": 1})


def test_configure_missing_required_param():
    """Test configure() with missing required parameter."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 64),
        block_shape=(768, 64),
        stream_tiling=["MW", "MH"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"MW": {1, 2}, "MH": {1, 2}},
    )

    # Try to configure with only MW (missing MH)
    with pytest.raises(ValueError, match="Missing required parameter: 'MH'"):
        invariant_model.configure({"MW": 1})


def test_configure_multi_parameter():
    """Test configure() with multiple parallelization parameters."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 64),
        block_shape=(768, 64),
        stream_tiling=["MW", "MH"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"MW": {1, 2, 4, 8}, "MH": {1, 2, 4}},
    )

    # Configure with both params
    configured = invariant_model.configure({"MW": 8, "MH": 4})

    assert configured.params == {"MW": 8, "MH": 4}
    assert configured.inputs[0].stream_shape == (8, 4)


def test_configure_none_stream_tiling():
    """Test configure() with interface that has no stream_tiling."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=None,  # No stream tiling
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={},  # No params
    )

    # Configure with empty params
    configured = invariant_model.configure({})

    # Stream shape should default to block shape
    assert configured.inputs[0].stream_shape == (768,)


def test_configure_literal_stream_tiling():
    """Test configure() with literal values in stream_tiling."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768, 64),
        block_shape=(768, 64),
        stream_tiling=[1, "PE"],  # 1 is literal, PE is param
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"PE": {1, 2, 4, 8}},
    )

    # Configure with PE=4
    configured = invariant_model.configure({"PE": 4})

    # Stream shape should be (1, 4) - 1 from literal, 4 from PE
    assert configured.inputs[0].stream_shape == (1, 4)


def test_configure_preserves_design_space():
    """Test configure() preserves reference to design space."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4}},
    )

    # Configure twice with different params
    configured1 = invariant_model.configure({"SIMD": 1})
    configured2 = invariant_model.configure({"SIMD": 4})

    # Both should reference same design space (flyweight pattern)
    assert configured1.design_space is invariant_model
    assert configured2.design_space is invariant_model
    assert configured1.design_space is configured2.design_space


def test_configure_different_stream_shapes():
    """Test configure() creates different stream shapes for different params."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 64, 128}},
    )

    # Configure with different SIMD values
    configured_64 = invariant_model.configure({"SIMD": 64})
    configured_128 = invariant_model.configure({"SIMD": 128})

    assert configured_64.inputs[0].stream_shape == (64,)
    assert configured_128.inputs[0].stream_shape == (128,)
    assert configured_64.params == {"SIMD": 64}
    assert configured_128.params == {"SIMD": 128}


def test_configure_interface_delegation():
    """Test configured interfaces properly delegate to design_space."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
        is_weight=True,
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 64}},
    )

    configured = invariant_model.configure({"SIMD": 64})

    # Configured interface should delegate design space properties
    cfg_input = configured.inputs[0]
    assert cfg_input.name == "input"
    assert cfg_input.tensor_shape == (768,)
    assert cfg_input.block_shape == (768,)
    assert cfg_input.datatype == DataType["FLOAT32"]
    assert cfg_input.is_weight is True
    # But stream_shape is resolved
    assert cfg_input.stream_shape == (64,)


def test_configure_error_message_shows_valid_values():
    """Test configure() error message includes valid values."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(12,),
        block_shape=(12,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 3, 4, 6, 12}},
    )

    # Try invalid value
    with pytest.raises(ValueError) as exc_info:
        invariant_model.configure({"SIMD": 5})

    # Error message should include valid values
    error_msg = str(exc_info.value)
    assert "Invalid value for SIMD=5" in error_msg
    assert "Valid values:" in error_msg
    assert "[1, 2, 3, 4, 6, 12]" in error_msg


def test_configure_multiple_configurations():
    """Test creating multiple configurations from same design space."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(768,),
        block_shape=(768,),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8, 16, 32, 64}},
    )

    # Create 7 different configurations
    configurations = []
    for simd in [1, 2, 4, 8, 16, 32, 64]:
        cfg = invariant_model.configure({"SIMD": simd})
        configurations.append(cfg)

    # All should have different stream shapes
    assert len(configurations) == 7
    for i, simd in enumerate([1, 2, 4, 8, 16, 32, 64]):
        assert configurations[i].params == {"SIMD": simd}
        assert configurations[i].inputs[0].stream_shape == (simd,)

    # All should reference same design_space
    for cfg in configurations:
        assert cfg.design_space is invariant_model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
