############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for configured model types (two-phase construction)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.models import (
    InvariantInterfaceModel,
    InvariantKernelModel,
    ConfiguredInterfaceModel,
    ConfiguredKernelModel,
)
from brainsmith.dataflow.types import ShapeHierarchy


# =============================================================================
# Test ConfiguredInterfaceModel
# =============================================================================

def test_configured_interface_model_creation():
    """Test ConfiguredInterfaceModel can be created."""
    invariant = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(64,),
    )

    assert configured.stream_shape == (64,)
    assert configured.invariant is invariant


def test_configured_interface_model_property_delegation():
    """Test ConfiguredInterfaceModel properties delegate to invariant."""
    invariant = InvariantInterfaceModel(
        name="test_interface",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
        is_weight=True,
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(32,),
    )

    # Properties should delegate to invariant
    assert configured.name == "test_interface"
    assert configured.tensor_shape == (1, 224, 224, 768)
    assert configured.block_shape == (1, 1, 768)
    assert configured.datatype == DataType["FLOAT32"]
    assert configured.is_weight is True


def test_configured_interface_model_get_shape():
    """Test ConfiguredInterfaceModel.get_shape() for all hierarchies."""
    invariant = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(64,),
    )

    # Test all hierarchy levels
    tensor_shape = configured.get_shape(ShapeHierarchy.TENSOR)
    assert tensor_shape == (1, 224, 224, 768)

    block_shape = configured.get_shape(ShapeHierarchy.BLOCK)
    assert block_shape == (1, 1, 768)

    stream_shape = configured.get_shape(ShapeHierarchy.STREAM)
    assert stream_shape == (64,)


def test_configured_interface_model_get_shape_invalid_hierarchy():
    """Test ConfiguredInterfaceModel.get_shape() with invalid hierarchy raises error."""
    invariant = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(64,),
    )

    # Invalid hierarchy should raise ValueError
    with pytest.raises(ValueError, match="Invalid hierarchy"):
        configured.get_shape("invalid")


def test_configured_interface_model_folding_factors():
    """Test ConfiguredInterfaceModel computed properties."""
    invariant = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(1, 64),  # Match dimensions with block_shape
    )

    # Test computed properties
    assert configured.tensor_folding_factor == 1  # tensor / block = (1,768) / (1,768)
    assert configured.block_folding_factor == 12  # block / stream = (1,768) / (1,64)
    assert configured.streaming_bandwidth == 64  # prod(stream_shape) = 1 * 64
    assert configured.stream_width_bits == 64 * 8  # bandwidth * bitwidth


def test_configured_interface_model_immutable():
    """Test ConfiguredInterfaceModel is immutable."""
    invariant = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = ConfiguredInterfaceModel(
        invariant=invariant,
        stream_shape=(64,),
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured.stream_shape = (128,)


# =============================================================================
# Test ConfiguredKernelModel
# =============================================================================

def test_configured_kernel_model_creation():
    """Test ConfiguredKernelModel can be created."""
    input_inv = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_inv = InvariantInterfaceModel(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8, 16}},
    )

    input_cfg = ConfiguredInterfaceModel(invariant=input_inv, stream_shape=(64,))
    output_cfg = ConfiguredInterfaceModel(invariant=output_inv, stream_shape=(64,))

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(input_cfg,),
        outputs=(output_cfg,),
        params={"SIMD": 64},
    )

    assert configured_model.name == "TestKernel"
    assert configured_model.params == {"SIMD": 64}
    assert len(configured_model.inputs) == 1
    assert len(configured_model.outputs) == 1


def test_configured_kernel_model_property_delegation():
    """Test ConfiguredKernelModel properties delegate to invariant."""
    invariant_model = InvariantKernelModel(
        name="LayerNorm",
        inputs=(),
        outputs=(),
        internal_datatypes={"accumulator": DataType["INT32"]},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={},
    )

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(),
        outputs=(),
        params={},
    )

    # Properties should delegate to invariant
    assert configured_model.name == "LayerNorm"
    assert configured_model.internal_datatypes == {"accumulator": DataType["INT32"]}


def test_configured_kernel_model_get_input():
    """Test ConfiguredKernelModel.get_input() lookup."""
    input_inv = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2}},
    )

    input_cfg = ConfiguredInterfaceModel(invariant=input_inv, stream_shape=(64,))

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(input_cfg,),
        outputs=(),
        params={"SIMD": 64},
    )

    found = configured_model.get_input("input")
    assert found is input_cfg

    not_found = configured_model.get_input("nonexistent")
    assert not_found is None


def test_configured_kernel_model_get_output():
    """Test ConfiguredKernelModel.get_output() lookup."""
    output_inv = InvariantInterfaceModel(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = InvariantKernelModel(
        name="TestKernel",
        inputs=(),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2}},
    )

    output_cfg = ConfiguredInterfaceModel(invariant=output_inv, stream_shape=(64,))

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(),
        outputs=(output_cfg,),
        params={"SIMD": 64},
    )

    found = configured_model.get_output("output")
    assert found is output_cfg

    not_found = configured_model.get_output("nonexistent")
    assert not_found is None


def test_configured_kernel_model_get_interface():
    """Test ConfiguredKernelModel.get_interface() searches both inputs and outputs."""
    input_inv = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_inv = InvariantInterfaceModel(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    invariant_model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1}},
    )

    input_cfg = ConfiguredInterfaceModel(invariant=input_inv, stream_shape=(64,))
    output_cfg = ConfiguredInterfaceModel(invariant=output_inv, stream_shape=(64,))

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(input_cfg,),
        outputs=(output_cfg,),
        params={"SIMD": 64},
    )

    # Find input
    found_input = configured_model.get_interface("input")
    assert found_input is input_cfg

    # Find output
    found_output = configured_model.get_interface("output")
    assert found_output is output_cfg

    # Not found
    not_found = configured_model.get_interface("nonexistent")
    assert not_found is None


def test_configured_kernel_model_immutable():
    """Test ConfiguredKernelModel is immutable."""
    invariant_model = InvariantKernelModel(
        name="TestKernel",
        inputs=(),
        outputs=(),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={},
    )

    configured_model = ConfiguredKernelModel(
        invariant=invariant_model,
        inputs=(),
        outputs=(),
        params={},
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured_model.params = {"SIMD": 128}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
