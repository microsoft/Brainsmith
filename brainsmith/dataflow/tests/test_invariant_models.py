############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for invariant model types (two-phase construction)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.models import (
    InvariantInterfaceModel,
    InvariantKernelModel,
)
from brainsmith.dataflow.types import ShapeHierarchy


# =============================================================================
# Test InvariantInterfaceModel
# =============================================================================

def test_invariant_interface_model_creation():
    """Test InvariantInterfaceModel can be created with required fields."""
    model = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    assert model.name == "input"
    assert model.tensor_shape == (1, 224, 224, 768)
    assert model.block_shape == (1, 1, 768)
    assert model.stream_tiling == ["SIMD"]
    assert model.datatype == DataType["INT8"]
    assert model.is_weight is False


def test_invariant_interface_model_default_is_weight():
    """Test InvariantInterfaceModel is_weight defaults to False."""
    model = InvariantInterfaceModel(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
    )

    assert model.is_weight is False


def test_invariant_interface_model_immutable():
    """Test InvariantInterfaceModel is immutable (frozen dataclass)."""
    model = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    # Attempt to modify should raise FrozenInstanceError
    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        model.name = "modified"


def test_invariant_interface_model_weight_flag():
    """Test InvariantInterfaceModel can be created as weight."""
    model = InvariantInterfaceModel(
        name="weight",
        tensor_shape=(768, 768),
        block_shape=(768, 768),
        stream_tiling=None,
        datatype=DataType["INT8"],
        is_weight=True,
    )

    assert model.is_weight is True


# =============================================================================
# Test InvariantKernelModel
# =============================================================================

def test_invariant_kernel_model_creation():
    """Test InvariantKernelModel can be created with all required fields."""
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

    model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 3, 4, 6, 8, 12, 16}},
    )

    assert model.name == "TestKernel"
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert model.inputs[0] == input_inv
    assert model.outputs[0] == output_inv
    assert "SIMD" in model.parallelization_params
    assert 8 in model.parallelization_params["SIMD"]


def test_invariant_kernel_model_get_input():
    """Test InvariantKernelModel.get_input() lookup."""
    input_inv = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8}},
    )

    found = model.get_input("input")
    assert found is input_inv

    not_found = model.get_input("nonexistent")
    assert not_found is None


def test_invariant_kernel_model_get_output():
    """Test InvariantKernelModel.get_output() lookup."""
    output_inv = InvariantInterfaceModel(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    model = InvariantKernelModel(
        name="TestKernel",
        inputs=(),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8}},
    )

    found = model.get_output("output")
    assert found is output_inv

    not_found = model.get_output("nonexistent")
    assert not_found is None


def test_invariant_kernel_model_get_interface():
    """Test InvariantKernelModel.get_interface() searches both inputs and outputs."""
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
        stream_tiling=["PE"],
        datatype=DataType["INT8"],
    )

    model = InvariantKernelModel(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": {1, 2}, "PE": {1, 2}},
    )

    # Find input
    found_input = model.get_interface("input")
    assert found_input is input_inv

    # Find output
    found_output = model.get_interface("output")
    assert found_output is output_inv

    # Not found
    not_found = model.get_interface("nonexistent")
    assert not_found is None


def test_invariant_kernel_model_immutable():
    """Test InvariantKernelModel is immutable."""
    model = InvariantKernelModel(
        name="TestKernel",
        inputs=(),
        outputs=(),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={},
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        model.name = "modified"


def test_invariant_kernel_model_valid_ranges():
    """Test InvariantKernelModel stores valid parallelization ranges."""
    # Simulate divisors of 768
    valid_simd = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}

    input_inv = InvariantInterfaceModel(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    model = InvariantKernelModel(
        name="LayerNorm",
        inputs=(input_inv,),
        outputs=(),
        internal_datatypes={},
        invariant_constraints=[],
        variant_constraints=[],
        parallelization_params={"SIMD": valid_simd},
    )

    assert len(model.parallelization_params["SIMD"]) == 18
    assert 64 in model.parallelization_params["SIMD"]
    assert 128 in model.parallelization_params["SIMD"]
    assert 768 in model.parallelization_params["SIMD"]
    assert 1 in model.parallelization_params["SIMD"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
