############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for design space model types (two-phase construction)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.dse_models import (
    InterfaceDesignSpace,
    KernelDesignSpace,
)
from brainsmith.dataflow.types import ShapeHierarchy


# =============================================================================
# Test InterfaceDesignSpace
# =============================================================================

def test_invariant_interface_model_creation():
    """Test InterfaceDesignSpace can be created with required fields."""
    model = InterfaceDesignSpace(
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
    """Test InterfaceDesignSpace is_weight defaults to False."""
    model = InterfaceDesignSpace(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
    )

    assert model.is_weight is False


def test_invariant_interface_model_immutable():
    """Test InterfaceDesignSpace is immutable (frozen dataclass)."""
    model = InterfaceDesignSpace(
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
    """Test InterfaceDesignSpace can be created as weight."""
    model = InterfaceDesignSpace(
        name="weight",
        tensor_shape=(768, 768),
        block_shape=(768, 768),
        stream_tiling=None,
        datatype=DataType["INT8"],
        is_weight=True,
    )

    assert model.is_weight is True


# =============================================================================
# Test KernelDesignSpace
# =============================================================================

def test_kernel_design_space_creation():
    """Test KernelDesignSpace can be created with all required fields."""
    input_ds = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_ds = InterfaceDesignSpace(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    model = KernelDesignSpace(
        name="TestKernel",
        inputs={"input": input_ds},
        outputs={"output": output_ds},
        internal_datatypes={},
        optimization_constraints=[],
        parameters={"SIMD": {1, 2, 3, 4, 6, 8, 12, 16}},
    )

    assert model.name == "TestKernel"
    assert len(model.inputs) == 1
    assert len(model.outputs) == 1
    assert "input" in model.inputs
    assert "output" in model.outputs
    assert model.inputs["input"] == input_ds
    assert model.outputs["output"] == output_ds
    assert "SIMD" in model.parameters
    assert 8 in model.parameters["SIMD"]


def test_kernel_design_space_dict_access():
    """Test KernelDesignSpace dict-based interface access."""
    input_ds = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_ds = InterfaceDesignSpace(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["PE"],
        datatype=DataType["INT8"],
    )

    model = KernelDesignSpace(
        name="TestKernel",
        inputs={"input": input_ds},
        outputs={"output": output_ds},
        internal_datatypes={},
        optimization_constraints=[],
        parameters={"SIMD": {1, 2}, "PE": {1, 2}},
    )

    # Dict access works
    assert model.inputs["input"] is input_ds
    assert model.outputs["output"] is output_ds

    # Check membership
    assert "input" in model.inputs
    assert "output" in model.outputs
    assert "nonexistent" not in model.inputs


def test_kernel_design_space_immutable():
    """Test KernelDesignSpace is immutable."""
    model = KernelDesignSpace(
        name="TestKernel",
        inputs={},
        outputs={},
        internal_datatypes={},
        optimization_constraints=[],
        parameters={},
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        model.name = "modified"


def test_kernel_design_space_valid_ranges():
    """Test KernelDesignSpace stores valid parallelization ranges."""
    # Simulate divisors of 768
    valid_simd = {1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 768}

    input_ds = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    model = KernelDesignSpace(
        name="LayerNorm",
        inputs={"input": input_ds},
        outputs={},
        internal_datatypes={},
        optimization_constraints=[],
        parameters={"SIMD": valid_simd},
    )

    assert len(model.parameters["SIMD"]) == 18
    assert 64 in model.parameters["SIMD"]
    assert 128 in model.parameters["SIMD"]
    assert 768 in model.parameters["SIMD"]
    assert 1 in model.parameters["SIMD"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
