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
    InterfaceDesignSpace,
    KernelDesignSpace,
    InterfaceConfiguration,
    KernelInstance,
)
from brainsmith.dataflow.types import ShapeHierarchy


# =============================================================================
# Test InterfaceConfiguration
# =============================================================================

def test_configured_interface_model_creation():
    """Test InterfaceConfiguration can be created."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    assert configured.stream_shape == (64,)
    assert configured.design_space is ds_interface


def test_configured_interface_model_property_delegation():
    """Test InterfaceConfiguration properties delegate to design space."""
    ds_interface = InterfaceDesignSpace(
        name="test_interface",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
        is_weight=True,
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
        stream_shape=(32,),
    )

    # Properties should delegate to design space
    assert configured.name == "test_interface"
    assert configured.tensor_shape == (1, 224, 224, 768)
    assert configured.block_shape == (1, 1, 768)
    assert configured.datatype == DataType["FLOAT32"]
    assert configured.is_weight is True


def test_configured_interface_model_get_shape():
    """Test InterfaceConfiguration.get_shape() for all hierarchies."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
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
    """Test InterfaceConfiguration.get_shape() with invalid hierarchy raises error."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    # Invalid hierarchy should raise ValueError
    with pytest.raises(ValueError, match="Invalid hierarchy"):
        configured.get_shape("invalid")


def test_configured_interface_model_folding_factors():
    """Test InterfaceConfiguration computed properties."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
        stream_shape=(1, 64),  # Match dimensions with block_shape
    )

    # Test computed properties
    assert configured.tensor_folding_factor == 1  # tensor / block = (1,768) / (1,768)
    assert configured.block_folding_factor == 12  # block / stream = (1,768) / (1,64)
    assert configured.streaming_bandwidth == 64  # prod(stream_shape) = 1 * 64
    assert configured.stream_width_bits == 64 * 8  # bandwidth * bitwidth


def test_configured_interface_model_immutable():
    """Test InterfaceConfiguration is immutable."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceConfiguration(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured.stream_shape = (128,)


# =============================================================================
# Test KernelInstance
# =============================================================================

def test_kernel_instance_creation():
    """Test KernelInstance can be created."""
    input_inv = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    output_inv = InterfaceDesignSpace(
        name="output",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    design_space_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(input_inv,),
        outputs=(output_inv,),
        internal_datatypes={},
        parametric_constraints=[],
        parallelization_params={"SIMD": {1, 2, 4, 8, 16}},
    )

    input_cfg = InterfaceConfiguration(design_space=input_inv, stream_shape=(64,))
    output_cfg = InterfaceConfiguration(design_space=output_inv, stream_shape=(64,))

    configured_model = KernelInstance(
        design_space=design_space_model,
        inputs=(input_cfg,),
        outputs=(output_cfg,),
        params={"SIMD": 64},
    )

    assert configured_model.name == "TestKernel"
    assert configured_model.params == {"SIMD": 64}
    assert len(configured_model.inputs) == 1
    assert len(configured_model.outputs) == 1


def test_kernel_instance_property_delegation():
    """Test KernelInstance properties delegate to design space."""
    design_space_model = KernelDesignSpace(
        name="LayerNorm",
        inputs=(),
        outputs=(),
        internal_datatypes={"accumulator": DataType["INT32"]},
        parametric_constraints=[],
        parallelization_params={},
    )

    configured_model = KernelInstance(
        design_space=design_space_model,
        inputs=(),
        outputs=(),
        params={},
    )

    # Properties should delegate to design space
    assert configured_model.name == "LayerNorm"
    assert configured_model.internal_datatypes == {"accumulator": DataType["INT32"]}


def test_kernel_instance_immutable():
    """Test KernelInstance is immutable."""
    design_space_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(),
        outputs=(),
        internal_datatypes={},
        parametric_constraints=[],
        parallelization_params={},
    )

    configured_model = KernelInstance(
        design_space=design_space_model,
        inputs=(),
        outputs=(),
        params={},
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured_model.params = {"SIMD": 128}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
