############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for configured model types (two-phase construction)."""

import pytest
from qonnx.core.datatype import DataType

from brainsmith.dataflow.dse_models import (
    InterfaceDesignSpace,
    KernelDesignSpace,
    InterfaceDesignPoint,
    KernelDesignPoint,
)
from brainsmith.dataflow.types import ShapeHierarchy


# =============================================================================
# Test InterfaceDesignPoint
# =============================================================================

def test_configured_interface_model_creation():
    """Test InterfaceDesignPoint can be created."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
        is_weight=False,
    )

    configured = InterfaceDesignPoint(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    assert configured.stream_shape == (64,)
    assert configured.design_space is ds_interface


def test_configured_interface_model_property_delegation():
    """Test InterfaceDesignPoint properties delegate to design space."""
    ds_interface = InterfaceDesignSpace(
        name="test_interface",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["FLOAT32"],
        is_weight=True,
    )

    configured = InterfaceDesignPoint(
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
    """Test InterfaceDesignPoint.get_shape() for all hierarchies."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 224, 224, 768),
        block_shape=(1, 1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceDesignPoint(
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
    """Test InterfaceDesignPoint.get_shape() with invalid hierarchy raises error."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceDesignPoint(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    # Invalid hierarchy should raise ValueError
    with pytest.raises(ValueError, match="Invalid hierarchy"):
        configured.get_shape("invalid")


def test_configured_interface_model_folding_factors():
    """Test InterfaceDesignPoint computed properties."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceDesignPoint(
        design_space=ds_interface,
        stream_shape=(1, 64),  # Match dimensions with block_shape
    )

    # Test per-dimension folding shapes
    assert configured.tensor_blocks_shape == (1, 1)  # ceil((1,768) / (1,768))
    assert configured.stream_cycles_shape == (1, 12)  # ceil((1,768) / (1,64))

    # Test computed properties (products of folding shapes)
    assert configured.tensor_folding_factor == 1  # prod(tensor_blocks_shape)
    assert configured.block_folding_factor == 12  # prod(stream_cycles_shape)
    assert configured.streaming_bandwidth == 64  # prod(stream_shape) = 1 * 64
    assert configured.stream_width_bits == 64 * 8  # bandwidth * bitwidth


def test_configured_interface_model_immutable():
    """Test InterfaceDesignPoint is immutable."""
    ds_interface = InterfaceDesignSpace(
        name="input",
        tensor_shape=(1, 768),
        block_shape=(1, 768),
        stream_tiling=["SIMD"],
        datatype=DataType["INT8"],
    )

    configured = InterfaceDesignPoint(
        design_space=ds_interface,
        stream_shape=(64,),
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured.stream_shape = (128,)


# =============================================================================
# Test KernelDesignPoint
# =============================================================================

def test_design_point_creation():
    """Test KernelDesignPoint can be created."""
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
        optimization_constraints=[],
        dimensions={"SIMD": {1, 2, 4, 8, 16}},
    )

    input_cfg = InterfaceDesignPoint(design_space=input_inv, stream_shape=(64,))
    output_cfg = InterfaceDesignPoint(design_space=output_inv, stream_shape=(64,))

    configured_model = KernelDesignPoint(
        design_space=design_space_model,
        inputs=(input_cfg,),
        outputs=(output_cfg,),
        config={"SIMD": 64},
    )

    assert configured_model.name == "TestKernel"
    assert configured_model.config == {"SIMD": 64}
    assert len(configured_model.inputs) == 1
    assert len(configured_model.outputs) == 1


def test_design_point_property_delegation():
    """Test KernelDesignPoint properties delegate to design space."""
    design_space_model = KernelDesignSpace(
        name="LayerNorm",
        inputs=(),
        outputs=(),
        internal_datatypes={"accumulator": DataType["INT32"]},
        optimization_constraints=[],
        dimensions={},
    )

    configured_model = KernelDesignPoint(
        design_space=design_space_model,
        inputs=(),
        outputs=(),
        config={},
    )

    # Properties should delegate to design space
    assert configured_model.name == "LayerNorm"
    assert configured_model.internal_datatypes == {"accumulator": DataType["INT32"]}


def test_design_point_immutable():
    """Test KernelDesignPoint is immutable."""
    design_space_model = KernelDesignSpace(
        name="TestKernel",
        inputs=(),
        outputs=(),
        internal_datatypes={},
        optimization_constraints=[],
        dimensions={},
    )

    configured_model = KernelDesignPoint(
        design_space=design_space_model,
        inputs=(),
        outputs=(),
        config={},
    )

    with pytest.raises(Exception):  # dataclasses.FrozenInstanceError
        configured_model.params = {"SIMD": 128}


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
