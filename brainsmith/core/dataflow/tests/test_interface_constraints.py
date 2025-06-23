############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for Interface native constraints"""

import pytest
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.types import InterfaceDirection, DataType


def test_interface_alignment_constraint():
    """Test alignment constraint validation"""
    # Valid alignment (64 bytes)
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(1024,),  # 1024 bytes, divisible by 64
        block_dims=(64,),
        stream_dims=(8,),
        alignment=64
    )
    
    # Should not raise
    assert interface.alignment == 64
    
    # Invalid alignment (not divisible)
    with pytest.raises(ValueError, match="not aligned"):
        Interface(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(100,),  # 100 bytes, not divisible by 64
            block_dims=(10,),
            stream_dims=(1,),
            alignment=64
        )


def test_interface_dimension_bounds():
    """Test min/max dimension constraints"""
    # Valid dimensions within bounds
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 64),
        block_dims=(8, 8),
        stream_dims=(1, 1),
        min_dims=(16, 16),
        max_dims=(128, 128)
    )
    
    assert interface.min_dims == (16, 16)
    assert interface.max_dims == (128, 128)
    
    # Dimension below minimum
    with pytest.raises(ValueError, match="< min="):
        Interface(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(8, 64),  # First dim too small
            block_dims=(8, 8),
            stream_dims=(1, 1),
            min_dims=(16, 16)
        )
    
    # Dimension above maximum
    with pytest.raises(ValueError, match="> max="):
        Interface(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(256, 64),  # First dim too large
            block_dims=(8, 8),
            stream_dims=(1, 1),
            max_dims=(128, 128)
        )


def test_interface_granularity_constraint():
    """Test granularity constraint validation"""
    # Valid granularity (dimensions are multiples)
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 32),  # Both divisible by 8
        block_dims=(8, 8),
        stream_dims=(1, 1),
        granularity=(8, 8)
    )
    
    assert interface.granularity == (8, 8)
    
    # Invalid granularity (not divisible)
    with pytest.raises(ValueError, match="not divisible by granularity"):
        Interface(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(65, 32),  # 65 not divisible by 8
            block_dims=(8, 8),
            stream_dims=(1, 1),
            granularity=(8, 8)
        )


def test_interface_dataflow_metadata():
    """Test dataflow metadata functionality"""
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 64),
        block_dims=(8, 8),
        stream_dims=(1, 1)
    )
    
    # Initially empty
    assert len(interface.produces) == 0
    assert len(interface.consumes) == 0
    assert len(interface.synchronized_with) == 0
    
    # Add metadata
    interface.add_produces("output")
    interface.add_consumes("weights")
    interface.add_synchronized_with("bias")
    
    assert "output" in interface.produces
    assert "weights" in interface.consumes
    assert "bias" in interface.synchronized_with


def test_interface_constraint_checking():
    """Test constraint checking methods"""
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 64),
        block_dims=(8, 8),
        stream_dims=(1, 1),
        alignment=64,
        min_dims=(16, 16),
        granularity=(8, 8)
    )
    
    assert interface.has_constraint("alignment")
    assert interface.has_constraint("bounds")
    assert interface.has_constraint("granularity")
    assert not interface.has_constraint("unknown")


def test_interface_repr_with_constraints():
    """Test string representation includes constraints"""
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 64),
        block_dims=(8, 8),
        stream_dims=(1, 1),
        alignment=64,
        min_dims=(16, 16)
    )
    
    repr_str = repr(interface)
    assert "align=64" in repr_str
    assert "min=(16, 16)" in repr_str


def test_interface_validate_constraints_method():
    """Test direct constraint validation method"""
    # Valid interface
    interface = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        tensor_dims=(64, 64),
        block_dims=(8, 8),
        stream_dims=(1, 1),
        alignment=64,
        min_dims=(16, 16),
        max_dims=(128, 128),
        granularity=(8, 8)
    )
    
    errors = interface.validate_constraints()
    assert len(errors) == 0
    
    # Invalid interface (manually set bad values)
    interface.tensor_dims = (8, 65)  # Below min, not granular
    errors = interface.validate_constraints()
    assert len(errors) > 0
    assert any("< min=" in error for error in errors)
    assert any("not divisible by granularity" in error for error in errors)