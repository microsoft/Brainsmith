############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for Kernel native relationships and constraints"""

import pytest
from brainsmith.core.dataflow.kernel import Kernel
from brainsmith.core.dataflow.interface import Interface
from brainsmith.core.dataflow.relationships import RelationType
from brainsmith.core.dataflow.types import InterfaceDirection, DataType


def test_kernel_add_relationship():
    """Test adding relationships between interfaces"""
    # Create kernel with matrix multiplication interfaces
    kernel = Kernel(
        name="matmul",
        interfaces=[
            Interface("A", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(512, 512), block_dims=(64, 64), stream_dims=(8, 8)),
            Interface("B", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(512, 512), block_dims=(64, 64), stream_dims=(8, 8)),
            Interface("C", InterfaceDirection.OUTPUT, DataType.from_string("INT32"),
                     tensor_dims=(512, 512), block_dims=(64, 64), stream_dims=(8, 8))
        ]
    )
    
    # Add matrix multiplication relationships
    kernel.add_relationship("A", "B", RelationType.EQUAL, source_dim=1, target_dim=0,
                           description="Matrix A columns = Matrix B rows")
    kernel.add_relationship("C", "A", RelationType.EQUAL, source_dim=0, target_dim=0,
                           description="Result rows = Matrix A rows")
    kernel.add_relationship("C", "B", RelationType.EQUAL, source_dim=1, target_dim=1,
                           description="Result columns = Matrix B columns")
    
    # Check relationships were added
    assert len(kernel.relationships) == 3
    
    # Check dataflow metadata was updated
    intf_a = kernel.get_interface("A")
    intf_b = kernel.get_interface("B")
    intf_c = kernel.get_interface("C")
    
    assert "B" in intf_a.produces
    assert "C" in intf_a.produces
    assert "A" in intf_b.consumes
    assert "B" in intf_c.consumes


def test_kernel_add_constraint():
    """Test adding architectural constraints"""
    kernel = Kernel(
        name="conv2d",
        interfaces=[
            Interface("weights", InterfaceDirection.WEIGHT, DataType.from_string("INT8"),
                     tensor_dims=(64, 32, 3, 3), block_dims=(16, 32, 3, 3), stream_dims=(4, 4, 1, 1))
        ]
    )
    
    # Add DSP constraint
    kernel.add_constraint(
        name="dsp_limit",
        expression="weights.stream[0] * weights.stream[1]",
        operator="<=",
        value=16,
        description="DSP resource limitation"
    )
    
    # Check constraint was added
    assert len(kernel.constraints) == 1
    constraint = kernel.constraints[0]
    assert constraint.name == "dsp_limit"
    assert constraint.operator == "<="
    assert constraint.value == 16


def test_kernel_add_dependency():
    """Test adding parameter dependencies"""
    kernel = Kernel(
        name="buffer",
        interfaces=[
            Interface("input", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(64,), block_dims=(8,), stream_dims=(1,))
        ]
    )
    
    # Add buffer size dependency
    kernel.add_dependency(
        dependent="buffer_size",
        expression="max(input[0] * 64, 4096)",
        description="Buffer size based on input dimension"
    )
    
    # Check dependency was added
    assert len(kernel.dependencies) == 1
    dependency = kernel.dependencies[0]
    assert dependency.dependent == "buffer_size"
    assert dependency.expression == "max(input[0] * 64, 4096)"


def test_kernel_validation():
    """Test kernel validation with relationships"""
    # Create valid kernel
    kernel = Kernel(
        name="matmul",
        interfaces=[
            Interface("A", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(512, 256), block_dims=(64, 32), stream_dims=(8, 4)),
            Interface("B", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(256, 512), block_dims=(32, 64), stream_dims=(4, 8)),
            Interface("C", InterfaceDirection.OUTPUT, DataType.from_string("INT32"),
                     tensor_dims=(512, 512), block_dims=(64, 64), stream_dims=(8, 8))
        ]
    )
    
    # Add valid relationships
    kernel.add_relationship("A", "B", RelationType.EQUAL, source_dim=1, target_dim=0)
    kernel.add_relationship("C", "A", RelationType.EQUAL, source_dim=0, target_dim=0)
    kernel.add_relationship("C", "B", RelationType.EQUAL, source_dim=1, target_dim=1)
    
    # Validation should pass
    result = kernel.validate()
    assert result.is_valid
    
    # Create invalid kernel (incompatible dimensions)
    kernel_invalid = Kernel(
        name="matmul_invalid",
        interfaces=[
            Interface("A", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(512, 256), block_dims=(64, 32), stream_dims=(8, 4)),
            Interface("B", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(128, 512), block_dims=(16, 64), stream_dims=(2, 8)),  # Wrong size
        ]
    )
    
    # Add incompatible relationship
    kernel_invalid.add_relationship("A", "B", RelationType.EQUAL, source_dim=1, target_dim=0)
    
    # Validation should fail
    result = kernel_invalid.validate()
    assert not result.is_valid
    assert len(result.violations) > 0


def test_kernel_relationship_queries():
    """Test querying relationships"""
    kernel = Kernel(
        name="test",
        interfaces=[
            Interface("input", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(64, 64), block_dims=(8, 8), stream_dims=(1, 1)),
            Interface("weights", InterfaceDirection.WEIGHT, DataType.from_string("INT8"),
                     tensor_dims=(64, 64), block_dims=(8, 8), stream_dims=(1, 1)),
            Interface("output", InterfaceDirection.OUTPUT, DataType.from_string("INT16"),
                     tensor_dims=(64, 64), block_dims=(8, 8), stream_dims=(1, 1))
        ]
    )
    
    # Add relationships
    kernel.add_relationship("input", "weights", RelationType.EQUAL, source_dim=0, target_dim=1)
    kernel.add_relationship("output", "input", RelationType.EQUAL, source_dim=0, target_dim=0)
    
    # Test relationship queries
    input_rels = kernel.get_relationships_for_interface("input")
    assert len(input_rels) == 2
    
    input_deps = kernel.get_dependent_interfaces("input")
    assert "output" in input_deps
    
    constraint_graph = kernel.get_constraint_graph()
    assert "weights" in constraint_graph
    assert "output" in constraint_graph


def test_kernel_architectural_requirements():
    """Test architectural requirement flags"""
    kernel = Kernel(
        name="dma",
        interfaces=[
            Interface("data", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(1024,), block_dims=(64,), stream_dims=(8,))
        ],
        requires_burst_alignment=True,
        memory_architecture="HBM",
        pipeline_style="streaming"
    )
    
    kernel.requires_power_of_two.add("data")
    
    assert kernel.requires_burst_alignment
    assert kernel.memory_architecture == "HBM"
    assert kernel.pipeline_style == "streaming"
    assert "data" in kernel.requires_power_of_two


def test_kernel_validate_and_raise():
    """Test validate_and_raise method"""
    kernel = Kernel(
        name="invalid",
        interfaces=[
            Interface("A", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(64,), block_dims=(8,), stream_dims=(1,)),
            Interface("B", InterfaceDirection.INPUT, DataType.from_string("INT8"),
                     tensor_dims=(32,), block_dims=(4,), stream_dims=(1,))  # Different size
        ]
    )
    
    # Add incompatible relationship
    kernel.add_relationship("A", "B", RelationType.EQUAL)
    
    # Should raise exception
    with pytest.raises(ValueError, match="Constraint validation failed"):
        kernel.validate_and_raise()