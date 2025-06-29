############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Example usage of native relationship modeling"""

from brainsmith.core.dataflow.kernel import Kernel
from brainsmith.core.dataflow.interface import Interface
from brainsmith.core.dataflow.relationships import RelationType
from brainsmith.core.dataflow.types import InterfaceDirection, DataType


def test_matrix_multiplication_example():
    """Example: Matrix multiplication with native constraints"""
    
    # Create kernel with matrix multiplication interfaces
    kernel = Kernel(
        name="matrix_multiply",
        hw_module="matmul_v1",
        requires_burst_alignment=True,
        memory_architecture="HBM"
    )
    
    # Add interfaces with native constraints
    kernel.interfaces = [
        Interface(
            name="matrix_A",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(512, 512),
            block_dims=(64, 64),
            stream_dims=(8, 8),
            alignment=64,  # DMA alignment
            min_dims=(16, 16),  # Minimum for efficiency
            granularity=(8, 8)  # Must be multiples of 8
        ),
        Interface(
            name="matrix_B",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(512, 512),
            block_dims=(64, 64),
            stream_dims=(8, 8),
            alignment=64,
            min_dims=(16, 16),
            granularity=(8, 8)
        ),
        Interface(
            name="result_C",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT32"),
            tensor_dims=(512, 512),
            block_dims=(64, 64),
            stream_dims=(8, 8),
            alignment=128,  # Wider output alignment
            produces={"next_layer"}
        )
    ]
    
    # Add mathematical relationships for matrix multiplication
    kernel.add_relationship(
        source="matrix_A", source_dim=1,
        target="matrix_B", target_dim=0,
        relation=RelationType.EQUAL,
        description="Matrix A columns must equal Matrix B rows"
    )
    
    kernel.add_relationship(
        source="result_C", source_dim=0,
        target="matrix_A", target_dim=0,
        relation=RelationType.EQUAL,
        description="Result rows equal Matrix A rows"
    )
    
    kernel.add_relationship(
        source="result_C", source_dim=1,
        target="matrix_B", target_dim=1,
        relation=RelationType.EQUAL,
        description="Result columns equal Matrix B columns"
    )
    
    # Add architectural constraints
    kernel.add_constraint(
        name="dsp_usage",
        expression="matrix_A.stream[0] * matrix_A.stream[1]",
        operator="<=",
        value=64,
        description="DSP slice limitation"
    )
    
    kernel.add_constraint(
        name="memory_bandwidth",
        expression="matrix_A.bandwidth + matrix_B.bandwidth + result_C.bandwidth",
        operator="<=",
        value=25600,  # 25.6 GB/s
        description="Memory bandwidth limit"
    )
    
    # Add parameter dependencies
    kernel.add_dependency(
        dependent="total_operations",
        expression="matrix_A[0] * matrix_A[1] * matrix_B[1]",
        description="Total multiply-accumulate operations"
    )
    
    kernel.add_dependency(
        dependent="buffer_size",
        expression="max(matrix_A[0] * 64, 8192)",
        description="Accumulator buffer size requirement"
    )
    
    # Validate the kernel
    result = kernel.validate()
    assert result.is_valid, f"Validation failed: {result.get_detailed_report()}"
    
    # Check relationships were created
    assert len(kernel.relationships) == 3
    assert len(kernel.constraints) == 2
    assert len(kernel.dependencies) == 2
    
    # Check dataflow metadata was updated
    matrix_a = kernel.get_interface("matrix_A")
    matrix_b = kernel.get_interface("matrix_B")
    result_c = kernel.get_interface("result_C")
    
    # Basic relationship checking - dataflow metadata is updated automatically
    assert "matrix_B" in matrix_a.produces
    assert "matrix_A" in matrix_b.consumes
    assert "next_layer" in result_c.produces
    
    # Verify relationships can be queried
    a_relationships = kernel.get_relationships_for_interface("matrix_A")
    assert len(a_relationships) >= 1
    
    print("Matrix multiplication kernel created successfully!")
    print(f"  - {len(kernel.interfaces)} interfaces")
    print(f"  - {len(kernel.relationships)} relationships")
    print(f"  - {len(kernel.constraints)} constraints")
    print(f"  - {len(kernel.dependencies)} dependencies")
    print(f"  - Architecture: {kernel.memory_architecture}")
    print(f"  - Burst aligned: {kernel.requires_burst_alignment}")


def test_convolution_example():
    """Example: 2D Convolution with rich constraints"""
    
    kernel = Kernel(
        name="conv2d",
        hw_module="convolution_2d_v3",
        memory_architecture="distributed",
        pipeline_style="streaming"
    )
    
    # Input feature map
    kernel.interfaces.append(
        Interface(
            name="ifmap",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            tensor_dims=(1, 32, 224, 224),  # NCHW
            block_dims=(1, 32, 14, 14),     # Process 14x14 tiles  
            stream_dims=(1, 4, 1, 1),       # 4 channels parallel
            alignment=128,
            min_dims=(1, 1, 7, 7),
            synchronized_with={"weights"}
        )
    )
    
    # Convolution weights
    kernel.interfaces.append(
        Interface(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8"),
            tensor_dims=(64, 32, 3, 3),    # OIHW
            block_dims=(16, 32, 3, 3),      # 16 output channels at a time
            stream_dims=(4, 4, 1, 1),       # 4x4 parallel MAC units
            granularity=(16, None, None, None),  # Output channels multiple of 16
            synchronized_with={"ifmap"}
        )
    )
    
    # Output feature map
    kernel.interfaces.append(
        Interface(
            name="ofmap",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT16"),
            tensor_dims=(1, 64, 222, 222),  # NCHW (with valid padding)
            block_dims=(1, 16, 12, 12),     # Output tile size
            stream_dims=(1, 4, 1, 1),
            alignment=128
        )
    )
    
    # Channel relationships
    kernel.add_relationship(
        "weights", "ifmap", RelationType.EQUAL,
        source_dim=1, target_dim=1,
        description="Weight input channels = ifmap channels"
    )
    
    kernel.add_relationship(
        "ofmap", "weights", RelationType.EQUAL,
        source_dim=1, target_dim=0,
        description="Output channels = weight output channels"
    )
    
    # Spatial constraints (valid padding)
    kernel.add_constraint(
        "spatial_height",
        "ofmap[2]", "==", "ifmap[2] - weights[2] + 1",
        "Output height with valid padding"
    )
    
    kernel.add_constraint(
        "spatial_width", 
        "ofmap[3]", "==", "ifmap[3] - weights[3] + 1",
        "Output width with valid padding"
    )
    
    # Hardware constraints
    kernel.add_constraint(
        "dsp_limit",
        "weights.stream[0] * weights.stream[1]", "<=", 16,
        "DSP resource limitation"
    )
    
    # Power-of-two requirement
    kernel.requires_power_of_two.add("weights")
    
    # Dependencies
    kernel.add_dependency(
        "line_buffer_size",
        "ifmap[1] * weights[2] * ifmap[3]",
        "Line buffer for convolution sliding window"
    )
    
    # Validate
    result = kernel.validate()
    assert result.is_valid, f"Validation failed: {result.get_detailed_report()}"
    
    print("Convolution kernel created successfully!")
    print(f"  - Pipeline style: {kernel.pipeline_style}")
    print(f"  - Power-of-2 interfaces: {kernel.requires_power_of_two}")
    print(f"  - Synchronized interfaces: weights ↔ ifmap")


if __name__ == "__main__":
    test_matrix_multiplication_example()
    test_convolution_example()
    print("All examples completed successfully!")