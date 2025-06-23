############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Example kernel definitions demonstrating block/stream dimension features"""

from typing import Dict, Any, List
from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.relationships import RelationType
from brainsmith.core.dataflow.core.tiling_functions import (
    fixed_tiles, channel_major_tiling, parameterized_tiles,
    adaptive_tiles, composite_tiling, memory_constrained_tiles,
    power_of_two_tiles, ratio_based_tiles, phase_dependent_tiles,
    full_tensor
)
from brainsmith.core.dataflow.core.tiling_config import TilingConfig, TilingStrategy


def create_matmul_kernel() -> KernelDefinition:
    """Create MatMul kernel with explicit parameterized tiling
    
    Demonstrates:
    - Parameterized tiling for flexible block dimensions
    - Relationship-based constraints
    - Parallelism propagation through matrix multiply
    """
    # Define interfaces with parameterized tiling
    a_def = InterfaceDefinition(
        name="A",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_K"),
        granularity=(1, 8)  # K dimension must be multiple of 8
    )
    
    b_def = InterfaceDefinition(
        name="B",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_K", "TILE_N"),
        granularity=(8, 1)  # K dimension must be multiple of 8
    )
    
    c_def = InterfaceDefinition(
        name="C",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT32"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_N")
    )
    
    # Create kernel with relationships
    kernel_def = KernelDefinition(
        name="matmul_tiled",
        interface_definitions=[a_def, b_def, c_def]
    )
    
    # Add matmul constraints
    kernel_def.add_relationship("A", "B", RelationType.EQUAL, 
                               source_dim=1, target_dim=0)
    kernel_def.add_relationship("A", "C", RelationType.EQUAL,
                               source_dim=0, target_dim=0)
    kernel_def.add_relationship("B", "C", RelationType.EQUAL,
                               source_dim=1, target_dim=1)
    
    return kernel_def


def create_conv2d_kernel() -> KernelDefinition:
    """Create Conv2D kernel with channel-major tiling
    
    Demonstrates:
    - Channel-major tiling for CNN operations
    - Layout-aware tiling (NCHW, OIHW)
    - Spatial tiling modes
    """
    # Input feature map with channel tiling
    ifmap_def = InterfaceDefinition(
        name="ifmap",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("UINT8"),
        block_dims_expr=channel_major_tiling(channel_tile="C_TILE", spatial_mode="tile"),
        onnx_layout="NCHW"
    )
    
    # Weights with channel tiling, full spatial
    weights_def = InterfaceDefinition(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=channel_major_tiling(channel_tile="C_TILE", spatial_mode="full"),
        onnx_layout="OIHW"  # Output channels, Input channels, Height, Width
    )
    
    # Output feature map with channel tiling
    ofmap_def = InterfaceDefinition(
        name="ofmap",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        block_dims_expr=channel_major_tiling(channel_tile="C_TILE", spatial_mode="tile"),
        onnx_layout="NCHW"
    )
    
    # Bias (optional)
    bias_def = InterfaceDefinition(
        name="bias",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT16"),
        block_dims_expr=parameterized_tiles("C_TILE"),
        optional=True
    )
    
    # Create kernel
    kernel_def = KernelDefinition(
        name="conv2d_channel_tiled",
        interface_definitions=[ifmap_def, weights_def, ofmap_def, bias_def]
    )
    
    
    return kernel_def


def create_transformer_kernel() -> KernelDefinition:
    """Create Transformer attention kernel with adaptive tiling
    
    Demonstrates:
    - Adaptive tiling based on configuration
    - Complex multi-tensor operations
    - Memory-aware tiling
    """
    # Query, Key, Value with adaptive tiling
    q_def = InterfaceDefinition(
        name="query",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP16"),
        block_dims_expr=adaptive_tiles("attention_mode", 
                                     default=[1, 8, 64]),  # [batch, heads, seq_tile]
        onnx_layout="NLC"  # Batch, Length, Channels
    )
    
    k_def = InterfaceDefinition(
        name="key",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP16"),
        block_dims_expr=adaptive_tiles("attention_mode",
                                     default=[1, 8, 64]),
        onnx_layout="NLC"
    )
    
    v_def = InterfaceDefinition(
        name="value",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP16"),
        block_dims_expr=adaptive_tiles("attention_mode",
                                     default=[1, 8, 64]),
        onnx_layout="NLC"
    )
    
    # Attention output with composite strategy
    attn_def = InterfaceDefinition(
        name="attention",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("FP16"),
        block_dims_expr=composite_tiling(
            adaptive_tiles("attention_mode"),
            memory_constrained_tiles(memory_limit_bytes=512*1024, bytes_per_element=2),
            fixed_tiles(1, 8, 64)
        ),
        onnx_layout="NLC"
    )
    
    # Create kernel
    kernel_def = KernelDefinition(
        name="transformer_attention",
        interface_definitions=[q_def, k_def, v_def, attn_def]
    )
    
    
    return kernel_def


def create_elementwise_kernel() -> KernelDefinition:
    """Create elementwise operation kernel
    
    Demonstrates:
    - Full tensor processing (no tiling by default)
    - Simple element-wise operations
    - Broadcast support
    """
    # Input tensors - default to full tensor
    input_a = InterfaceDefinition(
        name="input_a",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP32"),
        block_dims_expr=full_tensor()  # Process entire tensor
    )
    
    input_b = InterfaceDefinition(
        name="input_b",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP32"),
        block_dims_expr=full_tensor()  # Process entire tensor
    )
    
    # Output tensor
    output = InterfaceDefinition(
        name="output",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("FP32"),
        block_dims_expr=full_tensor()  # Process entire tensor
    )
    
    # Create kernel
    kernel_def = KernelDefinition(
        name="elementwise_add",
        interface_definitions=[input_a, input_b, output]
    )
    
    # Add relationships (broadcasting handled by runtime)
    kernel_def.add_relationship("input_a", "output", RelationType.EQUAL)
    kernel_def.add_relationship("input_b", "output", RelationType.EQUAL)
    
    return kernel_def


def create_csdf_kernel() -> KernelDefinition:
    """Create CSDF kernel with phase-dependent tiling
    
    Demonstrates:
    - Phase-dependent block dimensions
    - Cyclo-static dataflow patterns
    - Variable rate processing
    """
    # Input with phase-dependent tiling
    input_def = InterfaceDefinition(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("INT16"),
        block_dims_expr=phase_dependent_tiles([
            [128, 128],      # Phase 0: Large blocks
            [64, 64],        # Phase 1: Medium blocks
            [32, 32],        # Phase 2: Small blocks
            ["TILE", "TILE"] # Phase 3: Parameterized
        ])
    )
    
    # Processing buffer with fixed pattern
    buffer_def = InterfaceDefinition(
        name="buffer",
        direction=InterfaceDirection.WEIGHT,  # Use WEIGHT for internal buffers
        dtype=DataType.from_string("INT32"),
        block_dims_expr=fixed_tiles(64, 64)
    )
    
    # Output with matching phase pattern
    output_def = InterfaceDefinition(
        name="output",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("INT16"),
        block_dims_expr=phase_dependent_tiles([
            [64, 64],        # Phase 0: Accumulate from large
            [64, 64],        # Phase 1: Process medium
            [64, 64],        # Phase 2: Merge small
            ["TILE", "TILE"] # Phase 3: Output parameterized
        ])
    )
    
    # Create kernel
    kernel_def = KernelDefinition(
        name="csdf_variable_rate",
        interface_definitions=[input_def, buffer_def, output_def]
    )
    
    
    # Add phase relationships
    kernel_def.add_relationship("input", "output", RelationType.EQUAL)
    
    return kernel_def


def create_memory_optimized_kernel() -> KernelDefinition:
    """Create kernel optimized for memory constraints
    
    Demonstrates:
    - Memory-constrained tiling
    - Power-of-two tiles for hardware efficiency
    - Ratio-based tiling
    """
    # Large activation with memory constraints
    activation_def = InterfaceDefinition(
        name="activation",
        direction=InterfaceDirection.INPUT,
        dtype=DataType.from_string("FP32"),
        block_dims_expr=memory_constrained_tiles(
            memory_limit_bytes=1024*1024,  # 1MB limit
            bytes_per_element=4  # FP32
        )
    )
    
    # Weights with power-of-two tiling for hardware
    weights_def = InterfaceDefinition(
        name="weights",
        direction=InterfaceDirection.WEIGHT,
        dtype=DataType.from_string("INT8"),
        block_dims_expr=power_of_two_tiles(min_size=16, max_size=256)
    )
    
    # Output with ratio-based tiling
    output_def = InterfaceDefinition(
        name="output",
        direction=InterfaceDirection.OUTPUT,
        dtype=DataType.from_string("FP32"),
        block_dims_expr=ratio_based_tiles([1.0, 0.25, 0.1, 0.1])  # Full batch, 1/4 channels, 1/10 spatial
    )
    
    # Create kernel
    kernel_def = KernelDefinition(
        name="memory_optimized_layer",
        interface_definitions=[activation_def, weights_def, output_def]
    )
    
    return kernel_def


def demonstrate_kernel_usage():
    """Demonstrate how to use the example kernels"""
    print("=== Kernel Examples with Block/Stream Dimensions ===\n")
    
    # 1. MatMul Example
    print("1. MatMul Kernel:")
    matmul = create_matmul_kernel()
    
    # Create models with specific parameters
    params = {"TILE_M": 64, "TILE_K": 32, "TILE_N": 128}
    a_model = matmul.interface_definitions[0].create_model(
        (512, 1024), parameter_binding=params
    )
    b_model = matmul.interface_definitions[1].create_model(
        (1024, 256), parameter_binding=params
    )
    c_model = matmul.interface_definitions[2].create_model(
        (512, 256), parameter_binding=params
    )
    
    print(f"  A: tensor={a_model.tensor_dims}, block={a_model.block_dims[0]}")
    print(f"  B: tensor={b_model.tensor_dims}, block={b_model.block_dims[0]}")
    print(f"  C: tensor={c_model.tensor_dims}, block={c_model.block_dims[0]}")
    
    # Apply parallelism
    c_model.ipar = 16
    print(f"  C with iPar=16: stream_dims={c_model.stream_dims}")
    print()
    
    # 2. Conv2D Example
    print("2. Conv2D Kernel:")
    conv2d = create_conv2d_kernel()
    
    # Create models
    params = {"C_TILE": 32}
    ifmap = conv2d.interface_definitions[0].create_model(
        (1, 128, 224, 224), 
        parameter_binding=params,
        config={"layout": "NCHW"}
    )
    print(f"  Input: tensor={ifmap.tensor_dims}, block={ifmap.block_dims[0]}")
    print()
    
    # 3. Transformer Example
    print("3. Transformer Attention:")
    transformer = create_transformer_kernel()
    
    # Test different attention modes
    configs = [
        {"attention_mode": [1, 8, 32]},   # Small tiles
        {"attention_mode": [1, 8, 128]},  # Large tiles
        {}  # Default
    ]
    
    for i, config in enumerate(configs):
        q_model = transformer.interface_definitions[0].create_model(
            (1, 512, 512),  # [batch, seq_len, hidden]
            config=config
        )
        print(f"  Config {i}: block_dims={q_model.block_dims[0]}")
    print()
    
    # 4. CSDF Example
    print("4. CSDF Variable Rate:")
    csdf = create_csdf_kernel()
    
    # Show phase-dependent tiling
    params = {"TILE": 48}
    for phase in range(4):
        input_model = csdf.interface_definitions[0].create_model(
            (256, 256),
            parameter_binding=params,
            config={"csdf_phase": phase}
        )
        print(f"  Phase {phase}: block_dims={input_model.block_dims[0]}")
    
    print("\n=== Examples Complete ===")


# Usage examples for testing
def example_matmul_workflow():
    """Complete MatMul workflow example"""
    from brainsmith.core.dataflow.core.kernel_model import KernelModel
    
    # Create kernel
    kernel_def = create_matmul_kernel()
    
    # Define problem size
    M, K, N = 1024, 2048, 512
    
    # Try different tile configurations
    tile_configs = [
        {"TILE_M": 32, "TILE_K": 32, "TILE_N": 32},
        {"TILE_M": 64, "TILE_K": 64, "TILE_N": 64},
        {"TILE_M": 128, "TILE_K": 32, "TILE_N": 128},
    ]
    
    for config in tile_configs:
        # Create models
        a_model = kernel_def.interface_definitions[0].create_model(
            (M, K), parameter_binding=config
        )
        b_model = kernel_def.interface_definitions[1].create_model(
            (K, N), parameter_binding=config
        )
        c_model = kernel_def.interface_definitions[2].create_model(
            (M, N), parameter_binding=config
        )
        
        # Create kernel model
        kernel_model = KernelModel(
            interface_models=[a_model, b_model, c_model],
            definition=kernel_def
        )
        
        # Apply parallelism
        kernel_model.apply_parallelism({"C": 8})
        
        # Get metrics
        metrics = kernel_model.calculate_performance_metrics()
        print(f"Config {config}: II={metrics['initiation_interval']}, "
              f"BW={metrics['total_bandwidth_mbps']:.1f} MB/s")


if __name__ == "__main__":
    # Run demonstration
    demonstrate_kernel_usage()
    
    # Show workflow example
    print("\n=== MatMul Optimization Workflow ===")
    example_matmul_workflow()