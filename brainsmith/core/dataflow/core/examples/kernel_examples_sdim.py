############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Example kernel definitions using the new SDIM architecture

This file demonstrates the clean separation of inputs and outputs,
multi-dimensional SDIM configuration, and DEPENDENT relationships.
"""

from typing import Dict, Any, List
from brainsmith.core.dataflow.core import (
    InputDefinition, OutputDefinition, KernelDefinitionV2, KernelModelV2,
    DataType, RelationType,
    fixed_tiles, parameterized_tiles, adaptive_tiles
)


def create_matmul_kernel() -> KernelDefinitionV2:
    """Create MatMul kernel with SDIM support
    
    Demonstrates:
    - Weights as regular inputs (no WEIGHT type)
    - DEPENDENT relationship for K dimension
    - Multi-dimensional SDIM configuration
    """
    kernel_def = KernelDefinitionV2(name="matmul_sdim")
    
    # All inputs (no special WEIGHT type)
    kernel_def.add_input(InputDefinition(
        name="A",  # [M, K]
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_K"),
        granularity=(1, 8)  # K dimension must be multiple of 8
    ))
    
    kernel_def.add_input(InputDefinition(
        name="B",  # [K, N] - weights are just inputs
        dtype=DataType.from_string("INT8"),
        block_dims_expr=parameterized_tiles("TILE_K", "TILE_N"),
        granularity=(8, 1)  # K dimension must be multiple of 8
    ))
    
    # Output (no SDIM configuration needed)
    kernel_def.add_output(OutputDefinition(
        name="C",  # [M, N]
        dtype=DataType.from_string("INT32"),
        block_dims_expr=parameterized_tiles("TILE_M", "TILE_N")
    ))
    
    # DEPENDENT relationship for K dimension
    kernel_def.add_relationship(
        "A", "B", RelationType.DEPENDENT,
        source_dim=1, target_dim=0,
        dependency_type="copy",
        description="K dimensions must stream together"
    )
    
    return kernel_def


def create_conv2d_kernel() -> KernelDefinitionV2:
    """Create Conv2D kernel with clean architecture
    
    Demonstrates:
    - All weights/bias as inputs
    - Multi-dimensional SDIM for spatial/channel streaming
    - No output SDIM configuration
    """
    kernel_def = KernelDefinitionV2(name="conv2d_sdim")
    
    # Input feature map
    kernel_def.add_input(InputDefinition(
        name="ifmap",  # [N, C, H, W]
        dtype=DataType.from_string("UINT8"),
        block_dims_expr=fixed_tiles(1, 32, 14, 14),
        onnx_layout="NCHW"
    ))
    
    # Convolution kernels (not WEIGHT type)
    kernel_def.add_input(InputDefinition(
        name="kernels",  # [OC, IC, KH, KW]
        dtype=DataType.from_string("INT8"),
        block_dims_expr=fixed_tiles(64, 32, 3, 3),
        onnx_layout="OIHW"
    ))
    
    # Bias (also just an input)
    kernel_def.add_input(InputDefinition(
        name="bias",  # [OC]
        dtype=DataType.from_string("INT32"),
        block_dims_expr=fixed_tiles(64)
    ))
    
    # Output feature map
    kernel_def.add_output(OutputDefinition(
        name="ofmap",  # [N, OC, H, W]
        dtype=DataType.from_string("INT32"),
        block_dims_expr=fixed_tiles(1, 64, 14, 14),
        onnx_layout="NCHW"
    ))
    
    # Channel relationships
    kernel_def.add_relationship(
        "ifmap", "kernels", RelationType.EQUAL,
        source_dim=1, target_dim=1,
        description="Input channels must match"
    )
    
    kernel_def.add_relationship(
        "kernels", "bias", RelationType.EQUAL,
        source_dim=0, target_dim=0,
        description="Output channels must match"
    )
    
    return kernel_def


def create_elementwise_kernel() -> KernelDefinitionV2:
    """Create element-wise operation with SDIM
    
    Demonstrates:
    - Simple EQUAL relationship
    - Uniform SDIM configuration
    - Clean input/output separation
    """
    kernel_def = KernelDefinitionV2(name="elementwise_add")
    
    # Two inputs that must stream together
    kernel_def.add_input(InputDefinition(
        name="x",
        dtype=DataType.from_string("FP16"),
        block_dims_expr=fixed_tiles(64, 64)
    ))
    
    kernel_def.add_input(InputDefinition(
        name="y",
        dtype=DataType.from_string("FP16"),
        block_dims_expr=fixed_tiles(64, 64)
    ))
    
    # Output
    kernel_def.add_output(OutputDefinition(
        name="z",
        dtype=DataType.from_string("FP16"),
        block_dims_expr=fixed_tiles(64, 64)
    ))
    
    # Inputs must stream at same rate
    kernel_def.add_relationship("x", "y", RelationType.EQUAL)
    
    return kernel_def


def create_pooling_kernel() -> KernelDefinitionV2:
    """Create pooling kernel with spatial streaming
    
    Demonstrates:
    - Spatial dimension SDIM configuration
    - No weight inputs (pooling has no weights)
    - Output rate determined by pooling pattern
    """
    kernel_def = KernelDefinitionV2(name="maxpool2d")
    
    # Input only (no weights for pooling)
    kernel_def.add_input(InputDefinition(
        name="input",  # [N, C, H, W]
        dtype=DataType.from_string("FP16"),
        block_dims_expr=fixed_tiles(1, 16, 28, 28),
        onnx_layout="NCHW"
    ))
    
    # Output with different spatial dimensions
    kernel_def.add_output(OutputDefinition(
        name="output",  # [N, C, H/2, W/2]
        dtype=DataType.from_string("FP16"),
        block_dims_expr=fixed_tiles(1, 16, 14, 14),
        onnx_layout="NCHW"
    ))
    
    # No relationships needed for single input
    
    return kernel_def


def demonstrate_sdim_configuration():
    """Demonstrate SDIM configuration with the new API"""
    print("=== SDIM Configuration Examples ===\n")
    
    # Create MatMul kernel
    matmul_def = create_matmul_kernel()
    
    # Create models with concrete shapes
    a_model = matmul_def.get_input("A").create_model(
        (512, 256), {"TILE_M": 64, "TILE_K": 32}
    )
    b_model = matmul_def.get_input("B").create_model(
        (256, 1024), {"TILE_K": 32, "TILE_N": 128}
    )
    c_model = matmul_def.get_output("C").create_model(
        (512, 1024), {"TILE_M": 64, "TILE_N": 128}
    )
    
    # Create kernel model
    kernel = KernelModelV2(
        input_models=[a_model, b_model],
        output_models=[c_model],
        definition=matmul_def
    )
    
    # Configure SDIM - only inputs!
    print("1. MatMul SDIM Configuration:")
    kernel.configure_sdim({
        "A": [8, 16],  # Stream 8x16 patch of A
        # B's first dimension constrained by DEPENDENT relationship
    })
    
    sdim_state = kernel.get_sdim_state()
    print(f"   A: sdim={sdim_state['A']}")
    print(f"   B: sdim={sdim_state['B']} (constrained by relationship)")
    print(f"   C: No SDIM (output rate computed)\n")
    
    # Conv2D example
    conv_def = create_conv2d_kernel()
    
    ifmap_model = conv_def.get_input("ifmap").create_model((1, 256, 224, 224))
    kernel_model = conv_def.get_input("kernels").create_model((512, 256, 3, 3))
    bias_model = conv_def.get_input("bias").create_model((512,))
    ofmap_model = conv_def.get_output("ofmap").create_model((1, 512, 224, 224))
    
    conv_kernel = KernelModelV2(
        input_models=[ifmap_model, kernel_model, bias_model],
        output_models=[ofmap_model],
        definition=conv_def
    )
    
    print("2. Conv2D SDIM Configuration:")
    conv_kernel.configure_sdim({
        "ifmap": [1, 8, 1, 1],     # Stream 8 channels at a time
        "kernels": [16, 8, 3, 3],  # 16 output channels, 8 input channels
        # bias automatically gets [16] from relationship
    })
    
    conv_state = conv_kernel.get_sdim_state()
    print(f"   ifmap: sdim={conv_state['ifmap']}")
    print(f"   kernels: sdim={conv_state['kernels']}")
    print(f"   bias: sdim={conv_state['bias']} (from relationship)")
    print(f"   ofmap: No SDIM (computed from convolution pattern)\n")
    
    # Element-wise example
    elem_def = create_elementwise_kernel()
    
    x_model = elem_def.get_input("x").create_model((1024, 1024))
    y_model = elem_def.get_input("y").create_model((1024, 1024))
    z_model = elem_def.get_output("z").create_model((1024, 1024))
    
    elem_kernel = KernelModelV2(
        input_models=[x_model, y_model],
        output_models=[z_model],
        definition=elem_def
    )
    
    print("3. Element-wise SDIM Configuration:")
    # Uniform configuration
    elem_kernel.configure_sdim({"x": 16})  # y gets same from EQUAL relationship
    
    elem_state = elem_kernel.get_sdim_state()
    print(f"   x: sdim={elem_state['x']}")
    print(f"   y: sdim={elem_state['y']} (from EQUAL relationship)")
    print(f"   z: Streams at rate determined by operation\n")
    
    # Show exposed parameters
    print("4. Exposed SDIM Parameters:")
    params = conv_kernel.get_sdim_parameters()
    for name, info in params.items():
        print(f"   {name}: {info.total_dimensions}D, free dims: {info.free_dimensions}")


def demonstrate_migration():
    """Show key differences from old API"""
    print("\n=== Migration Guide ===\n")
    
    print("OLD API:")
    print("--------")
    print("kernel_def.add_interface(InterfaceDefinition(")
    print("    name='weights',")
    print("    direction=InterfaceDirection.WEIGHT,  # Special type")
    print("    ...")
    print("))")
    print("kernel.apply_parallelism({'input': 16, 'weights': 8, 'output': 16})")
    print()
    
    print("NEW API:")
    print("--------")
    print("kernel_def.add_input(InputDefinition(")
    print("    name='weights',  # Just an input!")
    print("    ...")
    print("))")
    print("kernel.configure_sdim({'input': 16, 'weights': 8})  # No output config!")
    print()
    
    print("Key Changes:")
    print("1. No WEIGHT type - use InputDefinition")
    print("2. Separate add_input() and add_output() methods")
    print("3. configure_sdim() only for inputs")
    print("4. Use DEPENDENT for dimension-specific constraints")
    print("5. Output streaming rates are computed, not configured")


if __name__ == "__main__":
    demonstrate_sdim_configuration()
    demonstrate_migration()