############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Example usage of function-based block dimension specifications"""

import pytest
from typing import Dict, Any

from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.kernel_model import KernelModel
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.tiling_functions import (
    fixed_tiles, channel_major_tiling, adaptive_tiles, 
    parameterized_tiles, composite_tiling, memory_constrained_tiles
)
from brainsmith.core.dataflow.core.tiling_config import (
    TilingConfig, TilingStrategy, get_tiling_template
)
from brainsmith.core.dataflow.core.base import ParameterBinding


class TestFunctionBasedExamples:
    """Examples demonstrating function-based block dimension specifications"""
    
    def test_conv2d_with_channel_tiling(self):
        """Example: Conv2D with channel-major tiling function"""
        # Define interfaces with function-based tiling
        ifmap_def = InterfaceDefinition(
            name="ifmap",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="tile"),
            onnx_layout="NCHW"
        )
        
        weights_def = InterfaceDefinition(
            name="weights",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="full"),
            onnx_layout="OIHW"  # Output, Input, Height, Width
        )
        
        ofmap_def = InterfaceDefinition(
            name="ofmap",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT16"),
            block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="tile"),
            onnx_layout="NCHW"
        )
        
        # Create kernel definition
        kernel_def = KernelDefinition(
            name="conv2d",
            interface_definitions=[ifmap_def, weights_def, ofmap_def]
        )
        
        # Create models with specific tensor dimensions
        ifmap_model = ifmap_def.create_model(
            tensor_dims=(1, 128, 224, 224),
            parameter_binding={},
            config={"layout": "NCHW"}
        )
        
        weights_model = weights_def.create_model(
            tensor_dims=(256, 128, 3, 3),
            parameter_binding={},
            config={"layout": "OIHW"}
        )
        
        ofmap_model = ofmap_def.create_model(
            tensor_dims=(1, 256, 222, 222),
            parameter_binding={},
            config={"layout": "NCHW"}
        )
        
        # Verify block dimensions were derived correctly
        assert ifmap_model.block_dims == [(1, 32, 14, 14)]  # Channel tiled, spatial tiled
        assert weights_model.block_dims == [(32, 32, 3, 3)]  # Both channels tiled, spatial full
        assert ofmap_model.block_dims == [(1, 32, 14, 14)]  # Output channels tiled
        
        # Create kernel model and apply parallelism
        kernel_model = KernelModel(
            interface_models=[ifmap_model, weights_model, ofmap_model],
            definition=kernel_def
        )
        
        kernel_model.apply_parallelism({"ofmap": 8})
        
        # Check stream dimensions
        assert ofmap_model.stream_dims == (1, 8, 1, 1)  # Parallelism on channel dimension
    
    def test_matmul_with_parameterized_tiling(self):
        """Example: MatMul with parameter-based tiling"""
        # Define interfaces with parameterized tiling
        a_def = InterfaceDefinition(
            name="A",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=parameterized_tiles("TILE_M", "TILE_K")
        )
        
        b_def = InterfaceDefinition(
            name="B",
            direction=InterfaceDirection.WEIGHT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=parameterized_tiles("TILE_K", "TILE_N")
        )
        
        c_def = InterfaceDefinition(
            name="C",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT32"),
            block_dims_expr=parameterized_tiles("TILE_M", "TILE_N")
        )
        
        # Create kernel with relationships
        kernel_def = KernelDefinition(
            name="matmul",
            interface_definitions=[a_def, b_def, c_def]
        )
        
        # Add matmul constraint: A.cols == B.rows
        kernel_def.add_relationship("A", "B", source_dim=1, target_dim=0)
        
        # Create models with parameter binding
        params = {
            "TILE_M": 64,
            "TILE_K": 128,
            "TILE_N": 32
        }
        
        a_model = a_def.create_model((512, 1024), parameter_binding=params)
        b_model = b_def.create_model((1024, 256), parameter_binding=params)
        c_model = c_def.create_model((512, 256), parameter_binding=params)
        
        # Verify parameterized tiling
        assert a_model.block_dims == [(64, 128)]
        assert b_model.block_dims == [(128, 32)]
        assert c_model.block_dims == [(64, 32)]
    
    def test_adaptive_tiling_with_configuration(self):
        """Example: Adaptive tiling based on runtime configuration"""
        # Define interface with adaptive tiling
        data_def = InterfaceDefinition(
            name="data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("FP32"),
            block_dims_expr=adaptive_tiles(
                "optimization_mode",
                default=[1, 16, 256]  # Default for transformer
            )
        )
        
        # Test different configurations
        configs = [
            # Low latency mode - larger tiles
            {"optimization_mode": [1, 32, 512]},
            # High throughput mode - smaller tiles  
            {"optimization_mode": [1, 8, 128]},
            # Default mode - no config
            {}
        ]
        
        tensor_dims = (4, 64, 1024)  # [batch, heads, seq_len]
        
        for config in configs:
            model = data_def.create_model(
                tensor_dims=tensor_dims,
                parameter_binding={},
                config=config
            )
            
            if "optimization_mode" in config:
                expected = tuple(config["optimization_mode"])
            else:
                expected = (1, 16, 256)  # Default
                
            assert model.block_dims == [expected]
    
    def test_composite_tiling_strategy(self):
        """Example: Composite tiling that tries multiple strategies"""
        # Create a composite strategy that:
        # 1. First tries adaptive config
        # 2. Falls back to parameterized
        # 3. Finally uses fixed tiles
        composite_func = composite_tiling(
            adaptive_tiles("custom_tiles"),
            parameterized_tiles("TILE_H", "TILE_W"),
            fixed_tiles(28, 28)
        )
        
        data_def = InterfaceDefinition(
            name="feature_map",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT16"),
            block_dims_expr=composite_func
        )
        
        tensor_dims = (224, 224)
        
        # Test 1: Config provides custom tiles
        model1 = data_def.create_model(
            tensor_dims=tensor_dims,
            parameter_binding={},
            config={"custom_tiles": [56, 56]}
        )
        assert model1.block_dims == [(56, 56)]
        
        # Test 2: No config, but params available
        model2 = data_def.create_model(
            tensor_dims=tensor_dims,
            parameter_binding={"TILE_H": 112, "TILE_W": 112},
            config={}
        )
        assert model2.block_dims == [(112, 112)]
        
        # Test 3: No config or params, use fixed fallback
        model3 = data_def.create_model(
            tensor_dims=tensor_dims,
            parameter_binding={},
            config={}
        )
        assert model3.block_dims == [(28, 28)]
    
    def test_memory_constrained_tiling(self):
        """Example: Tiling constrained by memory limits"""
        # Create memory-constrained tiling for large tensors
        # Limit to 256KB per tile with FP16 elements
        memory_func = memory_constrained_tiles(
            memory_limit_bytes=256 * 1024,  # 256KB
            bytes_per_element=2  # FP16
        )
        
        large_tensor_def = InterfaceDefinition(
            name="large_activation",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("FP16"),
            block_dims_expr=memory_func
        )
        
        # Very large tensor that would exceed memory if not tiled
        tensor_dims = (512, 512, 512)  # 256MB with FP16
        
        model = large_tensor_def.create_model(
            tensor_dims=tensor_dims,
            parameter_binding={}
        )
        
        # Calculate actual memory usage of block
        block_elements = 1
        for dim in model.block_dims[0]:
            block_elements *= dim
        block_memory = block_elements * 2  # 2 bytes per FP16
        
        # Should be within limit
        assert block_memory <= 256 * 1024
        
        # Should have reduced dimensions
        assert all(b <= t for b, t in zip(model.block_dims[0], tensor_dims))
    
    def test_using_tiling_config(self):
        """Example: Using TilingConfig for structured configuration"""
        # Load a predefined template
        conv_config = get_tiling_template("conv2d_standard")
        
        # Modify for specific use case
        conv_config.channel_tile = 64  # Larger channel tiles
        
        # Create interface with config
        conv_input = InterfaceDefinition(
            name="input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=conv_config.to_function(),
            onnx_layout="NCHW"
        )
        
        # Create from JSON configuration
        json_config = """
        {
            "strategy": "composite",
            "sub_strategies": [
                {
                    "strategy": "adaptive",
                    "config_key": "layer_tiles"
                },
                {
                    "strategy": "channel_major",
                    "channel_tile": 32,
                    "spatial_mode": "tile"
                }
            ]
        }
        """
        
        config_from_json = TilingConfig.from_json(json_config)
        
        fallback_input = InterfaceDefinition(
            name="input2",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=config_from_json.to_function()
        )
        
        # Test the configuration
        tensor_dims = (1, 128, 224, 224)
        
        # Without layer_tiles config, falls back to channel_major
        model = fallback_input.create_model(
            tensor_dims=tensor_dims,
            parameter_binding={},
            config={"layout": "NCHW"}
        )
        assert model.block_dims == [(1, 32, 14, 14)]  # Falls back to channel_major with tile mode
    
    def test_performance_aware_tiling(self):
        """Example: Tiling that considers performance metrics"""
        # Create a custom tiling function that considers bandwidth
        def bandwidth_aware_tiles(max_bandwidth_mbps: float):
            def _tiles(tensor_dims, params, config):
                # Simple heuristic: smaller tiles for higher bandwidth
                if max_bandwidth_mbps > 10000:  # 10 GB/s
                    return [min(16, d) for d in tensor_dims]
                elif max_bandwidth_mbps > 5000:  # 5 GB/s
                    return [min(32, d) for d in tensor_dims]
                else:
                    return [min(64, d) for d in tensor_dims]
            return _tiles
        
        # Use with different bandwidth constraints
        high_bw_def = InterfaceDefinition(
            name="ddr_data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT32"),
            block_dims_expr=bandwidth_aware_tiles(20000)  # 20 GB/s
        )
        
        low_bw_def = InterfaceDefinition(
            name="sram_data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT32"),
            block_dims_expr=bandwidth_aware_tiles(2000)  # 2 GB/s
        )
        
        tensor_dims = (256, 256)
        
        high_bw_model = high_bw_def.create_model(tensor_dims, parameter_binding={})
        low_bw_model = low_bw_def.create_model(tensor_dims, parameter_binding={})
        
        # High bandwidth -> smaller tiles
        assert high_bw_model.block_dims == [(16, 16)]
        # Low bandwidth -> larger tiles
        assert low_bw_model.block_dims == [(64, 64)]


if __name__ == "__main__":
    pytest.main([__file__])