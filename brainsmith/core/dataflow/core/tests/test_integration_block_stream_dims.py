############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Integration tests for block/stream dimension system"""

import pytest
from typing import Dict, Any, List

from brainsmith.core.dataflow.core.kernel_definition import KernelDefinition
from brainsmith.core.dataflow.core.kernel_model import KernelModel
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection
from brainsmith.core.dataflow.core.relationships import RelationType
from brainsmith.core.dataflow.core.tiling_functions import (
    fixed_tiles, channel_major_tiling, parameterized_tiles,
    adaptive_tiles, composite_tiling, memory_constrained_tiles
)
from brainsmith.core.dataflow.core.tiling_config import TilingConfig, TilingStrategy


class TestBlockStreamIntegration:
    """Integration tests for complete kernel definitions with block/stream dimensions"""
    
    def test_complete_matmul_kernel(self):
        """Test complete MatMul kernel with parameterized tiling and relationships"""
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
        kernel_def.add_relationship("A", "B", RelationType.EQUAL, source_dim=1, target_dim=0)
        # Add output constraints
        kernel_def.add_relationship("A", "C", RelationType.EQUAL, source_dim=0, target_dim=0)
        kernel_def.add_relationship("B", "C", RelationType.EQUAL, source_dim=1, target_dim=1)
        
        # Create models with specific dimensions
        params = {
            "TILE_M": 64,
            "TILE_K": 32,
            "TILE_N": 128
        }
        
        a_model = a_def.create_model((512, 1024), parameter_binding=params)
        b_model = b_def.create_model((1024, 256), parameter_binding=params)
        c_model = c_def.create_model((512, 256), parameter_binding=params)
        
        # Create kernel model
        kernel_model = KernelModel(
            interface_models=[a_model, b_model, c_model],
            definition=kernel_def
        )
        
        # Verify block dimensions
        assert a_model.block_dims == [(64, 32)]
        assert b_model.block_dims == [(32, 128)]
        assert c_model.block_dims == [(64, 128)]
        
        # Apply parallelism
        kernel_model.apply_parallelism({"C": 16})
        
        # Verify parallelism propagation
        assert c_model.ipar == 16
        assert c_model.stream_dims == (16, 1)  # Applied to M dimension
        
        # Check performance metrics
        metrics = kernel_model.calculate_performance_metrics()
        assert "total_bandwidth_mbps" in metrics
        assert "initiation_interval" in metrics
    
    def test_complete_conv2d_kernel(self):
        """Test complete Conv2D kernel with channel-major tiling"""
        # Define interfaces with channel-major tiling
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
            onnx_layout="OIHW"
        )
        
        ofmap_def = InterfaceDefinition(
            name="ofmap",
            direction=InterfaceDirection.OUTPUT,
            dtype=DataType.from_string("INT16"),
            block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="tile"),
            onnx_layout="NCHW"
        )
        
        # Create kernel
        kernel_def = KernelDefinition(
            name="conv2d",
            interface_definitions=[ifmap_def, weights_def, ofmap_def]
        )
        
        # Create models
        ifmap_model = ifmap_def.create_model(
            tensor_dims=(1, 128, 224, 224),
            config={"layout": "NCHW"}
        )
        
        weights_model = weights_def.create_model(
            tensor_dims=(256, 128, 3, 3),
            config={"layout": "OIHW"}
        )
        
        ofmap_model = ofmap_def.create_model(
            tensor_dims=(1, 256, 222, 222),
            config={"layout": "NCHW"}
        )
        
        # Create kernel model
        kernel_model = KernelModel(
            interface_models=[ifmap_model, weights_model, ofmap_model],
            definition=kernel_def
        )
        
        # Verify tiling
        assert ifmap_model.block_dims == [(1, 32, 14, 14)]
        assert weights_model.block_dims == [(32, 32, 3, 3)]
        assert ofmap_model.block_dims == [(1, 32, 14, 14)]
        
        # Apply output-stationary parallelism
        kernel_model.apply_parallelism({"ofmap": 8})
        
        # Verify stream dimensions
        assert ofmap_model.stream_dims == (1, 8, 1, 1)
    
    def test_adaptive_tiling_with_config(self):
        """Test adaptive tiling based on runtime configuration"""
        # Define adaptive interface
        data_def = InterfaceDefinition(
            name="data",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("FP16"),
            block_dims_expr=adaptive_tiles("optimization_mode", default=[1, 16, 256])
        )
        
        # Test different optimization modes
        configs = [
            ({"optimization_mode": [1, 32, 512]}, "low_latency"),
            ({"optimization_mode": [1, 8, 128]}, "high_throughput"),
            ({}, "default")
        ]
        
        for config, mode in configs:
            model = data_def.create_model(
                tensor_dims=(4, 64, 1024),
                config=config
            )
            
            if mode == "low_latency":
                assert model.block_dims == [(1, 32, 512)]
            elif mode == "high_throughput":
                assert model.block_dims == [(1, 8, 128)]
            else:  # default
                assert model.block_dims == [(1, 16, 256)]
            
            # Apply parallelism based on mode
            if mode == "low_latency":
                model.ipar = 32  # High parallelism for low latency
            elif mode == "high_throughput":
                model.ipar = 8   # Moderate parallelism
            else:
                model.ipar = 16
            
            # Check performance characteristics
            metrics = model.calculate_performance_metrics()
            assert metrics["interface_parallelism"] == model.ipar
    
    def test_composite_tiling_strategy(self):
        """Test composite tiling with fallback strategies"""
        # Create composite strategy
        composite_func = composite_tiling(
            adaptive_tiles("custom_tiles"),
            parameterized_tiles("TILE_H", "TILE_W"),
            fixed_tiles(32, 32)
        )
        
        feature_def = InterfaceDefinition(
            name="features",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=composite_func
        )
        
        # Test 1: Config provides custom tiles
        model1 = feature_def.create_model(
            tensor_dims=(256, 256),
            config={"custom_tiles": [64, 64]}
        )
        assert model1.block_dims == [(64, 64)]
        
        # Test 2: No config, but params available
        model2 = feature_def.create_model(
            tensor_dims=(256, 256),
            parameter_binding={"TILE_H": 128, "TILE_W": 128}
        )
        assert model2.block_dims == [(128, 128)]
        
        # Test 3: No config or params, use fixed fallback
        model3 = feature_def.create_model(
            tensor_dims=(256, 256)
        )
        assert model3.block_dims == [(32, 32)]
    
    def test_memory_constrained_tiling(self):
        """Test memory-constrained tiling for large tensors"""
        # 512KB memory limit with FP32 elements
        memory_func = memory_constrained_tiles(
            memory_limit_bytes=512 * 1024,
            bytes_per_element=4
        )
        
        large_tensor_def = InterfaceDefinition(
            name="large_activation",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("FP32"),
            block_dims_expr=memory_func
        )
        
        # Very large tensor (1GB with FP32)
        model = large_tensor_def.create_model(
            tensor_dims=(512, 512, 1024)
        )
        
        # Calculate actual memory usage
        block_elements = 1
        for dim in model.block_dims[0]:
            block_elements *= dim
        block_memory = block_elements * 4  # 4 bytes per FP32
        
        # Should be within limit
        assert block_memory <= 512 * 1024
        
        # Apply parallelism to further reduce per-stream memory
        model.ipar = 16
        stream_elements = 1
        for dim in model.stream_dims:
            stream_elements *= dim
        assert stream_elements == 16  # Parallelism applied
    
    def test_tiling_config_serialization(self):
        """Test TilingConfig serialization and deserialization"""
        # Create a complex config
        config = TilingConfig(
            strategy=TilingStrategy.COMPOSITE,
            sub_strategies=[
                TilingConfig(
                    strategy=TilingStrategy.ADAPTIVE,
                    config_key="layer_tiles",
                    default_tiles=[16, 16, 64]
                ),
                TilingConfig(
                    strategy=TilingStrategy.CHANNEL_MAJOR,
                    channel_tile=32,
                    spatial_mode="tile"
                )
            ]
        )
        
        # Serialize to JSON
        json_str = config.to_json()
        
        # Deserialize
        config2 = TilingConfig.from_json(json_str)
        
        # Create interface with deserialized config
        tensor_def = InterfaceDefinition(
            name="tensor",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=config2.to_function()
        )
        
        # Test with layer_tiles config
        model = tensor_def.create_model(
            tensor_dims=(1, 64, 128, 128),
            config={"layer_tiles": [1, 32, 16, 16], "layout": "NCHW"}
        )
        assert model.block_dims == [(1, 32, 16, 16)]
    
    def test_csdf_with_phase_dependent_tiling(self):
        """Test CSDF kernel with phase-dependent tiling"""
        from brainsmith.core.dataflow.core.tiling_functions import phase_dependent_tiles
        
        # Define phase-dependent tiling
        phase_func = phase_dependent_tiles([
            [32, 64, 14, 14],  # Phase 0: standard conv
            [32, 64, 1, 1],    # Phase 1: 1x1 conv
            [32, 64, 7, 7],    # Phase 2: pooled conv
        ])
        
        conv_def = InterfaceDefinition(
            name="conv_csdf",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=phase_func
        )
        
        # Test each phase
        for phase in range(3):
            model = conv_def.create_model(
                tensor_dims=(128, 256, 56, 56),
                config={"csdf_phase": phase}
            )
            
            if phase == 0:
                assert model.block_dims == [(32, 64, 14, 14)]
            elif phase == 1:
                assert model.block_dims == [(32, 64, 1, 1)]
            else:
                assert model.block_dims == [(32, 64, 7, 7)]
    
    def test_error_handling_and_validation(self):
        """Test error handling for invalid configurations"""
        # Test 1: Mismatched parameter tiles
        with pytest.raises(KeyError):
            param_def = InterfaceDefinition(
                name="param_test",
                direction=InterfaceDirection.INPUT,
                dtype=DataType.from_string("INT8"),
                block_dims_expr=parameterized_tiles("MISSING_PARAM")
            )
            param_def.create_model((128,), parameter_binding={})
        
        # Test 2: Invalid dimension count
        with pytest.raises(ValueError):
            fixed_def = InterfaceDefinition(
                name="fixed_test",
                direction=InterfaceDirection.INPUT,
                dtype=DataType.from_string("INT8"),
                block_dims_expr=fixed_tiles(32, 64)  # 2 dims
            )
            fixed_def.create_model((128, 256, 512))  # 3 dims
        
        # Test 3: Invalid iPar value
        model = InterfaceDefinition(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8")
        ).create_model((128, 256))
        
        with pytest.raises(ValueError):
            model.ipar = 0
        
        with pytest.raises(ValueError):
            model.ipar = -1
    
    def test_performance_optimization_workflow(self):
        """Test complete performance optimization workflow"""
        # Start with basic kernel
        kernel_def = KernelDefinition(
            name="optimizable_kernel",
            interface_definitions=[
                InterfaceDefinition(
                    name="input",
                    direction=InterfaceDirection.INPUT,
                    dtype=DataType.from_string("INT8"),
                    block_dims_expr=adaptive_tiles("tile_config", default=[32, 32])
                ),
                InterfaceDefinition(
                    name="output",
                    direction=InterfaceDirection.OUTPUT,
                    dtype=DataType.from_string("INT32"),
                    block_dims_expr=adaptive_tiles("tile_config", default=[32, 32])
                )
            ]
        )
        
        # Test different tile configurations
        tile_configs = [
            [16, 16],  # Small tiles
            [32, 32],  # Medium tiles
            [64, 64],  # Large tiles
        ]
        
        best_throughput = 0
        best_config = None
        best_ipar = None
        
        for tiles in tile_configs:
            # Create models with config
            input_model = kernel_def.interface_definitions[0].create_model(
                tensor_dims=(256, 256),
                config={"tile_config": tiles}
            )
            output_model = kernel_def.interface_definitions[1].create_model(
                tensor_dims=(256, 256),
                config={"tile_config": tiles}
            )
            
            kernel_model = KernelModel(
                interface_models=[input_model, output_model],
                definition=kernel_def
            )
            
            # Try different parallelism levels
            for ipar in [1, 4, 8, 16]:
                try:
                    kernel_model.apply_parallelism({"output": ipar})
                    metrics = kernel_model.calculate_performance_metrics()
                    
                    # Simple throughput estimate
                    if "initiation_interval" in metrics and metrics["initiation_interval"] > 0:
                        throughput = metrics["total_bandwidth_mbps"] / metrics["initiation_interval"]
                        
                        if throughput > best_throughput:
                            best_throughput = throughput
                            best_config = tiles
                            best_ipar = ipar
                except Exception:
                    pass  # Skip invalid configurations
        
        # Verify we found an optimization
        assert best_config is not None
        assert best_ipar is not None
        assert best_throughput > 0


if __name__ == "__main__":
    pytest.main([__file__])