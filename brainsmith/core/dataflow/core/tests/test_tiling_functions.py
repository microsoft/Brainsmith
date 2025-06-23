############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for tiling functions library"""

import pytest
from typing import Dict, Any

from brainsmith.core.dataflow.core.tiling_functions import (
    fixed_tiles, adaptive_tiles, full_tensor, parameterized_tiles,
    channel_major_tiling, power_of_two_tiles, ratio_based_tiles,
    memory_constrained_tiles, phase_dependent_tiles, composite_tiling,
    validate_tiling_function
)
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection


class TestTilingFunctions:
    """Test the tiling functions library"""
    
    def test_fixed_tiles(self):
        """Test fixed tile size function"""
        func = fixed_tiles(32, 64, 16, 16)
        
        tensor_dims = (128, 256, 64, 64)
        params = {}
        config = {}
        
        result = func(tensor_dims, params, config)
        assert result == [32, 64, 16, 16]
        
        # Test error on wrong dimensions
        with pytest.raises(ValueError, match="Expected 2 tile sizes, got 4"):
            func((128, 256), params, config)
    
    def test_adaptive_tiles(self):
        """Test configuration-based adaptive tiling"""
        func = adaptive_tiles("tile_config", default=[16, 32, 8, 8])
        
        tensor_dims = (64, 128, 32, 32)
        params = {}
        
        # Test with config
        config = {"tile_config": [32, 64, 16, 16]}
        result = func(tensor_dims, params, config)
        assert result == [32, 64, 16, 16]
        
        # Test without config - use default
        result = func(tensor_dims, params, {})
        assert result == [16, 32, 8, 8]
        
        # Test no config, no default - full tensor
        func_no_default = adaptive_tiles("missing_key")
        result = func_no_default(tensor_dims, params, {})
        assert result == [":", ":", ":", ":"]
    
    def test_full_tensor(self):
        """Test full tensor (no tiling) function"""
        func = full_tensor()
        
        tensor_dims = (128, 256, 64, 64)
        result = func(tensor_dims, {}, {})
        assert result == [":", ":", ":", ":"]
        
        # Works with any dimension
        result = func((10, 20), {}, {})
        assert result == [":", ":"]
    
    def test_parameterized_tiles(self):
        """Test parameter-based tiling"""
        func = parameterized_tiles("TILE_N", "TILE_C", "TILE_H", "TILE_W")
        
        tensor_dims = (128, 256, 64, 64)
        params = {
            "TILE_N": 1,
            "TILE_C": 32,
            "TILE_H": 16,
            "TILE_W": 16
        }
        
        result = func(tensor_dims, params, {})
        assert result == [1, 32, 16, 16]
        
        # Test missing parameter
        params_incomplete = {"TILE_N": 1, "TILE_C": 32}
        with pytest.raises(KeyError, match="TILE_H"):
            func(tensor_dims, params_incomplete, {})
    
    def test_channel_major_tiling(self):
        """Test channel-major tiling for CNNs"""
        func = channel_major_tiling(channel_tile=32, spatial_mode="tile")
        
        # Test NCHW
        tensor_dims = (8, 128, 224, 224)
        params = {}
        config = {"layout": "NCHW"}
        
        result = func(tensor_dims, params, config)
        assert result[0] == 1  # Batch not tiled
        assert result[1] == 32  # Channel tile
        assert result[2] == 14  # Spatial tile (capped at 14)
        assert result[3] == 14
        
        # Test NHWC
        config = {"layout": "NHWC"}
        result = func(tensor_dims, params, config)
        assert result == [1, 14, 14, 32]  # N, H, W, C
        
        # Test spatial_mode="full"
        func_full = channel_major_tiling(channel_tile=16, spatial_mode="full")
        result = func_full(tensor_dims, params, {"layout": "NCHW"})
        assert result == [1, 16, ":", ":"]
        
        # Test channel tile from parameter
        func_param = channel_major_tiling(channel_tile="C_TILE", spatial_mode="full")
        params = {"C_TILE": 64}
        result = func_param(tensor_dims, params, {"layout": "NCHW"})
        assert result == [1, 64, ":", ":"]
    
    def test_power_of_two_tiles(self):
        """Test power-of-two tile selection"""
        func = power_of_two_tiles(min_size=8, max_size=64)
        
        tensor_dims = (128, 256, 96, 48)
        result = func(tensor_dims, {}, {})
        
        # Should find largest power-of-2 divisors
        assert result[0] == 64  # 128 / 2 = 64 (max_size)
        assert result[1] == 64  # 256 / 4 = 64 (max_size)
        assert result[2] == 32  # 96 / 3 = 32 (largest power-of-2)
        assert result[3] == 16  # 48 / 3 = 16
        
        # All results should be powers of 2
        for tile in result:
            assert tile & (tile - 1) == 0  # Check power of 2
    
    def test_ratio_based_tiles(self):
        """Test ratio-based tiling"""
        func = ratio_based_tiles([1.0, 0.25, 0.1, 0.1])
        
        tensor_dims = (8, 256, 100, 100)
        result = func(tensor_dims, {}, {})
        
        assert result[0] == ":"  # 1.0 ratio = full
        assert result[1] == 64   # 0.25 * 256 = 64
        assert result[2] == 10   # 0.1 * 100 = 10
        assert result[3] == 10   # 0.1 * 100 = 10
        
        # Test invalid ratios
        with pytest.raises(ValueError, match="between 0 and 1"):
            bad_func = ratio_based_tiles([1.5, 0.5])
            bad_func((100, 100), {}, {})
    
    def test_memory_constrained_tiles(self):
        """Test memory-constrained tiling"""
        # 1MB limit, 4 bytes per element
        func = memory_constrained_tiles(memory_limit_bytes=1024*1024, bytes_per_element=4)
        
        # Large tensor that would exceed memory
        tensor_dims = (64, 512, 128, 128)  # 536M elements * 4 bytes = 2GB+
        result = func(tensor_dims, {}, {})
        
        # Calculate resulting memory usage
        total_elements = 1
        for tile in result:
            total_elements *= tile
        memory_usage = total_elements * 4
        
        assert memory_usage <= 1024 * 1024
        
        # All dimensions should be reduced
        assert all(result[i] <= tensor_dims[i] for i in range(len(result)))
    
    def test_phase_dependent_tiles(self):
        """Test CSDF phase-dependent tiling"""
        phase_configs = [
            [32, 64, 14, 14],  # Phase 0: standard conv
            [32, 64, 1, 1],    # Phase 1: 1x1 conv
            [32, 64, 7, 7]     # Phase 2: smaller conv
        ]
        
        func = phase_dependent_tiles(phase_configs)
        tensor_dims = (128, 256, 56, 56)
        params = {}
        
        # Test each phase
        for phase, expected in enumerate(phase_configs):
            config = {"csdf_phase": phase}
            result = func(tensor_dims, params, config)
            assert result == expected
        
        # Test invalid phase
        with pytest.raises(ValueError, match="No configuration for phase"):
            func(tensor_dims, params, {"csdf_phase": 3})
    
    def test_composite_tiling(self):
        """Test composite tiling strategies"""
        # Try adaptive first, then power-of-two, then full tensor
        func = composite_tiling(
            adaptive_tiles("custom_tiles"),
            power_of_two_tiles(min_size=8, max_size=64),
            full_tensor()
        )
        
        tensor_dims = (128, 256, 64, 64)
        params = {}
        
        # First strategy succeeds with config
        config = {"custom_tiles": [16, 32, 8, 8]}
        result = func(tensor_dims, params, config)
        assert result == [16, 32, 8, 8]
        
        # First strategy fails, second succeeds
        result = func(tensor_dims, params, {})
        # Should fall through to power_of_two_tiles since adaptive returns full tensor
        assert result == [64, 64, 64, 64]  # Power-of-two tiles with max_size=64
        
        # Test fallback when strategies return None or fail
        func_fallback = composite_tiling(
            lambda t, p, c: None,  # Returns None
            adaptive_tiles("missing_key"),  # Will return full tensor - this is the last strategy
        )
        result = func_fallback(tensor_dims, params, {})
        assert result == [":", ":", ":", ":"]  # Returns full tensor since it's the last strategy
    
    def test_validate_tiling_function(self):
        """Test tiling function validation"""
        tensor_dims = (128, 256, 64, 64)
        params = {"TILE_C": 32}
        config = {"layout": "NCHW"}
        
        # Valid function
        func = fixed_tiles(32, 64, 16, 16)
        assert validate_tiling_function(func, tensor_dims, params, config)
        
        # Invalid: wrong number of tiles
        bad_func = lambda t, p, c: [32, 64]  # Only 2 tiles for 4D tensor
        with pytest.raises(ValueError, match="returned 2 tiles for 4 dimensions"):
            validate_tiling_function(bad_func, tensor_dims)
        
        # Invalid: non-divisible tile
        bad_func2 = lambda t, p, c: [32, 64, 13, 16]  # 13 doesn't divide 64
        with pytest.raises(ValueError, match="does not divide tensor dimension"):
            validate_tiling_function(bad_func2, tensor_dims)
        
        # Invalid: returns wrong type
        bad_func3 = lambda t, p, c: "not a list"
        with pytest.raises(ValueError, match="must return list/tuple"):
            validate_tiling_function(bad_func3, tensor_dims)
    
    def test_integration_with_interface_definition(self):
        """Test tiling functions with InterfaceDefinition"""
        # Create interface with function-based block dims
        idef = InterfaceDefinition(
            name="conv_input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=channel_major_tiling(channel_tile=32, spatial_mode="tile"),
            onnx_layout="NCHW"
        )
        
        # Derive block dims
        tensor_dims = (1, 128, 224, 224)
        params = {}
        config = {"layout": "NCHW"}
        
        block_dims = idef.derive_block_dims(tensor_dims, params, config)
        assert block_dims == (1, 32, 14, 14)
        
        # Test with parameter-based function
        idef2 = InterfaceDefinition(
            name="matmul_input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("INT8"),
            block_dims_expr=parameterized_tiles("TILE_M", "TILE_K")
        )
        
        params = {"TILE_M": 64, "TILE_K": 128}
        block_dims = idef2.derive_block_dims((512, 1024), params, {})
        assert block_dims == (64, 128)
    
    def test_edge_cases(self):
        """Test edge cases and error handling"""
        # Empty tensor dimensions
        func = full_tensor()
        result = func((), {}, {})
        assert result == []
        
        # Single dimension
        func = fixed_tiles(32)
        result = func((128,), {}, {})
        assert result == [32]
        
        # Very large dimensions
        func = power_of_two_tiles(max_size=1024)
        result = func((1048576,), {}, {})  # 2^20
        assert result == [1024]  # Capped at max_size


if __name__ == "__main__":
    pytest.main([__file__])