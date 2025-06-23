############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unit tests for tiling configuration system"""

import pytest
import json
from typing import Dict, Any

from brainsmith.core.dataflow.core.tiling_config import (
    TilingConfig, TilingStrategy, TILING_TEMPLATES,
    get_tiling_template, create_tiling_config
)
from brainsmith.core.dataflow.core.interface_definition import InterfaceDefinition
from brainsmith.core.dataflow.core.types import DataType, InterfaceDirection


class TestTilingConfig:
    """Test tiling configuration system"""
    
    def test_tiling_config_validation(self):
        """Test configuration validation"""
        # Valid fixed config
        config = TilingConfig(
            strategy=TilingStrategy.FIXED,
            tile_sizes=[32, 64, 16, 16]
        )
        errors = config.validate()
        assert len(errors) == 0
        
        # Invalid fixed config - missing tile_sizes
        config = TilingConfig(strategy=TilingStrategy.FIXED)
        errors = config.validate()
        assert len(errors) == 1
        assert "tile_sizes" in errors[0]
        
        # Invalid ratio config - ratios out of range
        config = TilingConfig(
            strategy=TilingStrategy.RATIO_BASED,
            ratios=[0.5, 1.5, 0.1]  # 1.5 is invalid
        )
        errors = config.validate()
        assert len(errors) == 1
        assert "between 0 and 1" in errors[0]
    
    def test_to_function_conversion(self):
        """Test converting config to function"""
        # Fixed tiles
        config = TilingConfig(
            strategy=TilingStrategy.FIXED,
            tile_sizes=[32, 64]
        )
        func = config.to_function()
        result = func((128, 256), {}, {})
        assert result == [32, 64]
        
        # Channel major
        config = TilingConfig(
            strategy=TilingStrategy.CHANNEL_MAJOR,
            channel_tile=16,
            spatial_mode="full",
            layout="NCHW"
        )
        func = config.to_function()
        result = func((1, 64, 224, 224), {}, {"layout": "NCHW"})
        assert result == [1, 16, ":", ":"]
        
        # Parameterized
        config = TilingConfig(
            strategy=TilingStrategy.PARAMETERIZED,
            param_names=["M", "K"]
        )
        func = config.to_function()
        result = func((512, 1024), {"M": 64, "K": 128}, {})
        assert result == [64, 128]
    
    def test_serialization(self):
        """Test to_dict and from_dict"""
        # Complex config with sub-strategies
        config = TilingConfig(
            strategy=TilingStrategy.COMPOSITE,
            parameters={"custom": "value"},
            sub_strategies=[
                TilingConfig(
                    strategy=TilingStrategy.ADAPTIVE,
                    config_key="tiles",
                    default_tiles=[16, 32]
                ),
                TilingConfig(
                    strategy=TilingStrategy.FULL_TENSOR
                )
            ]
        )
        
        # Convert to dict
        config_dict = config.to_dict()
        assert config_dict["strategy"] == "composite"
        assert config_dict["parameters"]["custom"] == "value"
        assert len(config_dict["sub_strategies"]) == 2
        assert config_dict["sub_strategies"][0]["strategy"] == "adaptive"
        
        # Convert back
        config2 = TilingConfig.from_dict(config_dict)
        assert config2.strategy == TilingStrategy.COMPOSITE
        assert len(config2.sub_strategies) == 2
        assert config2.sub_strategies[0].config_key == "tiles"
    
    def test_json_serialization(self):
        """Test JSON serialization"""
        config = TilingConfig(
            strategy=TilingStrategy.MEMORY_CONSTRAINED,
            memory_limit_bytes=2048 * 1024,
            bytes_per_element=4
        )
        
        # To JSON
        json_str = config.to_json()
        data = json.loads(json_str)
        assert data["strategy"] == "memory_constrained"
        assert data["memory_limit_bytes"] == 2048 * 1024
        
        # From JSON
        config2 = TilingConfig.from_json(json_str)
        assert config2.strategy == TilingStrategy.MEMORY_CONSTRAINED
        assert config2.memory_limit_bytes == 2048 * 1024
        assert config2.bytes_per_element == 4
    
    def test_templates(self):
        """Test predefined templates"""
        # Get conv2d template
        config = get_tiling_template("conv2d_standard")
        assert config.strategy == TilingStrategy.CHANNEL_MAJOR
        assert config.channel_tile == 32
        assert config.spatial_mode == "tile"
        
        # Get transformer template
        config = get_tiling_template("transformer_attention")
        assert config.strategy == TilingStrategy.ADAPTIVE
        assert config.default_tiles == [1, 8, 64]
        
        # Unknown template
        with pytest.raises(KeyError, match="Unknown template"):
            get_tiling_template("unknown")
        
        # Verify template is a copy
        config1 = get_tiling_template("conv2d_standard")
        config2 = get_tiling_template("conv2d_standard")
        config1.channel_tile = 64
        assert config2.channel_tile == 32  # Not modified
    
    def test_create_tiling_config(self):
        """Test convenience function"""
        # Create with string strategy
        config = create_tiling_config("fixed", tile_sizes=[16, 32, 8, 8])
        assert config.strategy == TilingStrategy.FIXED
        assert config.tile_sizes == [16, 32, 8, 8]
        
        # Create with enum strategy
        config = create_tiling_config(
            TilingStrategy.POWER_OF_TWO,
            min_size=8,
            max_size=64
        )
        assert config.min_size == 8
        assert config.max_size == 64
        
        # Invalid config raises
        with pytest.raises(ValueError, match="Invalid configuration"):
            create_tiling_config("fixed")  # Missing tile_sizes
    
    def test_integration_with_interface(self):
        """Test using config with InterfaceDefinition"""
        # Create config
        config = TilingConfig(
            strategy=TilingStrategy.CHANNEL_MAJOR,
            channel_tile="C_TILE",  # Parameter reference
            spatial_mode="tile"
        )
        
        # Use with interface
        idef = InterfaceDefinition(
            name="conv_input",
            direction=InterfaceDirection.INPUT,
            dtype=DataType.from_string("UINT8"),
            block_dims_expr=config.to_function()
        )
        
        # Derive block dims
        tensor_dims = (1, 128, 224, 224)
        params = {"C_TILE": 32}
        config_dict = {"layout": "NCHW"}
        
        block_dims = idef.derive_block_dims(tensor_dims, params, config_dict)
        assert block_dims == (1, 32, 14, 14)
    
    def test_phase_dependent_config(self):
        """Test phase-dependent configuration"""
        config = TilingConfig(
            strategy=TilingStrategy.PHASE_DEPENDENT,
            phase_configs=[
                [32, 64, 14, 14],  # Phase 0
                [32, 64, 1, 1],    # Phase 1
            ]
        )
        
        func = config.to_function()
        
        # Test phase 0
        result = func((128, 256, 56, 56), {}, {"csdf_phase": 0})
        assert result == [32, 64, 14, 14]
        
        # Test phase 1
        result = func((128, 256, 56, 56), {}, {"csdf_phase": 1})
        assert result == [32, 64, 1, 1]
    
    def test_all_strategies_covered(self):
        """Ensure all strategies can be configured"""
        for strategy in TilingStrategy:
            # Skip strategies that need specific config
            if strategy in [TilingStrategy.FIXED, TilingStrategy.ADAPTIVE,
                          TilingStrategy.PARAMETERIZED, TilingStrategy.RATIO_BASED,
                          TilingStrategy.PHASE_DEPENDENT, TilingStrategy.COMPOSITE]:
                continue
            
            # Should be able to create basic config
            config = TilingConfig(strategy=strategy)
            if strategy == TilingStrategy.CHANNEL_MAJOR:
                config.channel_tile = 16
            elif strategy == TilingStrategy.MEMORY_CONSTRAINED:
                config.memory_limit_bytes = 1024
            
            # Should convert to function without error
            func = config.to_function()
            assert callable(func)


if __name__ == "__main__":
    pytest.main([__file__])