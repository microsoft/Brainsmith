############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Configuration schema and utilities for tiling functions"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Union
from enum import Enum
import json


class TilingStrategy(Enum):
    """Common tiling strategies"""
    FIXED = "fixed"
    ADAPTIVE = "adaptive"
    FULL_TENSOR = "full_tensor"
    PARAMETERIZED = "parameterized"
    CHANNEL_MAJOR = "channel_major"
    POWER_OF_TWO = "power_of_two"
    RATIO_BASED = "ratio_based"
    MEMORY_CONSTRAINED = "memory_constrained"
    PHASE_DEPENDENT = "phase_dependent"
    COMPOSITE = "composite"


@dataclass
class TilingConfig:
    """Configuration for tiling strategies
    
    This class provides a structured way to configure tiling functions
    with validation and serialization support.
    """
    
    strategy: TilingStrategy
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Common configuration options
    layout: Optional[str] = None  # ONNX layout hint
    tile_sizes: Optional[List[int]] = None  # Fixed tile sizes
    config_key: Optional[str] = None  # Key for adaptive tiling
    default_tiles: Optional[List[int]] = None  # Default tiles for adaptive
    param_names: Optional[List[str]] = None  # Parameter names for parameterized
    channel_tile: Optional[Union[int, str]] = None  # Channel tile size
    spatial_mode: Optional[str] = None  # Spatial tiling mode
    min_size: Optional[int] = None  # Minimum tile size
    max_size: Optional[int] = None  # Maximum tile size
    ratios: Optional[List[float]] = None  # Ratios for ratio-based
    memory_limit_bytes: Optional[int] = None  # Memory constraint
    bytes_per_element: Optional[int] = None  # Bytes per element
    phase_configs: Optional[List[List[Union[int, str]]]] = None  # Phase configs
    sub_strategies: Optional[List['TilingConfig']] = None  # For composite
    
    def validate(self) -> List[str]:
        """Validate configuration consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Strategy-specific validation
        if self.strategy == TilingStrategy.FIXED:
            if not self.tile_sizes:
                errors.append("FIXED strategy requires tile_sizes")
                
        elif self.strategy == TilingStrategy.ADAPTIVE:
            if not self.config_key:
                errors.append("ADAPTIVE strategy requires config_key")
                
        elif self.strategy == TilingStrategy.PARAMETERIZED:
            if not self.param_names:
                errors.append("PARAMETERIZED strategy requires param_names")
                
        elif self.strategy == TilingStrategy.CHANNEL_MAJOR:
            if self.channel_tile is None:
                errors.append("CHANNEL_MAJOR strategy requires channel_tile")
            if self.spatial_mode not in ["full", "tile", None]:
                errors.append("spatial_mode must be 'full' or 'tile'")
                
        elif self.strategy == TilingStrategy.POWER_OF_TWO:
            if self.min_size and self.max_size and self.min_size > self.max_size:
                errors.append("min_size must be <= max_size")
                
        elif self.strategy == TilingStrategy.RATIO_BASED:
            if not self.ratios:
                errors.append("RATIO_BASED strategy requires ratios")
            elif any(r <= 0 or r > 1 for r in self.ratios):
                errors.append("All ratios must be between 0 and 1")
                
        elif self.strategy == TilingStrategy.MEMORY_CONSTRAINED:
            if not self.memory_limit_bytes:
                errors.append("MEMORY_CONSTRAINED strategy requires memory_limit_bytes")
                
        elif self.strategy == TilingStrategy.PHASE_DEPENDENT:
            if not self.phase_configs:
                errors.append("PHASE_DEPENDENT strategy requires phase_configs")
                
        elif self.strategy == TilingStrategy.COMPOSITE:
            if not self.sub_strategies:
                errors.append("COMPOSITE strategy requires sub_strategies")
        
        return errors
    
    def to_function(self):
        """Convert configuration to a tiling function
        
        Returns:
            Callable tiling function
        """
        from .tiling_functions import (
            fixed_tiles, adaptive_tiles, full_tensor, parameterized_tiles,
            channel_major_tiling, power_of_two_tiles, ratio_based_tiles,
            memory_constrained_tiles, phase_dependent_tiles, composite_tiling
        )
        
        if self.strategy == TilingStrategy.FIXED:
            return fixed_tiles(*self.tile_sizes)
            
        elif self.strategy == TilingStrategy.ADAPTIVE:
            return adaptive_tiles(self.config_key, default=self.default_tiles)
            
        elif self.strategy == TilingStrategy.FULL_TENSOR:
            return full_tensor()
            
        elif self.strategy == TilingStrategy.PARAMETERIZED:
            return parameterized_tiles(*self.param_names)
            
        elif self.strategy == TilingStrategy.CHANNEL_MAJOR:
            return channel_major_tiling(
                channel_tile=self.channel_tile,
                spatial_mode=self.spatial_mode or "full"
            )
            
        elif self.strategy == TilingStrategy.POWER_OF_TWO:
            return power_of_two_tiles(
                min_size=self.min_size or 1,
                max_size=self.max_size or 1024
            )
            
        elif self.strategy == TilingStrategy.RATIO_BASED:
            return ratio_based_tiles(self.ratios)
            
        elif self.strategy == TilingStrategy.MEMORY_CONSTRAINED:
            return memory_constrained_tiles(
                memory_limit_bytes=self.memory_limit_bytes,
                bytes_per_element=self.bytes_per_element or 1
            )
            
        elif self.strategy == TilingStrategy.PHASE_DEPENDENT:
            return phase_dependent_tiles(self.phase_configs)
            
        elif self.strategy == TilingStrategy.COMPOSITE:
            sub_funcs = [sub.to_function() for sub in self.sub_strategies]
            return composite_tiling(*sub_funcs)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        result = {
            "strategy": self.strategy.value,
            "parameters": self.parameters
        }
        
        # Add non-None optional fields
        for field_name in ["layout", "tile_sizes", "config_key", "default_tiles",
                          "param_names", "channel_tile", "spatial_mode",
                          "min_size", "max_size", "ratios", "memory_limit_bytes",
                          "bytes_per_element", "phase_configs"]:
            value = getattr(self, field_name)
            if value is not None:
                result[field_name] = value
        
        # Handle sub_strategies specially
        if self.sub_strategies:
            result["sub_strategies"] = [s.to_dict() for s in self.sub_strategies]
        
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TilingConfig':
        """Create from dictionary"""
        # Convert strategy string to enum
        strategy = TilingStrategy(data["strategy"])
        
        # Handle sub_strategies specially
        sub_strategies = None
        if "sub_strategies" in data:
            sub_strategies = [cls.from_dict(sub) for sub in data["sub_strategies"]]
        
        # Create config
        config = cls(
            strategy=strategy,
            parameters=data.get("parameters", {}),
            sub_strategies=sub_strategies
        )
        
        # Set optional fields
        for field_name in ["layout", "tile_sizes", "config_key", "default_tiles",
                          "param_names", "channel_tile", "spatial_mode",
                          "min_size", "max_size", "ratios", "memory_limit_bytes",
                          "bytes_per_element", "phase_configs"]:
            if field_name in data:
                setattr(config, field_name, data[field_name])
        
        return config
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TilingConfig':
        """Create from JSON string"""
        return cls.from_dict(json.loads(json_str))


# Predefined configuration templates
TILING_TEMPLATES = {
    "conv2d_standard": TilingConfig(
        strategy=TilingStrategy.CHANNEL_MAJOR,
        channel_tile=32,
        spatial_mode="tile",
        layout="NCHW"
    ),
    
    "conv2d_depthwise": TilingConfig(
        strategy=TilingStrategy.CHANNEL_MAJOR,
        channel_tile=1,
        spatial_mode="full",
        layout="NCHW"
    ),
    
    "matmul_tiled": TilingConfig(
        strategy=TilingStrategy.PARAMETERIZED,
        param_names=["TILE_M", "TILE_K"]
    ),
    
    "transformer_attention": TilingConfig(
        strategy=TilingStrategy.ADAPTIVE,
        config_key="attention_tiles",
        default_tiles=[1, 8, 64]  # [batch, heads, seq_len]
    ),
    
    "memory_limited": TilingConfig(
        strategy=TilingStrategy.MEMORY_CONSTRAINED,
        memory_limit_bytes=1024 * 1024,  # 1MB
        bytes_per_element=2  # INT16
    ),
    
    "adaptive_conv": TilingConfig(
        strategy=TilingStrategy.COMPOSITE,
        sub_strategies=[
            TilingConfig(
                strategy=TilingStrategy.ADAPTIVE,
                config_key="custom_conv_tiles"
            ),
            TilingConfig(
                strategy=TilingStrategy.CHANNEL_MAJOR,
                channel_tile=16,
                spatial_mode="tile"
            )
        ]
    )
}


def get_tiling_template(name: str) -> TilingConfig:
    """Get a predefined tiling configuration template
    
    Args:
        name: Template name
        
    Returns:
        TilingConfig instance
        
    Raises:
        KeyError: If template not found
    """
    if name not in TILING_TEMPLATES:
        raise KeyError(f"Unknown template: {name}. Available: {list(TILING_TEMPLATES.keys())}")
    
    # Return a copy to avoid mutation
    template = TILING_TEMPLATES[name]
    return TilingConfig.from_dict(template.to_dict())


def create_tiling_config(strategy: Union[str, TilingStrategy], **kwargs) -> TilingConfig:
    """Convenience function to create tiling configuration
    
    Args:
        strategy: Strategy name or enum
        **kwargs: Strategy-specific parameters
        
    Returns:
        TilingConfig instance
        
    Example:
        config = create_tiling_config("fixed", tile_sizes=[32, 64, 16, 16])
    """
    if isinstance(strategy, str):
        strategy = TilingStrategy(strategy)
    
    config = TilingConfig(strategy=strategy)
    
    # Set provided parameters
    for key, value in kwargs.items():
        if hasattr(config, key):
            setattr(config, key, value)
        else:
            config.parameters[key] = value
    
    # Validate
    errors = config.validate()
    if errors:
        raise ValueError(f"Invalid configuration: {', '.join(errors)}")
    
    return config