############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Library of common tiling functions for block dimension specification"""

from typing import List, Dict, Any, Callable, Optional, Union
from .types import Shape


def fixed_tiles(*tile_sizes: int) -> Callable[[Shape, Dict[str, int], Optional[Dict[str, Any]]], Shape]:
    """Create a function that returns fixed tile sizes
    
    Args:
        *tile_sizes: Fixed size for each dimension
        
    Returns:
        Function that returns the fixed tile sizes
        
    Example:
        block_dims_expr = fixed_tiles(32, 64)  # Always returns (32, 64)
    """
    def _fixed(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        if len(tile_sizes) != len(tensor_dims):
            raise ValueError(f"Expected {len(tensor_dims)} tile sizes, got {len(tile_sizes)}")
        return list(tile_sizes)
    
    return _fixed


def adaptive_tiles(config_key: str, default: Optional[List[int]] = None) -> Callable:
    """Create a function that adapts tile sizes based on configuration
    
    Args:
        config_key: Key in config dict containing tile sizes
        default: Default tile sizes if config key not found
        
    Returns:
        Function that returns config-based tile sizes
        
    Example:
        block_dims_expr = adaptive_tiles("conv_tiles", default=[1, 16, 14, 14])
    """
    def _adaptive(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        if config and config_key in config:
            tiles = config[config_key]
            if isinstance(tiles, (list, tuple)) and len(tiles) == len(tensor_dims):
                return list(tiles)
        
        if default is not None:
            if len(default) != len(tensor_dims):
                raise ValueError(f"Default tiles length {len(default)} doesn't match tensor dims {len(tensor_dims)}")
            return default
        
        # No config and no default - use full tensor
        return [":"] * len(tensor_dims)
    
    return _adaptive


def full_tensor() -> Callable[[Shape, Dict[str, int], Optional[Dict[str, Any]]], Shape]:
    """Create a function that uses full tensor dimensions (no tiling)
    
    Returns:
        Function that returns [":"] for all dimensions
        
    Example:
        block_dims_expr = full_tensor()  # No blocking
    """
    def _full(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        return [":"] * len(tensor_dims)
    
    return _full


def adaptive_parameterized_tiles(*param_names: str) -> Callable:
    """Create a function that uses parameter values for tile sizes with dynamic singleton handling
    
    Supports automatic left-padding with singletons for right-justification when tensor
    dimensions exceed parameter count. User can specify right-side singletons explicitly.
    
    Args:
        *param_names: Parameter names to look up for each dimension
        
    Returns:
        Function that returns parameter-based tile sizes with singleton padding
        
    Examples:
        # System left-padding for under-specified dimensions
        block_dims_expr = adaptive_parameterized_tiles("input_BDIM")
        # TDIM=[1, 64] → BDIM=[1, input_BDIM] (left-padded)
        
        # User right-singletons + system left-padding
        block_dims_expr = adaptive_parameterized_tiles("TILE_H", "TILE_W", "1")
        # TDIM=[32, 14, 14, 3] → BDIM=[1, TILE_H, TILE_W, 1]
    """
    def _adaptive_parameterized(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        # Validate: tensor must have at least as many dims as parameters
        if len(tensor_dims) < len(param_names):
            raise ValueError(
                f"Insufficient tensor dimensions: tensor has {len(tensor_dims)} dimensions "
                f"but {len(param_names)} BDIM parameters specified. "
                f"Tensor dimensions must be >= BDIM parameters for singleton padding."
            )
        
        # Calculate left-padding needed for right-justification
        padding_count = len(tensor_dims) - len(param_names)
        
        # Create padded parameter list (left-pad with singleton 1's)
        padded_params = ['1'] * padding_count + list(param_names)
        
        # Resolve parameter values
        tiles = []
        for param in padded_params:
            if param == '1':
                tiles.append(1)  # Singleton dimension (both system and user-specified)
            elif param in params:
                tiles.append(params[param])
            else:
                raise KeyError(f"Parameter '{param}' not found in parameter binding")
        
        return tiles
    
    return _adaptive_parameterized


def parameterized_tiles(*param_names: str) -> Callable:
    """Create a function that uses parameter values for tile sizes
    
    DEPRECATED: Use adaptive_parameterized_tiles() for automatic singleton handling.
    This function requires exact parameter count matching and lacks singleton support.
    
    Args:
        *param_names: Parameter names to look up for each dimension
        
    Returns:
        Function that returns parameter-based tile sizes
        
    Example:
        block_dims_expr = parameterized_tiles("TILE_N", "TILE_C", "TILE_H", "TILE_W")
    """
    import warnings
    warnings.warn(
        "parameterized_tiles() is deprecated and lacks singleton handling. "
        "Use adaptive_parameterized_tiles() for robust dimension handling.",
        DeprecationWarning,
        stacklevel=2
    )
    
    def _parameterized(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        if len(param_names) != len(tensor_dims):
            raise ValueError(f"Expected {len(tensor_dims)} parameter names, got {len(param_names)}")
        
        tiles = []
        for i, param_name in enumerate(param_names):
            if param_name in params:
                tiles.append(params[param_name])
            else:
                raise KeyError(f"Parameter '{param_name}' not found in parameter binding")
        
        return tiles
    
    return _parameterized


