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


def parameterized_tiles(*param_names: str) -> Callable:
    """Create a function that uses parameter values for tile sizes
    
    Args:
        *param_names: Parameter names to look up for each dimension
        
    Returns:
        Function that returns parameter-based tile sizes
        
    Example:
        block_dims_expr = parameterized_tiles("TILE_N", "TILE_C", "TILE_H", "TILE_W")
    """
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


def channel_major_tiling(channel_tile: Union[int, str] = 16, 
                        spatial_mode: str = "full") -> Callable:
    """Create a function for channel-major tiling (common in CNNs)
    
    Args:
        channel_tile: Tile size for channel dimension (or param name)
        spatial_mode: "full" for no spatial tiling, "tile" for default tiling
        
    Returns:
        Function for NCHW or NHWC layouts
        
    Example:
        block_dims_expr = channel_major_tiling(channel_tile=32, spatial_mode="tile")
    """
    def _channel_major(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        if len(tensor_dims) != 4:
            raise ValueError(f"Channel-major tiling expects 4D tensor, got {len(tensor_dims)}D")
        
        # Determine layout from config or ONNX hint
        layout = "NCHW"  # Default
        if config and "layout" in config:
            layout = config["layout"]
        
        # Get channel tile size
        if isinstance(channel_tile, str):
            if channel_tile in params:
                c_tile = params[channel_tile]
            else:
                raise KeyError(f"Parameter '{channel_tile}' not found")
        else:
            c_tile = channel_tile
        
        # Build tile sizes based on layout
        if layout == "NCHW":
            n_tile = 1  # Batch usually not tiled
            h_tile = ":" if spatial_mode == "full" else min(14, tensor_dims[2])
            w_tile = ":" if spatial_mode == "full" else min(14, tensor_dims[3])
            return [n_tile, c_tile, h_tile, w_tile]
        elif layout == "NHWC":
            n_tile = 1
            h_tile = ":" if spatial_mode == "full" else min(14, tensor_dims[1])
            w_tile = ":" if spatial_mode == "full" else min(14, tensor_dims[2])
            return [n_tile, h_tile, w_tile, c_tile]
        elif layout == "OIHW":
            # Output channels, Input channels, Height, Width (for conv weights)
            o_tile = c_tile  # Output channels use channel tile
            i_tile = c_tile  # Input channels also tiled
            h_tile = ":" if spatial_mode == "full" else min(3, tensor_dims[2])  # Small for kernels
            w_tile = ":" if spatial_mode == "full" else min(3, tensor_dims[3])
            return [o_tile, i_tile, h_tile, w_tile]
        else:
            raise ValueError(f"Unknown layout: {layout}")
    
    return _channel_major


def power_of_two_tiles(min_size: int = 1, max_size: int = 1024) -> Callable:
    """Create a function that chooses power-of-two tile sizes
    
    Args:
        min_size: Minimum tile size
        max_size: Maximum tile size
        
    Returns:
        Function that returns largest power-of-two tiles that fit
        
    Example:
        block_dims_expr = power_of_two_tiles(min_size=8, max_size=256)
    """
    def _power_of_two(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        tiles = []
        for dim in tensor_dims:
            # Find largest power of 2 that divides dim and is within bounds
            tile = min_size
            while tile * 2 <= min(dim, max_size) and dim % (tile * 2) == 0:
                tile *= 2
            tiles.append(tile)
        return tiles
    
    return _power_of_two


def ratio_based_tiles(ratios: List[float]) -> Callable:
    """Create a function that tiles based on ratios of tensor dimensions
    
    Args:
        ratios: Ratio for each dimension (0.0 to 1.0)
        
    Returns:
        Function that returns ratio-based tile sizes
        
    Example:
        block_dims_expr = ratio_based_tiles([1.0, 0.25, 0.1, 0.1])  # Full N, 1/4 C, 1/10 H&W
    """
    def _ratio_based(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        if len(ratios) != len(tensor_dims):
            raise ValueError(f"Expected {len(tensor_dims)} ratios, got {len(ratios)}")
        
        tiles = []
        for dim, ratio in zip(tensor_dims, ratios):
            if ratio <= 0 or ratio > 1:
                raise ValueError(f"Ratio must be between 0 and 1, got {ratio}")
            
            if ratio == 1.0:
                tiles.append(":")
            else:
                # Round to nearest divisor
                target = int(dim * ratio)
                # Find closest divisor
                best_tile = 1
                for t in range(max(1, target - 10), min(dim + 1, target + 10)):
                    if dim % t == 0 and abs(t - target) < abs(best_tile - target):
                        best_tile = t
                tiles.append(best_tile)
        
        return tiles
    
    return _ratio_based


def memory_constrained_tiles(memory_limit_bytes: int, 
                           bytes_per_element: int = 1) -> Callable:
    """Create a function that tiles to fit within memory constraints
    
    Args:
        memory_limit_bytes: Maximum memory for a tile
        bytes_per_element: Bytes per tensor element
        
    Returns:
        Function that returns memory-constrained tile sizes
        
    Example:
        block_dims_expr = memory_constrained_tiles(memory_limit_bytes=1024*1024, bytes_per_element=4)
    """
    def _memory_constrained(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        # Start with full tensor
        tiles = list(tensor_dims)
        
        # Iteratively reduce largest dimension until within memory
        while True:
            # Calculate current memory usage
            tile_elements = 1
            for t in tiles:
                if isinstance(t, int):
                    tile_elements *= t
                else:  # ":" means full dimension
                    idx = tiles.index(t)
                    tile_elements *= tensor_dims[idx]
            
            memory_usage = tile_elements * bytes_per_element
            
            if memory_usage <= memory_limit_bytes:
                break
            
            # Find largest tile dimension to reduce
            max_idx = -1
            max_size = 0
            for i, (tile, tensor_dim) in enumerate(zip(tiles, tensor_dims)):
                size = tile if isinstance(tile, int) else tensor_dim
                if size > max_size and size > 1:
                    max_size = size
                    max_idx = i
            
            if max_idx == -1:
                # Can't reduce further
                break
            
            # Halve the largest dimension
            current = tiles[max_idx] if isinstance(tiles[max_idx], int) else tensor_dims[max_idx]
            tiles[max_idx] = current // 2
        
        return tiles
    
    return _memory_constrained


def phase_dependent_tiles(phase_configs: List[List[Union[int, str]]]) -> Callable:
    """Create a function that returns different tiles per CSDF phase
    
    Args:
        phase_configs: List of tile configurations per phase
        
    Returns:
        Function that returns phase-dependent tiles
        
    Example:
        block_dims_expr = phase_dependent_tiles([
            [32, 64, 14, 14],  # Phase 0: standard conv
            [32, 64, 1, 1],    # Phase 1: 1x1 conv
        ])
    """
    def _phase_dependent(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> List[Shape]:
        # Get current phase from config if available
        phase = 0
        if config and "csdf_phase" in config:
            phase = config["csdf_phase"]
        
        if phase >= len(phase_configs):
            raise ValueError(f"No configuration for phase {phase}")
        
        phase_tiles = phase_configs[phase]
        if len(phase_tiles) != len(tensor_dims):
            raise ValueError(f"Phase {phase} tiles length {len(phase_tiles)} doesn't match tensor dims {len(tensor_dims)}")
        
        # Resolve any string parameters
        resolved_tiles = []
        for tile in phase_tiles:
            if isinstance(tile, str) and tile in params:
                resolved_tiles.append(params[tile])
            else:
                resolved_tiles.append(tile)
        
        return resolved_tiles
    
    return _phase_dependent


def composite_tiling(*strategies: Callable) -> Callable:
    """Compose multiple tiling strategies based on conditions
    
    Args:
        *strategies: Functions that return tiles or None
        
    Returns:
        Function that tries strategies in order until one succeeds
        
    Example:
        block_dims_expr = composite_tiling(
            adaptive_tiles("custom_tiles"),
            power_of_two_tiles(),
            full_tensor()
        )
    """
    def _composite(tensor_dims: Shape, params: Dict[str, int], config: Optional[Dict[str, Any]] = None) -> Shape:
        for strategy in strategies:
            try:
                result = strategy(tensor_dims, params, config)
                if result is not None:
                    # Check if result is full tensor (all dimensions are ":")
                    # If so, continue to next strategy unless it's the last one
                    if all(dim == ":" for dim in result) and strategy != strategies[-1]:
                        continue
                    return result
            except (KeyError, ValueError):
                # Try next strategy
                continue
        
        # All strategies failed, use full tensor as fallback
        return [":"] * len(tensor_dims)
    
    return _composite


# Helper function to validate tiling function
def validate_tiling_function(func: Callable, tensor_dims: Shape, 
                           params: Dict[str, int] = None,
                           config: Dict[str, Any] = None) -> bool:
    """Validate that a tiling function produces valid output
    
    Args:
        func: Tiling function to validate
        tensor_dims: Tensor dimensions to test with
        params: Parameters to pass to function
        config: Configuration to pass to function
        
    Returns:
        True if function produces valid tiles
        
    Raises:
        ValueError: If function produces invalid tiles
    """
    if params is None:
        params = {}
    
    try:
        tiles = func(tensor_dims, params, config)
    except Exception as e:
        raise ValueError(f"Tiling function raised exception: {e}")
    
    if not isinstance(tiles, (list, tuple)):
        raise ValueError(f"Tiling function must return list/tuple, got {type(tiles)}")
    
    if len(tiles) != len(tensor_dims):
        raise ValueError(f"Tiling function returned {len(tiles)} tiles for {len(tensor_dims)} dimensions")
    
    # Validate each tile
    for i, (tile, tensor_dim) in enumerate(zip(tiles, tensor_dims)):
        if tile == ":":
            continue  # Full dimension is always valid
        
        if not isinstance(tile, int) or tile <= 0:
            raise ValueError(f"Tile at index {i} must be positive integer or ':', got {tile}")
        
        if tensor_dim % tile != 0:
            raise ValueError(f"Tile {tile} at index {i} does not divide tensor dimension {tensor_dim}")
    
    return True