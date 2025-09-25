############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
Shape manipulation utilities for dataflow kernels.

This module provides utilities for working with tensor shapes, including
folding/unfolding, bandwidth calculations, and shape transformations.
"""

from typing import List, Tuple, Union


def create_folded_shape(
    tensor_dims: Tuple[int, ...],
    block_dims: Tuple[int, ...]
) -> List[int]:
    """Create folded shape from tensor and block dimensions.
    
    When a dimension is folded (block_dim < tensor_dim), it's split into
    two dimensions: [num_blocks, block_size].
    
    Args:
        tensor_dims: Full tensor dimensions
        block_dims: Block tiling dimensions
        
    Returns:
        List of folded dimensions
        
    Example:
        >>> create_folded_shape((32, 64), (8, 64))
        [4, 8, 64]  # First dim folded: 32/8 = 4 blocks of size 8
    """
    folded = []
    for tensor_dim, block_dim in zip(tensor_dims, block_dims):
        if block_dim < tensor_dim:
            # Dimension is folded
            num_blocks = tensor_dim // block_dim
            folded.extend([num_blocks, block_dim])
        else:
            # Dimension is not folded
            folded.append(tensor_dim)
    
    return folded


def unfold_shape(
    folded_shape: List[int],
    block_dims: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Reconstruct original tensor shape from folded shape.
    
    Args:
        folded_shape: Folded shape with potentially split dimensions
        block_dims: Block dimensions used for folding
        
    Returns:
        Original tensor dimensions
    """
    tensor_dims = []
    folded_idx = 0
    
    for block_dim in block_dims:
        if folded_idx + 1 < len(folded_shape) and folded_shape[folded_idx + 1] == block_dim:
            # This was a folded dimension
            num_blocks = folded_shape[folded_idx]
            tensor_dims.append(num_blocks * block_dim)
            folded_idx += 2
        else:
            # This was not folded
            tensor_dims.append(folded_shape[folded_idx])
            folded_idx += 1
    
    return tuple(tensor_dims)


def calculate_stream_width(
    streaming_bandwidth: int,
    datatype_bitwidth: int
) -> int:
    """Calculate stream width in bits.
    
    Args:
        streaming_bandwidth: Elements per cycle
        datatype_bitwidth: Bits per element
        
    Returns:
        Total stream width in bits
    """
    return streaming_bandwidth * datatype_bitwidth


def calculate_bandwidth_mbps(
    bandwidth_bits: int,
    clock_freq_mhz: float
) -> float:
    """Convert bandwidth from bits/cycle to MB/s.
    
    Args:
        bandwidth_bits: Bandwidth in bits per cycle
        clock_freq_mhz: Clock frequency in MHz
        
    Returns:
        Bandwidth in MB/s
    """
    return bandwidth_bits * clock_freq_mhz / 8.0


def calculate_bandwidth_gbps(
    bandwidth_bits: int,
    clock_freq_mhz: float
) -> float:
    """Convert bandwidth from bits/cycle to GB/s.
    
    Args:
        bandwidth_bits: Bandwidth in bits per cycle
        clock_freq_mhz: Clock frequency in MHz
        
    Returns:
        Bandwidth in GB/s
    """
    return bandwidth_bits * clock_freq_mhz / 8000.0


def is_compatible_folding(
    tensor_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...],
    stream_shape: Tuple[int, ...]
) -> bool:
    """Check if stream shape is compatible with block shape.
    
    Stream dimensions must evenly divide block dimensions.
    
    Args:
        tensor_shape: Full tensor shape
        block_shape: Block tiling shape
        stream_shape: Streaming shape
        
    Returns:
        True if compatible, False otherwise
    """
    if len(block_shape) != len(stream_shape):
        return False
    
    for block_dim, stream_dim in zip(block_shape, stream_shape):
        if stream_dim <= 0 or stream_dim > block_dim:
            return False
        if block_dim % stream_dim != 0:
            return False
    
    return True


def calculate_folding_factors(
    tensor_shape: Tuple[int, ...],
    block_shape: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Calculate folding factors for each dimension.
    
    Folding factor is the number of blocks needed to cover
    the tensor dimension.
    
    Args:
        tensor_shape: Full tensor shape
        block_shape: Block tiling shape
        
    Returns:
        Tuple of folding factors
    """
    import math
    
    factors = []
    for tensor_dim, block_dim in zip(tensor_shape, block_shape):
        factor = math.ceil(tensor_dim / block_dim)
        factors.append(factor)
    
    return tuple(factors)


def merge_shapes(
    shape1: Tuple[int, ...],
    shape2: Tuple[int, ...]
) -> Tuple[int, ...]:
    """Merge two shapes by taking the maximum of each dimension.
    
    Useful for computing output shapes that depend on multiple inputs.
    
    Args:
        shape1: First shape
        shape2: Second shape
        
    Returns:
        Merged shape
        
    Raises:
        ValueError: If shapes have different lengths
    """
    if len(shape1) != len(shape2):
        raise ValueError(
            f"Cannot merge shapes with different lengths: "
            f"{len(shape1)} vs {len(shape2)}"
        )
    
    return tuple(max(d1, d2) for d1, d2 in zip(shape1, shape2))


def compute_streaming_cycles(
    tensor_dims: Tuple[int, ...],
    block_dims: Tuple[int, ...],
    stream_dims: Tuple[int, ...]
) -> int:
    """Compute total cycles to stream a tensor.
    
    Args:
        tensor_dims: Full tensor dimensions
        block_dims: Block tiling dimensions
        stream_dims: Streaming dimensions
        
    Returns:
        Total number of cycles
    """
    import math
    
    # Calculate number of blocks
    total_blocks = 1
    for t, b in zip(tensor_dims, block_dims):
        total_blocks *= math.ceil(t / b)
    
    # Calculate cycles per block
    cycles_per_block = 1
    for b, s in zip(block_dims, stream_dims):
        cycles_per_block *= math.ceil(b / s)
    
    return total_blocks * cycles_per_block


def format_bandwidth(bandwidth_mbps: float) -> str:
    """Format bandwidth value with appropriate units.
    
    Args:
        bandwidth_mbps: Bandwidth in MB/s
        
    Returns:
        Formatted string with units (MB/s, GB/s, or TB/s)
    """
    if bandwidth_mbps < 1000:
        return f"{bandwidth_mbps:.1f} MB/s"
    elif bandwidth_mbps < 1000000:
        return f"{bandwidth_mbps/1000:.1f} GB/s"
    else:
        return f"{bandwidth_mbps/1000000:.1f} TB/s"