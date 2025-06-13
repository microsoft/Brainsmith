"""
Simplified block chunking system for interface-wise processing.

This module provides a unified block chunking strategy that chunks tensors
left-to-right from a configurable starting position (RINDEX).
"""

from dataclasses import dataclass
from typing import List, Tuple, Union
from enum import Enum


class ChunkingType(Enum):
    """Types of block chunking strategies."""
    BLOCK = "block"              # Standard block chunking with shape specification
    DEFAULT = "default"          # Layout-aware default chunking (legacy compatibility)


@dataclass
class BlockChunkingStrategy:
    """
    Simplified block chunking strategy.
    
    Always chunks left to right from a starting position.
    Default: rightmost dimensions (rindex=0)
    
    Attributes:
        block_shape: Block dimensions - can be:
            - ":" for full dimension  
            - String for parameter name (NO magic numbers allowed)
        rindex: Right-to-left starting index (0 = rightmost)
    """
    block_shape: List[str]  # Only strings - either ":" or parameter names
    rindex: int = 0
    
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """
        Compute tensor_dims and block_dims for the given tensor shape.
        
        Examples:
            tensor_shape=[A,B,C,D], block_shape=[:,:], rindex=0 -> block=[C,D]
            tensor_shape=[A,B,C,D], block_shape=[:,:], rindex=1 -> block=[B,C]
            tensor_shape=[A,B,C,D], block_shape=[PE], rindex=0 -> block=[PE] at D
            
        Args:
            tensor_shape: Input tensor shape
            interface_name: Interface name (for error context)
            
        Returns:
            Tuple of (tensor_dims, block_dims)
            - tensor_dims: Original tensor dimensions
            - block_dims: Block dimensions for processing
        """
        if not tensor_shape:
            return [1], [1]
        
        # tensor_dims is always the original shape
        tensor_dims = list(tensor_shape)
        
        # Start with full tensor as block
        block_dims = list(tensor_shape)
        
        # Calculate starting position from right
        # For shape [A,B,C,D] with rindex=0 and block_shape of length 2:
        # start_pos = 4 - 2 - 0 = 2 (position C)
        start_pos = len(tensor_shape) - len(self.block_shape) - self.rindex
        
        if start_pos < 0:
            raise ValueError(
                f"RINDEX {self.rindex} with block_shape length {len(self.block_shape)} "
                f"exceeds tensor dimensions {len(tensor_shape)} for interface '{interface_name}'"
            )
        
        # Apply block shape from starting position
        for i, shape_val in enumerate(self.block_shape):
            pos = start_pos + i
            if pos >= len(tensor_shape):
                break
                
            if shape_val == ":":
                # Keep full dimension
                block_dims[pos] = tensor_shape[pos]
            elif isinstance(shape_val, str):
                # Parameter name - keep as string for runtime resolution
                # This will be resolved by the HW kernel generator at template generation time
                block_dims[pos] = shape_val
            elif isinstance(shape_val, int):
                # Integer value - allow for backward compatibility
                block_dims[pos] = shape_val
            else:
                raise ValueError(
                    f"Invalid block shape element: {shape_val}. "
                    f"Must be ':' (full dimension), parameter name string, or integer."
                )
        
        return tensor_dims, block_dims
    
    @property
    def chunking_type(self) -> ChunkingType:
        return ChunkingType.BLOCK


@dataclass  
class DefaultChunkingStrategy:
    """
    Default layout-aware chunking strategy.
    
    Provides backward compatibility and smart defaults based on tensor layout.
    This is used when no BDIM pragma is specified.
    """
    
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """Apply layout-aware default chunking based on tensor shape."""
        if not tensor_shape:
            return [1], [1]
            
        # For now, default to full tensor (no chunking)
        # In practice, this would be overridden by interface type-specific logic
        tensor_dims = list(tensor_shape)
        block_dims = list(tensor_shape)
        return tensor_dims, block_dims
    
    @property
    def chunking_type(self) -> ChunkingType:
        return ChunkingType.DEFAULT


def get_default_block_shape(tensor_shape: List[int], interface_type: str) -> List[str]:
    """
    Get default block shape based on tensor layout and interface type.
    
    This implements smart defaults for different tensor layouts and interface types
    when no explicit BDIM pragma is provided. Uses only parameter names and ":",
    never magic numbers.
    
    Args:
        tensor_shape: Shape of the tensor
        interface_type: Type of interface (INPUT, OUTPUT, WEIGHT)
        
    Returns:
        Default block shape specification (only parameter names and ":")
    """
    from brainsmith.dataflow.core.interface_types import InterfaceType
    
    # Convert string to enum if needed
    if isinstance(interface_type, str):
        interface_type = InterfaceType(interface_type.upper())
    
    if interface_type in [InterfaceType.INPUT, InterfaceType.OUTPUT]:
        # For activation tensors, default to rightmost dimensions (full processing)
        if len(tensor_shape) == 4:  # NCHW layout
            # Default: process full H,W per block
            return [":", ":"]  # [H, W]
        elif len(tensor_shape) == 3:  # NLC or CHW layout
            # Default: process last dimension
            return [":"]  # [C] or [W]
        elif len(tensor_shape) == 2:  # NC layout
            # Default: process last dimension
            return [":"]  # [C]
        else:
            # Default: full tensor
            return [":"] * len(tensor_shape)
    
    elif interface_type == InterfaceType.WEIGHT:
        # For weight tensors - use parameter names for block sizes
        if len(tensor_shape) == 2:  # Matrix weights
            # Use PE parameter for block size (common in FINN)
            return ["PE"]  # Process PE rows at a time
        elif len(tensor_shape) == 1:  # Vector weights
            # Default: full vector
            return [":"]
        else:
            # Use PE parameter for first dimension
            return ["PE"]
    
    # Default: full tensor
    return [":"] * len(tensor_shape)


# Factory functions for creating strategies
def block_chunking(block_shape: List[str], rindex: int = 0) -> BlockChunkingStrategy:
    """Create a block chunking strategy with specified shape and starting index.
    
    Args:
        block_shape: List of strings - either ":" or parameter names (NO magic numbers)
        rindex: Right-to-left starting index
    """
    return BlockChunkingStrategy(block_shape=block_shape, rindex=rindex)


def default_chunking() -> DefaultChunkingStrategy:
    """Create a default chunking strategy (layout-aware defaults)."""
    return DefaultChunkingStrategy()
