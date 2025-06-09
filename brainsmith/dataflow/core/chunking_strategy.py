"""
Chunking strategies for tensor processing in dataflow interfaces.

This module provides chunking strategy definitions that can be attached
to individual interfaces, eliminating the need for a separate override system.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Union, Optional
from enum import Enum


class ChunkingType(Enum):
    """Types of chunking strategies."""
    DEFAULT = "default"          # Layout-aware default chunking
    INDEX_BASED = "index_based"  # Index and shape specification
    FULL_TENSOR = "full_tensor"  # No chunking (full tensor)
    CUSTOM = "custom"            # Custom strategy implementation


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""
    
    @abstractmethod
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """
        Compute num_tensors and tDim for the given tensor shape.
        
        Args:
            tensor_shape: Input tensor shape
            interface_name: Interface name for context
            
        Returns:
            Tuple of (num_tensors, tDim) lists
        """
        pass
    
    @property
    @abstractmethod
    def chunking_type(self) -> ChunkingType:
        """Get the type of this chunking strategy."""
        pass


@dataclass
class DefaultChunkingStrategy(ChunkingStrategy):
    """Default layout-aware chunking strategy."""
    
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """Apply layout-aware default chunking."""
        if not tensor_shape:
            return [1], [1]
        
        layout = self._infer_layout_from_shape(tensor_shape)
        return self._apply_layout_aware_chunking(tensor_shape, layout)
    
    @property
    def chunking_type(self) -> ChunkingType:
        return ChunkingType.DEFAULT
    
    def _infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
        """Infer tensor layout from shape dimensions."""
        if len(tensor_shape) == 4:
            return "NCHW"
        elif len(tensor_shape) == 3:
            return "CHW"
        elif len(tensor_shape) == 2:
            return "NC"
        elif len(tensor_shape) == 1:
            return "C"
        else:
            return "UNKNOWN"
    
    def _apply_layout_aware_chunking(self, tensor_shape: List[int], layout: str) -> Tuple[List[int], List[int]]:
        """Apply sophisticated layout-aware chunking."""
        # Default: minimal chunking, preserve full tensor dimensions
        num_tensors = [1] * len(tensor_shape)
        tDim = list(tensor_shape)
        return num_tensors, tDim


@dataclass
class IndexBasedChunkingStrategy(ChunkingStrategy):
    """Index-based chunking strategy with shape specification and broadcasting."""
    
    start_index: int                    # Starting dimension index for chunking
    shape: List[Union[str, int]]       # [tdim1, tdim2] or [":"] format
    
    def __post_init__(self):
        """Validate strategy parameters."""
        if not isinstance(self.start_index, int):
            raise ValueError("start_index must be an integer")
        if not isinstance(self.shape, list):
            raise ValueError("shape must be a list")
    
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """Apply index-based chunking with broadcasting rules."""
        if not tensor_shape:
            return [1], [1]
        
        start_idx = self.start_index
        
        # Normalize negative start index
        if start_idx < 0:
            start_idx = len(tensor_shape) + start_idx
        
        # Validate start index
        if not (0 <= start_idx < len(tensor_shape)):
            raise ValueError(f"Start index {self.start_index} out of bounds for shape {tensor_shape}")
        
        # Handle shape formats
        if self.shape == [":"] or self.shape == ":" or (len(self.shape) == 1 and self.shape[0] == ":"):
            # Full tensor - no chunking
            return self._apply_full_tensor_strategy(tensor_shape)
        else:
            # Shaped chunking with broadcasting
            return self._apply_shaped_chunking(tensor_shape, start_idx)
    
    @property
    def chunking_type(self) -> ChunkingType:
        return ChunkingType.INDEX_BASED
    
    def _apply_full_tensor_strategy(self, tensor_shape: List[int]) -> Tuple[List[int], List[int]]:
        """Apply full tensor strategy (no chunking)."""
        num_tensors = [1] * len(tensor_shape)
        tDim = list(tensor_shape)
        return num_tensors, tDim
    
    def _apply_shaped_chunking(self, tensor_shape: List[int], start_idx: int) -> Tuple[List[int], List[int]]:
        """Apply shaped chunking with broadcasting rules."""
        # Initialize with defaults
        num_tensors = [1] * len(tensor_shape)
        tDim = list(tensor_shape)
        
        # Resolve shape elements
        resolved_shape = [self._resolve_shape_element(s) for s in self.shape]
        
        # Apply broadcasting: 1D affects start_idx, 2D affects start_idx+1, etc.
        for i, shape_val in enumerate(resolved_shape):
            target_idx = start_idx + i
            if target_idx < len(tensor_shape):
                # Chunk at this dimension
                if shape_val > 0 and tensor_shape[target_idx] >= shape_val:
                    num_tensors[target_idx] = tensor_shape[target_idx] // shape_val
                    tDim[target_idx] = shape_val
                else:
                    # Fallback: preserve dimension
                    num_tensors[target_idx] = 1
                    tDim[target_idx] = tensor_shape[target_idx]
        
        return num_tensors, tDim
    
    def _resolve_shape_element(self, element: Union[str, int]) -> int:
        """Resolve a shape element to an integer."""
        if isinstance(element, int):
            return element
        elif isinstance(element, str):
            # String parameters resolved by HWKG layer - return safe default
            return 1
        else:
            raise ValueError(f"Invalid shape element: {element}")


@dataclass
class FullTensorChunkingStrategy(ChunkingStrategy):
    """Full tensor strategy - no chunking applied."""
    
    def compute_chunking(self, tensor_shape: List[int], interface_name: str) -> Tuple[List[int], List[int]]:
        """Return full tensor without chunking."""
        if not tensor_shape:
            return [1], [1]
        
        num_tensors = [1] * len(tensor_shape)
        tDim = list(tensor_shape)
        return num_tensors, tDim
    
    @property
    def chunking_type(self) -> ChunkingType:
        return ChunkingType.FULL_TENSOR


# Convenience factory functions
def default_chunking() -> DefaultChunkingStrategy:
    """Create default chunking strategy."""
    return DefaultChunkingStrategy()


def index_chunking(start_index: int, shape: List[Union[str, int]]) -> IndexBasedChunkingStrategy:
    """Create index-based chunking strategy."""
    return IndexBasedChunkingStrategy(start_index=start_index, shape=shape)


def full_tensor_chunking() -> FullTensorChunkingStrategy:
    """Create full tensor chunking strategy."""
    return FullTensorChunkingStrategy()


# Common chunking strategies
def last_dim_chunking(chunk_size: int) -> IndexBasedChunkingStrategy:
    """Chunk the last dimension with specified size."""
    return IndexBasedChunkingStrategy(start_index=-1, shape=[chunk_size])


def spatial_chunking(height: int, width: int) -> IndexBasedChunkingStrategy:
    """Chunk spatial dimensions (assumes NCHW layout)."""
    return IndexBasedChunkingStrategy(start_index=2, shape=[height, width])


def no_chunking() -> FullTensorChunkingStrategy:
    """No chunking - process full tensor."""
    return FullTensorChunkingStrategy()