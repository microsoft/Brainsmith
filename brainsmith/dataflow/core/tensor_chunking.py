"""
Tensor chunking utilities for dataflow modeling.

This module provides utilities for managing tensor chunking in dataflow models.
"""

from typing import List, Optional
from dataclasses import dataclass
from enum import Enum


class ChunkingStrategy(Enum):
    """Strategies for chunking tensors"""
    BROADCAST = "broadcast"
    DIVIDE = "divide"
    EXPLICIT = "explicit"


@dataclass
class TensorChunk:
    """Represents a chunk of a tensor"""
    original_shape: List[int]
    chunk_shape: List[int]
    chunk_index: List[int]
    strategy: ChunkingStrategy
    

def calculate_tensor_chunks(original_shape: List[int], 
                          chunk_size: List[int],
                          strategy: ChunkingStrategy = ChunkingStrategy.DIVIDE) -> List[TensorChunk]:
    """
    Calculate tensor chunks based on original shape and chunk size.
    
    Args:
        original_shape: Original tensor shape
        chunk_size: Desired chunk size
        strategy: Chunking strategy to use
        
    Returns:
        List of TensorChunk objects
    """
    # Simple implementation for now
    chunks = []
    
    if strategy == ChunkingStrategy.DIVIDE:
        # Simple division-based chunking
        chunk = TensorChunk(
            original_shape=original_shape,
            chunk_shape=chunk_size,
            chunk_index=[0] * len(original_shape),
            strategy=strategy
        )
        chunks.append(chunk)
    
    return chunks


class TensorChunking:
    """Utility class for tensor chunking operations in dataflow modeling."""
    
    def __init__(self):
        """Initialize tensor chunking utility."""
        pass
    
    def infer_dimensions(self, onnx_layout: str, onnx_shape: List[int]) -> tuple[List[int], List[int]]:
        """
        Infer qDim and tDim from ONNX layout and shape.
        
        Args:
            onnx_layout: ONNX tensor layout (e.g., "NCHW", "NHWC")
            onnx_shape: ONNX tensor shape
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        if not onnx_layout or not onnx_shape:
            return [1], [1]
            
        # Simple default: treat all dimensions as tensor dimensions
        # In a real implementation, this would be more sophisticated
        tDim = list(onnx_shape)
        qDim = [1] * len(onnx_shape)  # Default to single element per dimension
        
        return qDim, tDim
    
    def infer_dimensions_with_layout(self, tensor_layout: str, tensor_shape: List[int]) -> tuple[List[int], List[int]]:
        """
        Infer qDim and tDim from tensor layout and shape with layout-aware chunking.
        
        Args:
            tensor_layout: Tensor layout (e.g., "NCHW", "NHWC", "CHW", "HWC", "NC", "C")
            tensor_shape: Tensor shape
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        if not tensor_layout or not tensor_shape:
            return [1], [1]
            
        # Layout-aware default chunking strategies
        if tensor_layout == "NCHW" and len(tensor_shape) >= 4:
            # For NCHW: typically process one sample at a time, full channels, spatial chunking
            N, C, H, W = tensor_shape[:4]
            qDim = [1, 1, 1, 1]  # Process one element at a time by default
            tDim = [N, C, H, W]  # Full tensor dimensions
            
        elif tensor_layout == "NHWC" and len(tensor_shape) >= 4:
            # For NHWC: typically process one sample at a time, spatial chunking, channel parallelism
            N, H, W, C = tensor_shape[:4]
            qDim = [1, 1, 1, 1]  # Process one element at a time by default
            tDim = [N, H, W, C]  # Full tensor dimensions
            
        elif tensor_layout == "CHW" and len(tensor_shape) >= 3:
            # For CHW: channel-wise processing with spatial chunking
            C, H, W = tensor_shape[:3]
            qDim = [1, 1, 1]  # Process one element at a time by default
            tDim = [C, H, W]   # Full tensor dimensions
            
        elif tensor_layout == "HWC" and len(tensor_shape) >= 3:
            # For HWC: spatial processing with channel parallelism
            H, W, C = tensor_shape[:3]
            qDim = [1, 1, 1]  # Process one element at a time by default
            tDim = [H, W, C]   # Full tensor dimensions
            
        elif tensor_layout == "NC" and len(tensor_shape) >= 2:
            # For NC: batch processing with channel parallelism
            N, C = tensor_shape[:2]
            qDim = [1, 1]    # Process one element at a time by default
            tDim = [N, C]    # Full tensor dimensions
            
        elif tensor_layout == "C" and len(tensor_shape) >= 1:
            # For C: channel-wise processing
            C = tensor_shape[0]
            qDim = [1]       # Process one element at a time by default
            tDim = [C]       # Full tensor dimension
            
        else:
            # Fallback: treat as generic tensor
            qDim = [1] * len(tensor_shape)
            tDim = list(tensor_shape)
            
        # Extend dimensions if tensor_shape is longer than expected layout
        if len(tensor_shape) > len(tDim):
            extra_dims = len(tensor_shape) - len(tDim)
            qDim.extend([1] * extra_dims)
            tDim.extend(tensor_shape[len(tDim):])
            
        return qDim, tDim
    
    def apply_chunking_strategy(self, tensor_layout: str, tensor_shape: List[int],
                              strategy: str = "default") -> tuple[List[int], List[int]]:
        """
        Apply specific chunking strategy based on layout and use case.
        
        Args:
            tensor_layout: Tensor layout (e.g., "NCHW", "NHWC")
            tensor_shape: Tensor shape
            strategy: Chunking strategy ("default", "streaming", "block", "channel_parallel")
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        if strategy == "streaming":
            # Streaming strategy: process minimal chunks for low latency
            return self._apply_streaming_chunking(tensor_layout, tensor_shape)
        elif strategy == "block":
            # Block strategy: process larger blocks for throughput
            return self._apply_block_chunking(tensor_layout, tensor_shape)
        elif strategy == "channel_parallel":
            # Channel parallel strategy: exploit channel parallelism
            return self._apply_channel_parallel_chunking(tensor_layout, tensor_shape)
        else:
            # Default strategy
            return self.infer_dimensions_with_layout(tensor_layout, tensor_shape)
    
    def _apply_streaming_chunking(self, tensor_layout: str, tensor_shape: List[int]) -> tuple[List[int], List[int]]:
        """Apply streaming-oriented chunking for minimal latency."""
        if tensor_layout == "NCHW" and len(tensor_shape) >= 4:
            N, C, H, W = tensor_shape[:4]
            # Stream one pixel at a time across all channels
            qDim = [1, 1, H, W]
            tDim = [N, C, 1, 1]
        elif tensor_layout == "CHW" and len(tensor_shape) >= 3:
            C, H, W = tensor_shape[:3]
            # Stream one pixel at a time across all channels
            qDim = [1, H, W]
            tDim = [C, 1, 1]
        else:
            # Fallback to default
            return self.infer_dimensions_with_layout(tensor_layout, tensor_shape)
        
        return qDim, tDim
    
    def _apply_block_chunking(self, tensor_layout: str, tensor_shape: List[int]) -> tuple[List[int], List[int]]:
        """Apply block-oriented chunking for maximum throughput."""
        if tensor_layout == "NCHW" and len(tensor_shape) >= 4:
            N, C, H, W = tensor_shape[:4]
            # Process blocks of spatial data
            block_h = min(8, H)  # 8x8 blocks or smaller
            block_w = min(8, W)
            qDim = [1, 1, H // block_h, W // block_w]
            tDim = [N, C, block_h, block_w]
        elif tensor_layout == "CHW" and len(tensor_shape) >= 3:
            C, H, W = tensor_shape[:3]
            # Process blocks of spatial data
            block_h = min(8, H)
            block_w = min(8, W)
            qDim = [1, H // block_h, W // block_w]
            tDim = [C, block_h, block_w]
        else:
            # Fallback to default
            return self.infer_dimensions_with_layout(tensor_layout, tensor_shape)
        
        return qDim, tDim
    
    def _apply_channel_parallel_chunking(self, tensor_layout: str, tensor_shape: List[int]) -> tuple[List[int], List[int]]:
        """Apply channel-parallel chunking for maximum parallelism."""
        if tensor_layout == "NCHW" and len(tensor_shape) >= 4:
            N, C, H, W = tensor_shape[:4]
            # Parallelize across channels
            qDim = [1, C, 1, 1]
            tDim = [N, 1, H, W]
        elif tensor_layout == "NHWC" and len(tensor_shape) >= 4:
            N, H, W, C = tensor_shape[:4]
            # Parallelize across channels
            qDim = [1, 1, 1, C]
            tDim = [N, H, W, 1]
        elif tensor_layout == "CHW" and len(tensor_shape) >= 3:
            C, H, W = tensor_shape[:3]
            # Parallelize across channels
            qDim = [C, 1, 1]
            tDim = [1, H, W]
        else:
            # Fallback to default
            return self.infer_dimensions_with_layout(tensor_layout, tensor_shape)
        
        return qDim, tDim
    
    @staticmethod
    def _compute_qDim_from_chunking(original_shape: List[int], tDim: List[int]) -> List[int]:
        """
        Compute qDim from original shape and tDim.
        
        Args:
            original_shape: Original tensor shape
            tDim: Target tensor dimensions
            
        Returns:
            Computed qDim
        """
        if not original_shape or not tDim:
            return [1]
            
        # Simple computation: qDim = original_shape / tDim (with broadcasting)
        qDim = []
        for i in range(min(len(original_shape), len(tDim))):
            if tDim[i] > 0:
                qDim.append(max(1, original_shape[i] // tDim[i]))
            else:
                qDim.append(1)
                
        # Pad with 1s if tDim is longer
        while len(qDim) < len(tDim):
            qDim.append(1)
            
        return qDim
