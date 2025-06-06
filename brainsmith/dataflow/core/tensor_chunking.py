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
