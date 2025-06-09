"""
Tensor chunking system with strategy-based approach for interface-wise processing.

This module provides utilities for extracting tensor shapes and chunking strategies
for different types of tensor processing requirements.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple, Optional, Union
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
        Compute qDim and tDim for the given tensor shape.
        
        Args:
            tensor_shape: Input tensor shape
            interface_name: Interface name for context
            
        Returns:
            Tuple of (qDim, tDim) lists
            - qDim: Original tensor dimensions (typically same as tensor_shape)
            - tDim: Processing chunk dimensions
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
        # Default: no chunking - qDim equals tensor_shape, tDim equals tensor_shape
        qDim = list(tensor_shape)  # Original tensor dimensions
        tDim = list(tensor_shape)  # Process entire tensor (no chunking)
        return qDim, tDim


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
        qDim = list(tensor_shape)  # Original tensor dimensions
        tDim = list(tensor_shape)  # Process entire tensor (no chunking)
        return qDim, tDim
    
    def _apply_shaped_chunking(self, tensor_shape: List[int], start_idx: int) -> Tuple[List[int], List[int]]:
        """Apply shaped chunking with broadcasting rules."""
        # Initialize with original tensor shape
        qDim = list(tensor_shape)  # Original tensor dimensions
        tDim = list(tensor_shape)  # Start with original, will be modified for chunking
        
        # Resolve shape elements
        resolved_shape = [self._resolve_shape_element(s) for s in self.shape]
        
        # Apply broadcasting: 1D affects start_idx, 2D affects start_idx+1, etc.
        for i, shape_val in enumerate(resolved_shape):
            target_idx = start_idx + i
            if target_idx < len(tensor_shape):
                # Chunk at this dimension
                if shape_val > 0 and tensor_shape[target_idx] >= shape_val:
                    tDim[target_idx] = shape_val  # Set chunk size
                    # qDim stays as original tensor_shape[target_idx]
                    # num_tensors = qDim[target_idx] // tDim[target_idx] will be computed by DataflowInterface
                else:
                    # Fallback: preserve dimension (no chunking at this dimension)
                    tDim[target_idx] = tensor_shape[target_idx]
        
        return qDim, tDim
    
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
        
        qDim = list(tensor_shape)  # Original tensor dimensions
        tDim = list(tensor_shape)  # Process entire tensor (no chunking)
        return qDim, tDim
    
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


class TensorChunking:
    """
    Simplified tensor chunking system.
    
    Extracts tensor shapes and delegates chunking to interface-specific strategies.
    No override system needed - each interface has its own strategy.
    """
    
    def __init__(self):
        """Initialize tensor chunking system."""
        self._model_wrapper = None
    
    def compute_chunking_for_interface(self, interface_metadata, onnx_node) -> Tuple[List[int], List[int]]:
        """
        Compute chunking for a specific interface using its strategy.
        
        Args:
            interface_metadata: InterfaceMetadata with chunking strategy
            onnx_node: ONNX node for tensor shape extraction
            
        Returns:
            Tuple of (qDim, tDim) lists
        """
        # Extract tensor shape for this interface
        tensor_shape = self.extract_tensor_shape_from_input(interface_metadata.name, onnx_node)
        
        # Delegate to the interface's chunking strategy
        return interface_metadata.chunking_strategy.compute_chunking(tensor_shape, interface_metadata.name)
    
    def extract_tensor_shape_from_input(self, interface_name: str, onnx_node) -> List[int]:
        """
        Extract tensor shape from ONNX node input tensors.
        
        Args:
            interface_name: Interface name
            onnx_node: ONNX node containing input tensor references
            
        Returns:
            List of tensor dimensions
        """
        # Map interface name to input tensor index
        input_index = self._map_interface_to_input_index(interface_name)
        
        if input_index is None:
            raise RuntimeError(f"Cannot map interface '{interface_name}' to input tensor index")
        
        try:
            # Get input tensor name
            if hasattr(onnx_node, 'input') and len(onnx_node.input) > input_index:
                input_tensor_name = onnx_node.input[input_index]
            else:
                raise RuntimeError(f"No input tensor at index {input_index} for interface '{interface_name}'")
            
            # Try to extract shape from model wrapper
            if self._model_wrapper:
                try:
                    tensor_shape = self._model_wrapper.get_tensor_shape(input_tensor_name)
                    if tensor_shape and len(tensor_shape) > 0:
                        return [int(dim) for dim in tensor_shape]
                except Exception:
                    # Gracefully handle ModelWrapper errors
                    pass
            
            # No fallback - ModelWrapper must provide valid shape
            raise RuntimeError(f"ModelWrapper required but not available for interface '{interface_name}'")
            
        except (IndexError, AttributeError) as e:
            raise RuntimeError(f"Failed to extract tensor shape for interface '{interface_name}': {e}")
    
    def _map_interface_to_input_index(self, interface_name: str) -> Optional[int]:
        """Map interface name to input tensor index."""
        if "in0" in interface_name or interface_name.startswith("input"):
            return 0
        elif "in1" in interface_name:
            return 1
        elif "in2" in interface_name:
            return 2
        elif "weights" in interface_name or "weight" in interface_name:
            return 1
        elif "bias" in interface_name:
            return 2
        else:
            return 0
    
    def _get_default_shape_for_interface(self, interface_name: str) -> List[int]:
        """
        DEPRECATED: Default shapes not allowed - all dimensions must be runtime extracted.
        """
        raise RuntimeError(
            f"Default shapes not allowed for interface '{interface_name}'. "
            f"ModelWrapper must be provided for runtime dimension extraction."
        )
    
    def set_model_wrapper(self, model_wrapper):
        """Set model wrapper for tensor shape extraction."""
        self._model_wrapper = model_wrapper
    
    def infer_layout_from_shape(self, tensor_shape: List[int]) -> str:
        """Infer tensor layout with smart defaults."""
        layout_map = {
            4: "NCHW",  # 4D tensors → batch, channels, height, width
            3: "CHW",   # 3D tensors → channels, height, width
            2: "NC",    # 2D tensors → batch, channels
            1: "C"      # 1D tensors → channels
        }
        return layout_map.get(len(tensor_shape), f"DIM{len(tensor_shape)}")
    
    def get_layout_aware_chunking(self, tensor_shape: List[int], layout: str = None) -> Tuple[List[int], List[int]]:
        """Provide layout-aware default chunking strategies."""
        if layout is None:
            layout = self.infer_layout_from_shape(tensor_shape)
            
        if layout == "NCHW" and len(tensor_shape) == 4:
            # Default: no chunking on batch/channels, stream on width
            return ([1, 1, 1, tensor_shape[3]], [1, tensor_shape[1], tensor_shape[2], 1])
        elif layout == "CHW" and len(tensor_shape) == 3:
            # Default: stream on width
            return ([1, 1, tensor_shape[2]], [tensor_shape[0], tensor_shape[1], 1])
        elif layout == "NC" and len(tensor_shape) == 2:
            # Default: stream on channels
            return ([1, tensor_shape[1]], [tensor_shape[0], 1])
        elif layout == "C" and len(tensor_shape) == 1:
            # Default: stream elements
            return ([tensor_shape[0]], [1])
        else:
            # Conservative default: full tensor
            return ([1] * len(tensor_shape), tensor_shape)
