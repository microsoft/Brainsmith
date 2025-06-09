"""
Simplified tensor chunking system using per-interface strategies.

This module provides utilities for extracting tensor shapes and delegating
chunking computation to individual interface strategies.
"""

from typing import List, Tuple, Optional


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
