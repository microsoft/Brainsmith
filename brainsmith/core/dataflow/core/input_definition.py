############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Input interface definition"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Any
from .base import BaseDefinition
from .types import DataType, Shape
from .input_interface import InputInterface

@dataclass
class InputDefinition(BaseDefinition):
    """Definition for an input interface
    
    Defines the schema and constraints for an input that can be
    instantiated with different tensor dimensions.
    """
    
    name: str
    dtype: DataType
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]] = None
    onnx_layout: Optional[str] = None
    granularity: Optional[Shape] = None
    optional: bool = False
    rate_pattern: Optional[List[int]] = None
    
    def create_model(self,
                    tensor_dims: Shape,
                    parameter_binding: Optional[Dict[str, int]] = None,
                    config: Optional[Dict[str, Any]] = None) -> InputInterface:
        """Create a runtime input model instance
        
        Args:
            tensor_dims: Full tensor dimensions
            parameter_binding: Parameter values for expressions
            config: Runtime configuration
            
        Returns:
            InputInterface instance
        """
        # Derive block dimensions
        block_dims = self.derive_block_dims(tensor_dims, parameter_binding, config)
        
        # Create input model
        return InputInterface(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            definition=self,
            parameter_binding=parameter_binding
        )
    
    def derive_block_dims(self,
                         tensor_dims: Shape,
                         parameter_binding: Optional[Dict[str, int]] = None,
                         config: Optional[Dict[str, Any]] = None) -> Shape:
        """Derive concrete block dimensions
        
        This would implement the full expression evaluation logic
        from the original InterfaceDefinition.
        """
        # Simplified for now - would implement full logic
        if self.block_dims_expr is None:
            # Default chunking
            return self._default_block_chunking(tensor_dims)
        
        if callable(self.block_dims_expr):
            # Call tiling function
            return self.block_dims_expr(tensor_dims, parameter_binding or {}, config or {})
        
        # Expression-based tiling would go here
        return tensor_dims  # Placeholder
    
    def _default_block_chunking(self, tensor_dims: Shape) -> Shape:
        """Default chunking strategy based on layout"""
        # Simplified default
        return tensor_dims
    
    def validate(self) -> List[str]:
        """Validate definition consistency"""
        errors = []
        
        if not self.name:
            errors.append("Input name cannot be empty")
        
        if self.granularity:
            if len(self.granularity) == 0:
                errors.append("Granularity cannot be empty")
        
        return errors
    
    def __repr__(self) -> str:
        return f"InputDefinition(name='{self.name}', dtype={self.dtype})"