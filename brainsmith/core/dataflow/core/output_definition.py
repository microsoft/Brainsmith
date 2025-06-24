############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Output interface definition"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Any
from .base import BaseDefinition
from .types import DataType, Shape
from .output_interface import OutputInterface

@dataclass
class OutputDefinition(BaseDefinition):
    """Definition for an output interface
    
    Defines the schema for an output. Outputs don't have
    configurable SDIM - their streaming rate is determined
    by the kernel computation.
    """
    
    name: str
    dtype: DataType
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]] = None
    onnx_layout: Optional[str] = None
    optional: bool = False
    rate_pattern: Optional[List[int]] = None
    
    def create_model(self,
                    tensor_dims: Shape,
                    parameter_binding: Optional[Dict[str, int]] = None,
                    config: Optional[Dict[str, Any]] = None) -> OutputInterface:
        """Create a runtime output model instance
        
        Args:
            tensor_dims: Full tensor dimensions
            parameter_binding: Parameter values for expressions
            config: Runtime configuration
            
        Returns:
            OutputInterface instance
        """
        # Derive block dimensions
        block_dims = self.derive_block_dims(tensor_dims, parameter_binding, config)
        
        # Create output model
        return OutputInterface(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            definition=self,
            parameter_binding=parameter_binding
        )
    
    def derive_block_dims(self,
                         tensor_dims: Shape,
                         parameter_binding: Optional[Dict[str, int]] = None,
                         config: Optional[Dict[str, Any]] = None) -> Shape:
        """Derive concrete block dimensions"""
        # Simplified for now - would implement full logic
        if self.block_dims_expr is None:
            return self._default_block_chunking(tensor_dims)
        
        if callable(self.block_dims_expr):
            return self.block_dims_expr(tensor_dims, parameter_binding or {}, config or {})
        
        return tensor_dims  # Placeholder
    
    def _default_block_chunking(self, tensor_dims: Shape) -> Shape:
        """Default chunking strategy"""
        return tensor_dims
    
    def validate(self) -> List[str]:
        """Validate definition consistency"""
        errors = []
        
        if not self.name:
            errors.append("Output name cannot be empty")
        
        return errors
    
    def __repr__(self) -> str:
        return f"OutputDefinition(name='{self.name}', dtype={self.dtype})"