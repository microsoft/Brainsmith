############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tiling strategy for applying shape expressions to tensors"""

from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Union
from enum import Enum
from .tiling_spec import TilingSpec
from .types import Shape


class TilingOrder(Enum):
    """Order in which to apply tiling to dimensions"""
    ROW_MAJOR = "row_major"      # Right-to-left (C-style)
    COLUMN_MAJOR = "column_major" # Left-to-right (Fortran-style)
    CUSTOM = "custom"             # User-defined order
    

@dataclass
class TilingResult:
    """Result of applying a tiling strategy"""
    block_dims: Shape
    parameters_used: Dict[str, int]
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class TilingStrategy:
    """Strategy for tiling tensor dimensions into blocks
    
    Handles the application of TilingSpec expressions to actual
    tensor shapes, including parameter resolution and validation.
    """
    
    def __init__(self, 
                 block_spec: Optional[TilingSpec] = None,
                 stream_spec: Optional[TilingSpec] = None,
                 order: TilingOrder = TilingOrder.ROW_MAJOR):
        """Initialize tiling strategy
        
        Args:
            block_spec: Specification for block tiling
            stream_spec: Specification for stream tiling (within blocks)
            order: Order to apply tiling
        """
        self.block_spec = block_spec
        self.stream_spec = stream_spec
        self.order = order
    
    def get_required_parameters(self) -> Dict[str, str]:
        """Get all parameters required by this strategy
        
        Returns:
            Dict mapping parameter names to their usage context
        """
        params = {}
        
        if self.block_spec:
            for param in self.block_spec.get_parameters():
                params[param] = "block_tiling"
        
        if self.stream_spec:
            for param in self.stream_spec.get_parameters():
                if param in params:
                    params[param] = "block_and_stream_tiling"
                else:
                    params[param] = "stream_tiling"
        
        return params
    
    def apply_block_tiling(self, 
                          tensor_shape: Shape, 
                          parameters: Dict[str, int]) -> TilingResult:
        """Apply block tiling to tensor shape
        
        Args:
            tensor_shape: Full tensor dimensions
            parameters: Parameter values for resolution
            
        Returns:
            TilingResult with block dimensions
            
        Raises:
            ValueError: If tiling cannot be applied
        """
        if not self.block_spec:
            # No block tiling specified - use full tensor
            return TilingResult(
                block_dims=list(tensor_shape),
                parameters_used={}
            )
        
        # Validate spec against shape
        errors = self.block_spec.validate_against_shape(tensor_shape)
        if errors:
            raise ValueError(f"Block tiling validation failed: {'; '.join(errors)}")
        
        # Resolve expressions to concrete values
        try:
            block_dims = self.block_spec.resolve(tensor_shape, parameters)
        except ValueError as e:
            raise ValueError(f"Failed to resolve block tiling: {e}")
        
        # Validate resolved dimensions
        warnings = []
        for i, (block_dim, tensor_dim) in enumerate(zip(block_dims, tensor_shape)):
            if block_dim > tensor_dim:
                raise ValueError(
                    f"Block dimension {i}: size {block_dim} exceeds "
                    f"tensor dimension {tensor_dim}"
                )
            if tensor_dim % block_dim != 0:
                warnings.append(
                    f"Block dimension {i}: size {block_dim} does not evenly "
                    f"divide tensor dimension {tensor_dim}"
                )
        
        # Extract parameters that were actually used
        params_used = {}
        for expr in self.block_spec.expressions:
            if expr.is_parameter and expr.parameter_name in parameters:
                params_used[expr.parameter_name] = parameters[expr.parameter_name]
        
        return TilingResult(
            block_dims=block_dims,
            parameters_used=params_used,
            warnings=warnings
        )
    
    def apply_stream_tiling(self, 
                           block_shape: Shape, 
                           parameters: Dict[str, int]) -> TilingResult:
        """Apply stream tiling to block shape
        
        Args:
            block_shape: Block dimensions to tile into streams
            parameters: Parameter values for resolution
            
        Returns:
            TilingResult with stream dimensions
            
        Raises:
            ValueError: If tiling cannot be applied
        """
        if not self.stream_spec:
            # No stream tiling - stream entire blocks
            return TilingResult(
                block_dims=list(block_shape),
                parameters_used={}
            )
        
        # Validate spec against block shape
        errors = self.stream_spec.validate_against_shape(block_shape)
        if errors:
            raise ValueError(f"Stream tiling validation failed: {'; '.join(errors)}")
        
        # Resolve expressions
        try:
            stream_dims = self.stream_spec.resolve(block_shape, parameters)
        except ValueError as e:
            raise ValueError(f"Failed to resolve stream tiling: {e}")
        
        # Validate stream dimensions don't exceed block dimensions
        warnings = []
        for i, (stream_dim, block_dim) in enumerate(zip(stream_dims, block_shape)):
            if stream_dim > block_dim:
                raise ValueError(
                    f"Stream dimension {i}: size {stream_dim} exceeds "
                    f"block dimension {block_dim}"
                )
            if block_dim % stream_dim != 0:
                warnings.append(
                    f"Stream dimension {i}: size {stream_dim} does not evenly "
                    f"divide block dimension {block_dim}"
                )
        
        # Extract parameters used
        params_used = {}
        for expr in self.stream_spec.expressions:
            if expr.is_parameter and expr.parameter_name in parameters:
                params_used[expr.parameter_name] = parameters[expr.parameter_name]
        
        return TilingResult(
            block_dims=stream_dims,
            parameters_used=params_used,
            warnings=warnings
        )
    
    def apply_full_tiling(self,
                         tensor_shape: Shape,
                         parameters: Dict[str, int]) -> Tuple[TilingResult, TilingResult]:
        """Apply both block and stream tiling
        
        Args:
            tensor_shape: Full tensor dimensions
            parameters: Parameter values
            
        Returns:
            Tuple of (block_result, stream_result)
        """
        # First apply block tiling
        block_result = self.apply_block_tiling(tensor_shape, parameters)
        
        # Then apply stream tiling to the block dimensions
        stream_result = self.apply_stream_tiling(block_result.block_dims, parameters)
        
        return block_result, stream_result
    
    def validate_parameters(self, parameters: Dict[str, int]) -> List[str]:
        """Validate that all required parameters are provided
        
        Args:
            parameters: Available parameters
            
        Returns:
            List of error messages (empty if valid)
        """
        errors = []
        required = self.get_required_parameters()
        
        for param_name, usage in required.items():
            if param_name not in parameters:
                errors.append(f"Missing parameter '{param_name}' used in {usage}")
            elif not isinstance(parameters[param_name], int):
                errors.append(
                    f"Parameter '{param_name}' must be an integer, "
                    f"got {type(parameters[param_name]).__name__}"
                )
            elif parameters[param_name] <= 0:
                errors.append(
                    f"Parameter '{param_name}' must be positive, "
                    f"got {parameters[param_name]}"
                )
        
        return errors
    
    @classmethod
    def from_expressions(cls,
                        block_expr: Optional[List[Union[int, str]]] = None,
                        stream_expr: Optional[List[Union[int, str]]] = None,
                        order: TilingOrder = TilingOrder.ROW_MAJOR) -> 'TilingStrategy':
        """Create strategy from expression lists
        
        Args:
            block_expr: Block tiling expressions
            stream_expr: Stream tiling expressions
            order: Tiling order
            
        Returns:
            TilingStrategy instance
        """
        block_spec = TilingSpec(block_expr) if block_expr else None
        stream_spec = TilingSpec(stream_expr) if stream_expr else None
        
        return cls(block_spec, stream_spec, order)
    
    def __repr__(self) -> str:
        parts = []
        if self.block_spec:
            parts.append(f"block={self.block_spec.to_list()}")
        if self.stream_spec:
            parts.append(f"stream={self.stream_spec.to_list()}")
        if self.order != TilingOrder.ROW_MAJOR:
            parts.append(f"order={self.order.value}")
        
        return f"TilingStrategy({', '.join(parts)})"