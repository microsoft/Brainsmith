############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Output interface definition"""

from dataclasses import dataclass, field
from typing import Optional, Union, List, Callable, Dict, Any
from .base import BaseDefinition, ParameterBinding
from .types import Shape
from .qonnx_types import BaseDataType, DatatypeConstraintGroup, validate_datatype_against_constraints
from .output_interface import OutputInterface

@dataclass
class OutputDefinition(BaseDefinition):
    """Definition for an output interface
    
    Defines the schema for an output. Outputs don't have
    configurable SDIM - their streaming rate is determined
    by the kernel computation.
    """
    
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_dims_expr: Optional[Union[List[Union[str, int]], Callable]] = None
    onnx_layout: Optional[str] = None
    optional: bool = False
    rate_pattern: Optional[List[int]] = None
    
    def create_model(self,
                    tensor_dims: Shape,
                    datatype: BaseDataType,
                    parameter_binding: Optional[Union[Dict[str, int], ParameterBinding]] = None,
                    config: Optional[Dict[str, Any]] = None) -> OutputInterface:
        """Create a runtime output model instance
        
        Args:
            tensor_dims: Full tensor dimensions
            datatype: Concrete QONNX datatype (must satisfy constraints)
            parameter_binding: Parameter values for expressions
            config: Runtime configuration
            
        Returns:
            OutputInterface instance
            
        Raises:
            ValueError: If datatype doesn't satisfy constraints
        """
        # Validate datatype
        if self.datatype_constraints and not self.validates_datatype(datatype):
            valid_types = self._get_valid_type_names()
            raise ValueError(
                f"Datatype {datatype.get_canonical_name()} doesn't satisfy "
                f"constraints for output '{self.name}'. Valid types: {valid_types}"
            )
        
        # Convert parameter_binding if needed
        if isinstance(parameter_binding, dict):
            parameter_binding = ParameterBinding(parameter_binding)
        
        # Derive block dimensions
        block_dims = self.derive_block_dims(tensor_dims, parameter_binding, config)
        
        # Create output model
        return OutputInterface(
            tensor_dims=tensor_dims,
            block_dims=block_dims,
            datatype=datatype,
            definition=self,
            parameter_binding=parameter_binding
        )
    
    def validates_datatype(self, datatype: BaseDataType) -> bool:
        """Check if datatype satisfies constraints"""
        if not self.datatype_constraints:
            return True  # No constraints = allow any
        return validate_datatype_against_constraints(datatype, self.datatype_constraints)
    
    def _get_valid_type_names(self) -> List[str]:
        """Get list of valid type names from constraints"""
        valid_types = []
        for constraint in self.datatype_constraints:
            if constraint.base_type in ["INT", "UINT"]:
                for width in range(constraint.min_width, constraint.max_width + 1):
                    valid_types.append(f"{constraint.base_type}{width}")
            else:
                valid_types.append(constraint.base_type)
        return valid_types
    
    def derive_block_dims(self,
                         tensor_dims: Shape,
                         parameter_binding: Optional[ParameterBinding] = None,
                         config: Optional[Dict[str, Any]] = None) -> Shape:
        """Derive concrete block dimensions"""
        # Simplified for now - would implement full logic
        if self.block_dims_expr is None:
            return self._default_block_chunking(tensor_dims)
        
        if callable(self.block_dims_expr):
            return self.block_dims_expr(tensor_dims, parameter_binding or {}, config or {})
        
        # Handle list of expressions
        if isinstance(self.block_dims_expr, list):
            result = []
            for i, expr in enumerate(self.block_dims_expr):
                if isinstance(expr, int):
                    # Literal integer
                    result.append(expr)
                elif expr == ":" and i < len(tensor_dims):
                    # Full dimension
                    result.append(tensor_dims[i])
                else:
                    # For now, default to tensor dimension
                    result.append(tensor_dims[i] if i < len(tensor_dims) else 1)
            return tuple(result)
        
        # Default
        return tensor_dims
    
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
        constraint_str = f"{len(self.datatype_constraints)} constraints" if self.datatype_constraints else "no constraints"
        return f"OutputDefinition(name='{self.name}', {constraint_str})"