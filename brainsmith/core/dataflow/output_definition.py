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
from .tiling_spec import TilingSpec
from .tiling_strategy import TilingStrategy

@dataclass
class OutputDefinition(BaseDefinition):
    """Definition for an output interface
    
    Defines the schema for an output. Outputs don't have
    configurable SDIM - their streaming rate is determined
    by the kernel computation.
    """
    
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[List[Union[int, str]]] = None
    optional: bool = False
    
    # Internal tiling specification (created from list)
    _block_tiling_spec: Optional[TilingSpec] = field(init=False, default=None)
    _tiling_strategy: Optional[TilingStrategy] = field(init=False, default=None)
    
    def __post_init__(self):
        """Convert tiling list to internal TilingSpec object"""
        if self.block_tiling is not None:
            self._block_tiling_spec = TilingSpec(self.block_tiling)
            # Create tiling strategy (outputs only have block tiling)
            self._tiling_strategy = TilingStrategy(
                block_spec=self._block_tiling_spec,
                stream_spec=None  # Outputs don't have configurable stream tiling
            )
    
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
        """Derive concrete block dimensions using tiling strategy
        
        Args:
            tensor_dims: Full tensor dimensions
            parameter_binding: Parameter values for expressions
            config: Runtime configuration
            
        Returns:
            Block dimensions
            
        Raises:
            ValueError: If tiling cannot be applied
        """
        if not self._tiling_strategy or not self._tiling_strategy.block_spec:
            # No block tiling specified - use full tensor
            return list(tensor_dims)
        
        # Get parameters dict
        params_dict = parameter_binding.parameters if parameter_binding else {}
        
        # Apply block tiling
        result = self._tiling_strategy.apply_block_tiling(tensor_dims, params_dict)
        
        # Log warnings if any
        if result.warnings:
            for warning in result.warnings:
                print(f"Warning in {self.name} block tiling: {warning}")
        
        return result.block_dims
    
    def _default_block_chunking(self, tensor_dims: Shape) -> Shape:
        """Default chunking strategy"""
        return tensor_dims
    
    def validate(self) -> List[str]:
        """Validate definition consistency"""
        errors = []
        
        if not self.name:
            errors.append("Output name cannot be empty")
        
        return errors
    
    def get_tiling_parameters(self) -> Dict[str, str]:
        """Get all parameters used in tiling expressions
        
        Returns:
            Dict mapping parameter names to their usage context
        """
        if self._tiling_strategy:
            return self._tiling_strategy.get_required_parameters()
        return {}
    
    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        
        if self.datatype_constraints:
            parts.append(f"{len(self.datatype_constraints)} constraints")
        
        if self.block_tiling:
            parts.append(f"block_tiling={self.block_tiling}")
        
        return f"OutputDefinition({', '.join(parts)})"