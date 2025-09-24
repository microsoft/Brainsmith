############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel definition with separate input/output definitions"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING
from .base import BaseDefinition, ParameterBinding
from .types import Shape
from .qonnx_types import BaseDataType
from .input_definition import InputDefinition
from .output_definition import OutputDefinition
from .relationships import DimensionRelationship, RelationType


@dataclass
class KernelDefinition(BaseDefinition):
    """Definition of a kernel with separate input/output interfaces
    
    Key features:
    - Separate input_definitions and output_definitions
    - Clean API with add_input() and add_output()
    - Relationships primarily between inputs
    - Pure algorithmic abstraction independent of code generation
    """
    
    name: str
    input_definitions: List[InputDefinition] = field(default_factory=list)
    output_definitions: List[OutputDefinition] = field(default_factory=list)
    relationships: List[DimensionRelationship] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def add_input(self, input_def: InputDefinition) -> None:
        """Add an input definition"""
        # Check for duplicate names
        existing_names = {inp.name for inp in self.input_definitions}
        if input_def.name in existing_names:
            raise ValueError(f"Input '{input_def.name}' already exists")
        
        self.input_definitions.append(input_def)
    
    def add_output(self, output_def: OutputDefinition) -> None:
        """Add an output definition"""
        # Check for duplicate names
        existing_input_names = {inp.name for inp in self.input_definitions}
        existing_output_names = {out.name for out in self.output_definitions}
        
        if output_def.name in existing_input_names:
            raise ValueError(f"Name '{output_def.name}' already used by an input")
        if output_def.name in existing_output_names:
            raise ValueError(f"Output '{output_def.name}' already exists")
        
        self.output_definitions.append(output_def)
    
    def add_relationship(self,
                        source_name: str,
                        target_name: str,
                        relationship_type: RelationType,
                        source_dim: Optional[int] = None,
                        target_dim: Optional[int] = None,
                        **kwargs) -> None:
        """Add a relationship between interfaces
        
        For SDIM relationships, both source and target should be inputs.
        Relationships between inputs and outputs are only for dimension
        compatibility, not SDIM propagation.
        """
        # Validate interfaces exist
        all_names = ({inp.name for inp in self.input_definitions} |
                    {out.name for out in self.output_definitions})
        
        if source_name not in all_names:
            raise ValueError(f"Source interface '{source_name}' not found")
        if target_name not in all_names:
            raise ValueError(f"Target interface '{target_name}' not found")
        
        # Create relationship
        rel = DimensionRelationship(
            source_interface=source_name,
            target_interface=target_name,
            relation=relationship_type,
            source_dim=source_dim,
            target_dim=target_dim,
            **kwargs
        )
        
        self.relationships.append(rel)
    
    def get_input(self, name: str) -> Optional[InputDefinition]:
        """Get input definition by name"""
        for inp in self.input_definitions:
            if inp.name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> Optional[OutputDefinition]:
        """Get output definition by name"""
        for out in self.output_definitions:
            if out.name == name:
                return out
        return None
    
    def get_required_parameters(self) -> Dict[str, str]:
        """Get all parameters used in tiling expressions.
        
        Returns:
            Dict mapping parameter names to their usage context
        """
        params = {}
        
        # Extract from input definitions
        for inp in self.input_definitions:
            tiling_params = inp.get_tiling_parameters()
            for param_name, context in tiling_params.items():
                if param_name in params:
                    # Parameter used in multiple places
                    if params[param_name] != context:
                        params[param_name] = f"{params[param_name]}_and_{context}"
                else:
                    params[param_name] = f"{inp.name}_{context}"
        
        # Extract from output definitions
        for out in self.output_definitions:
            tiling_params = out.get_tiling_parameters()
            for param_name, context in tiling_params.items():
                if param_name in params:
                    # Parameter used in multiple places
                    if params[param_name] != context:
                        params[param_name] = f"{params[param_name]}_and_{out.name}_{context}"
                else:
                    params[param_name] = f"{out.name}_{context}"
        
        return params
    
    def has_weights(self) -> bool:
        """Check if kernel has weight inputs.
        
        Returns:
            True if any input has is_weight=True
        """
        return any(inp.is_weight for inp in self.input_definitions)
    
    def get_regular_inputs(self) -> List[InputDefinition]:
        """Get non-weight inputs.
        
        Returns:
            List of InputDefinitions where is_weight=False
        """
        return [inp for inp in self.input_definitions if not inp.is_weight]
    
    def get_weight_inputs(self) -> List[InputDefinition]:
        """Get weight inputs.
        
        Returns:
            List of InputDefinitions where is_weight=True
        """
        return [inp for inp in self.input_definitions if inp.is_weight]
    
    def create_model(self,
                    input_specs: Dict[str, Tuple[Shape, BaseDataType]],
                    output_specs: Dict[str, Tuple[Shape, BaseDataType]],
                    parameter_binding: Optional[Union[Dict[str, int], ParameterBinding]] = None) -> 'KernelModel':
        """Create runtime model with concrete datatypes
        
        Args:
            input_specs: Map of input names to (shape, datatype) tuples
            output_specs: Map of output names to (shape, datatype) tuples
            parameter_binding: Parameter values for block dimensions
            
        Returns:
            KernelModel instance with concrete types
            
        Raises:
            ValueError: If specs are missing or datatypes invalid
        """
        from .kernel_model import KernelModel
        
        # Convert parameter_binding if needed
        if isinstance(parameter_binding, dict):
            parameter_binding = ParameterBinding(parameter_binding)
        
        # Create input models
        input_models = []
        for inp_def in self.input_definitions:
            if inp_def.name not in input_specs:
                if not inp_def.optional:
                    raise ValueError(f"Missing required input specification for '{inp_def.name}'")
                continue
            
            shape, dtype = input_specs[inp_def.name]
            input_models.append(
                inp_def.create_model(shape, dtype, parameter_binding)
            )
        
        # Create output models
        output_models = []
        for out_def in self.output_definitions:
            if out_def.name not in output_specs:
                raise ValueError(f"Missing output specification for '{out_def.name}'")
            
            shape, dtype = output_specs[out_def.name]
            output_models.append(
                out_def.create_model(shape, dtype, parameter_binding)
            )
        
        return KernelModel(
            input_models=input_models,
            output_models=output_models,
            definition=self,
            parameter_binding=parameter_binding
        )
    
    def validate(self) -> List[str]:
        """Validate kernel definition consistency"""
        errors = []
        
        # Must have at least one input and output
        if not self.input_definitions:
            errors.append("Kernel must have at least one input")
        if not self.output_definitions:
            errors.append("Kernel must have at least one output")
        
        # Validate individual definitions
        for inp in self.input_definitions:
            inp_errors = inp.validate()
            errors.extend([f"Input '{inp.name}': {e}" for e in inp_errors])
        
        for out in self.output_definitions:
            out_errors = out.validate()
            errors.extend([f"Output '{out.name}': {e}" for e in out_errors])
        
        # Validate relationships reference existing interfaces
        all_names = ({inp.name for inp in self.input_definitions} |
                    {out.name for out in self.output_definitions})
        
        for rel in self.relationships:
            if rel.source_interface not in all_names:
                errors.append(f"Relationship source '{rel.source_interface}' not found")
            if rel.target_interface not in all_names:
                errors.append(f"Relationship target '{rel.target_interface}' not found")
        
        return errors
    
    def __repr__(self) -> str:
        return (f"KernelDefinition(name='{self.name}', "
                f"inputs={len(self.input_definitions)}, "
                f"outputs={len(self.output_definitions)})")
