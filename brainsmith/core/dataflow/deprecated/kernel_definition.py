############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel schema with separate input/output schemas"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Tuple, Union, TYPE_CHECKING
from .base import BaseSchema
from .types import Shape
from .qonnx_types import BaseDataType
from .input_definition import InputSchema
from .output_definition import OutputSchema
from .relationships import DimensionRelationship, RelationType


@dataclass
class KernelSchema(BaseSchema):
    """Schema of a kernel with separate input/output interfaces
    
    Key features:
    - Separate inputs and outputs
    - Clean API with add_input() and add_output()
    - Relationships primarily between inputs
    - Pure algorithmic abstraction independent of code generation
    """
    
    name: str
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    relationships: List[DimensionRelationship] = field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
    
    def add_input(self, input_schema: InputSchema) -> None:
        """Add an input schema"""
        # Check for duplicate names
        existing_names = {inp.name for inp in self.inputs}
        if input_schema.name in existing_names:
            raise ValueError(f"Input '{input_schema.name}' already exists")
        
        self.inputs.append(input_schema)
    
    def add_output(self, output_schema: OutputSchema) -> None:
        """Add an output schema"""
        # Check for duplicate names
        existing_input_names = {inp.name for inp in self.inputs}
        existing_output_names = {out.name for out in self.outputs}
        
        if output_schema.name in existing_input_names:
            raise ValueError(f"Name '{output_schema.name}' already used by an input")
        if output_schema.name in existing_output_names:
            raise ValueError(f"Output '{output_schema.name}' already exists")
        
        self.outputs.append(output_schema)
    
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
        all_names = ({inp.name for inp in self.inputs} |
                    {out.name for out in self.outputs})
        
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
    
    def get_input(self, name: str) -> Optional[InputSchema]:
        """Get input schema by name"""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> Optional[OutputSchema]:
        """Get output schema by name"""
        for out in self.outputs:
            if out.name == name:
                return out
        return None
    
    
    def has_weights(self) -> bool:
        """Check if kernel has weight inputs.
        
        Returns:
            True if any input has is_weight=True
        """
        return any(inp.is_weight for inp in self.inputs)
    
    def get_regular_inputs(self) -> List[InputSchema]:
        """Get non-weight inputs.
        
        Returns:
            List of InputSchemas where is_weight=False
        """
        return [inp for inp in self.inputs if not inp.is_weight]
    
    def get_weight_inputs(self) -> List[InputSchema]:
        """Get weight inputs.
        
        Returns:
            List of InputSchemas where is_weight=True
        """
        return [inp for inp in self.inputs if inp.is_weight]
    
    # Note: Model creation has been moved to AutoHWCustomOp factory methods
    # This keeps KernelSchema as a pure container of schema definitions
    
    def validate(self) -> List[str]:
        """Validate kernel schema consistency"""
        errors = []
        
        # Must have at least one input and output
        if not self.inputs:
            errors.append("Kernel must have at least one input")
        if not self.outputs:
            errors.append("Kernel must have at least one output")
        
        # Validate individual schemas
        for inp in self.inputs:
            inp_errors = inp.validate()
            errors.extend([f"Input '{inp.name}': {e}" for e in inp_errors])
        
        for out in self.outputs:
            out_errors = out.validate()
            errors.extend([f"Output '{out.name}': {e}" for e in out_errors])
        
        # Validate relationships reference existing interfaces
        all_names = ({inp.name for inp in self.inputs} |
                    {out.name for out in self.outputs})
        
        for rel in self.relationships:
            if rel.source_interface not in all_names:
                errors.append(f"Relationship source '{rel.source_interface}' not found")
            if rel.target_interface not in all_names:
                errors.append(f"Relationship target '{rel.target_interface}' not found")
        
        return errors
    
    def __repr__(self) -> str:
        parts = [f"name='{self.name}'"]
        
        parts.append(f"inputs={len(self.inputs)}")
        parts.append(f"outputs={len(self.outputs)}")
        
        if self.relationships:
            parts.append(f"relationships={len(self.relationships)}")
        
        if self.metadata:
            parts.append(f"metadata={len(self.metadata)}")
        
        weight_count = len(self.get_weight_inputs())
        if weight_count > 0:
            parts.append(f"weights={weight_count}")
        
        return f"KernelSchema({', '.join(parts)})"
