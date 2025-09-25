############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Consolidated schema definitions for dataflow kernels.

This module contains all schema classes for defining kernel interfaces:
- InterfaceSchema: Base class for input/output interfaces
- InputSchema: Schema for input interfaces with optional streaming
- OutputSchema: Schema for output interfaces
- KernelSchema: Complete kernel definition with inputs, outputs, and relationships
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union, Dict, Any
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .constraint_types import DatatypeConstraintGroup, validate_datatype_against_constraints
from .relationships import DimensionRelationship, RelationType

# Type aliases for better clarity
TilingSpec = Sequence[Union[int, str]]


def _build_repr(class_name: str, name: str, **kwargs) -> str:
    """Build consistent repr strings for schema classes.
    
    Args:
        class_name: Name of the class
        name: Name field value
        **kwargs: Other fields to include (only non-default values shown)
    
    Returns:
        Formatted repr string
    """
    parts = [f"name='{name}'"]
    
    for key, value in kwargs.items():
        if value is None or (isinstance(value, (list, dict)) and not value):
            continue  # Skip None and empty collections
            
        if isinstance(value, bool) and value:
            parts.append(f"{key}=True")
        elif isinstance(value, list):
            parts.append(f"{key}={len(value)}")
        elif isinstance(value, str):
            parts.append(f"{key}='{value}'")
        else:
            parts.append(f"{key}={value}")
    
    return f"{class_name}({', '.join(parts)})"


class BaseSchema(ABC):
    """Base class for all Schema classes
    
    Schemas specify constraints, relationships, and validation rules.
    They define "what should be" rather than "what is".
    """
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate the schema for internal consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass


@dataclass
class InterfaceSchema(BaseSchema):
    """Base class for input/output interface schemas.
    
    Provides common fields and validation for all interface types.
    """
    
    name: str
    datatype_constraints: List[DatatypeConstraintGroup] = field(default_factory=list)
    block_tiling: Optional[TilingSpec] = None
    optional: bool = False
    
    # Node attribute name for datatype (e.g., "inputDataType", "outputDataType")
    datatype_attr: Optional[str] = None
    
    def validate_datatype(self, datatype: BaseDataType) -> bool:
        """Check if datatype satisfies constraints.
        
        Args:
            datatype: Datatype to validate
            
        Returns:
            True if datatype is valid, False otherwise
        """
        if not self.datatype_constraints:
            return True  # No constraints = allow any
        return validate_datatype_against_constraints(datatype, self.datatype_constraints)
    
    def validate(self) -> List[str]:
        """Validate schema consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        if not self.name:
            errors.append(f"{self.__class__.__name__} name cannot be empty")
        
        if self.block_tiling:
            # Validate tiling spec format
            for item in self.block_tiling:
                if not isinstance(item, (int, str)):
                    errors.append(
                        f"Invalid tiling item {item!r} in block_tiling - "
                        f"must be int or str"
                    )
                elif isinstance(item, int) and item <= 0:
                    errors.append(
                        f"Invalid tiling value {item} in block_tiling - "
                        f"must be positive"
                    )
        
        return errors
    
    def get_datatype_attr(self, index: int) -> str:
        """Get the nodeattr name for this interface's datatype.
        
        Args:
            index: Position of this interface in the kernel's input/output list
            
        Returns:
            The node attribute name to use for this interface's datatype
        """
        if self.datatype_attr:
            return self.datatype_attr
        
        # Generate default based on type and index
        # This will be overridden by subclasses
        raise NotImplementedError("Subclasses must implement get_datatype_attr")


@dataclass
class InputSchema(InterfaceSchema):
    """Schema for an input interface.
    
    Extends InterfaceSchema with input-specific fields like streaming
    configuration and weight marking.
    """
    
    # Input-specific fields
    stream_tiling: Optional[TilingSpec] = None
    is_weight: bool = False  # Explicitly mark weight inputs for FINN
    
    def validate(self) -> List[str]:
        """Validate input schema consistency."""
        errors = super().validate()
        
        if self.stream_tiling:
            # Validate stream tiling spec
            for item in self.stream_tiling:
                if not isinstance(item, (int, str)):
                    errors.append(
                        f"Invalid tiling item {item!r} in stream_tiling - "
                        f"must be int or str"
                    )
                elif isinstance(item, int) and item <= 0:
                    errors.append(
                        f"Invalid tiling value {item} in stream_tiling - "
                        f"must be positive"
                    )
        
        return errors
    
    def get_datatype_attr(self, index: int) -> str:
        """Get the nodeattr name for this input's datatype.
        
        Args:
            index: Position of this input in the kernel's input list
            
        Returns:
            The node attribute name to use for this input's datatype
        """
        if self.datatype_attr:
            return self.datatype_attr
        
        # Generate default name
        return f"input{index}Datatype"
    
    def __repr__(self) -> str:
        """String representation of InputSchema."""
        return _build_repr(
            "InputSchema",
            self.name,
            datatype_constraints=self.datatype_constraints,
            datatype_attr=self.datatype_attr,
            block_tiling=self.block_tiling,
            stream_tiling=self.stream_tiling,
            is_weight=self.is_weight,
            optional=self.optional
        )


@dataclass
class OutputSchema(InterfaceSchema):
    """Schema for an output interface.
    
    Outputs don't have configurable streaming - their rate is determined
    by the kernel computation.
    """
    
    def get_datatype_attr(self, index: int) -> str:
        """Get the nodeattr name for this output's datatype.
        
        Args:
            index: Position of this output in the kernel's output list
            
        Returns:
            The node attribute name to use for this output's datatype
        """
        if self.datatype_attr:
            return self.datatype_attr
        
        # Generate default name
        return f"output{index}Datatype"
    
    def __repr__(self) -> str:
        """String representation of OutputSchema."""
        return _build_repr(
            "OutputSchema",
            self.name,
            datatype_constraints=self.datatype_constraints,
            datatype_attr=self.datatype_attr,
            block_tiling=self.block_tiling,
            optional=self.optional
        )


@dataclass
class KernelSchema(BaseSchema):
    """Schema for a complete kernel definition.
    
    Defines a kernel with its input/output interfaces and relationships
    between dimensions. This is a pure schema definition - model creation
    happens in AutoHWCustomOp.
    """
    
    name: str
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    relationships: List[DimensionRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_input(self, input_schema: InputSchema) -> None:
        """Add an input schema.
        
        Args:
            input_schema: Input schema to add
            
        Raises:
            TypeError: If input_schema is not an InputSchema
            ValueError: If name already exists
        """
        if not isinstance(input_schema, InputSchema):
            raise TypeError(
                f"Expected InputSchema, got {type(input_schema).__name__}"
            )
        
        # Check for duplicate names
        existing_names = {inp.name for inp in self.inputs}
        if input_schema.name in existing_names:
            raise ValueError(f"Input '{input_schema.name}' already exists")
        
        self.inputs.append(input_schema)
    
    def add_output(self, output_schema: OutputSchema) -> None:
        """Add an output schema.
        
        Args:
            output_schema: Output schema to add
            
        Raises:
            TypeError: If output_schema is not an OutputSchema
            ValueError: If name already exists or conflicts with input
        """
        if not isinstance(output_schema, OutputSchema):
            raise TypeError(
                f"Expected OutputSchema, got {type(output_schema).__name__}"
            )
        
        # Check for name conflicts
        existing_input_names = {inp.name for inp in self.inputs}
        existing_output_names = {out.name for out in self.outputs}
        
        if output_schema.name in existing_input_names:
            raise ValueError(
                f"Name '{output_schema.name}' already used by an input"
            )
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
        """Add a relationship between interfaces.
        
        Args:
            source_name: Name of source interface
            target_name: Name of target interface
            relationship_type: Type of relationship
            source_dim: Optional dimension index for source
            target_dim: Optional dimension index for target
            **kwargs: Additional relationship parameters
            
        Raises:
            ValueError: If interfaces don't exist
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
        """Get input schema by name."""
        for inp in self.inputs:
            if inp.name == name:
                return inp
        return None
    
    def get_output(self, name: str) -> Optional[OutputSchema]:
        """Get output schema by name."""
        for out in self.outputs:
            if out.name == name:
                return out
        return None
    
    @property
    def has_weights(self) -> bool:
        """Check if kernel has weight inputs."""
        return any(inp.is_weight for inp in self.inputs)
    
    @property
    def regular_inputs(self) -> List[InputSchema]:
        """Get non-weight inputs."""
        return [inp for inp in self.inputs if not inp.is_weight]
    
    @property
    def weight_inputs(self) -> List[InputSchema]:
        """Get weight inputs."""
        return [inp for inp in self.inputs if inp.is_weight]
    
    def _validate_schemas(self, 
                         schemas: List[BaseSchema], 
                         schema_type: str) -> List[str]:
        """Validate a list of schemas.
        
        Args:
            schemas: List of schemas to validate
            schema_type: Type name for error messages
            
        Returns:
            List of validation errors
        """
        errors = []
        for idx, schema in enumerate(schemas):
            schema_errors = schema.validate()
            for error in schema_errors:
                errors.append(f"{schema_type} '{schema.name}': {error}")
        return errors
    
    def _validate_relationships(self) -> List[str]:
        """Validate all relationships reference existing interfaces.
        
        Returns:
            List of validation errors
        """
        errors = []
        all_names = ({inp.name for inp in self.inputs} |
                    {out.name for out in self.outputs})
        
        for rel in self.relationships:
            if rel.source_interface not in all_names:
                errors.append(
                    f"Relationship source '{rel.source_interface}' not found"
                )
            if rel.target_interface not in all_names:
                errors.append(
                    f"Relationship target '{rel.target_interface}' not found"
                )
        
        return errors
    
    def validate(self) -> List[str]:
        """Validate kernel schema consistency.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Must have at least one input and output
        if not self.inputs:
            errors.append("Kernel must have at least one input")
        if not self.outputs:
            errors.append("Kernel must have at least one output")
        
        # Validate individual schemas
        errors.extend(self._validate_schemas(self.inputs, "Input"))
        errors.extend(self._validate_schemas(self.outputs, "Output"))
        
        # Validate relationships
        errors.extend(self._validate_relationships())
        
        return errors
    
    def __repr__(self) -> str:
        """String representation of KernelSchema."""
        return _build_repr(
            "KernelSchema",
            self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            relationships=self.relationships,
            metadata=self.metadata,
            weights=self.weight_inputs
        )