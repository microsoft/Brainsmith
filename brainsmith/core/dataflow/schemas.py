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
from typing import List, Optional, Sequence, Union, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .constraints import InterfaceConstraint, InterfaceRelationship

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


@dataclass
class InterfaceSchema:
    """Base class for input/output interface schemas.

    Provides common fields and validation for all interface types.
    """

    name: str
    constraints: List[InterfaceConstraint] = field(default_factory=list)
    block_tiling: Optional[TilingSpec] = None
    optional: bool = False

    # Node attribute name for datatype (e.g., "inputDataType", "outputDataType")
    datatype_attr: Optional[str] = None

    def add_constraint(self, constraint: InterfaceConstraint) -> None:
        """Add a constraint to this interface.

        Args:
            constraint: InterfaceConstraint to add

        Raises:
            ValueError: If constraint interface name doesn't match schema name
        """
        if constraint.interface_name != self.name:
            raise ValueError(
                f"Constraint interface '{constraint.interface_name}' must match schema name '{self.name}'"
            )
        self.constraints.append(constraint)

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
    
    def __repr__(self) -> str:
        """String representation of InputSchema."""
        return _build_repr(
            "InputSchema",
            self.name,
            constraints=self.constraints,
            datatype_attr=self.datatype_attr,
            block_tiling=self.block_tiling,
            stream_tiling=self.stream_tiling,
            is_weight=self.is_weight,
            optional=self.optional
        )


@dataclass
class OutputSchema(InterfaceSchema):
    """Schema for an output interface.

    Outputs can have explicit streaming (stream_tiling) or derived streaming
    (computed from relationships or default derivation logic).
    """

    # Output-specific fields
    stream_tiling: Optional[TilingSpec] = None  # None = derive from relationships/inputs

    def __repr__(self) -> str:
        """String representation of OutputSchema."""
        return _build_repr(
            "OutputSchema",
            self.name,
            constraints=self.constraints,
            datatype_attr=self.datatype_attr,
            block_tiling=self.block_tiling,
            stream_tiling=self.stream_tiling,
            optional=self.optional
        )


@dataclass
class KernelSchema:
    """Schema for a complete kernel definition.

    Defines a kernel with its input/output interfaces and cross-interface
    relationships. This is a pure schema definition - model creation
    happens in AutoHWCustomOp.
    """

    name: str
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    relationships: List[InterfaceRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_relationship(self, relationship: InterfaceRelationship) -> None:
        """Add a cross-interface relationship to this kernel.

        Args:
            relationship: InterfaceRelationship to add
        """
        self.relationships.append(relationship)
    
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
    def protected_attr_names(self) -> List[str]:
        """Get all node attribute names that are protected (set by tensor context).

        These are node attribute names that should not be modified by external
        code because they control datatype constraints (e.g., 'ActVal', 'WeightType').

        Returns:
            List of protected attribute names as strings
        """
        inp_attrs = [inp.datatype_attr for inp in self.inputs if inp.datatype_attr]
        out_attrs = [out.datatype_attr for out in self.outputs if out.datatype_attr]
        return inp_attrs + out_attrs    

    def get_datatype_attr(self, index: int, is_input: bool = True) -> str:
        """Get the nodeattr name for an interface's datatype."""
        interface = (self.inputs[index] if is_input else self.outputs[index])
        if interface.datatype_attr is not None:
            return interface.datatype_attr
        if is_input:
            return f"input{index}Datatype"
        else:
            return f"output{index}Datatype"

    def __repr__(self) -> str:
        """String representation of KernelSchema."""
        return _build_repr(
            "KernelSchema",
            self.name,
            inputs=self.inputs,
            outputs=self.outputs,
            relationships=self.relationships,
            metadata=self.metadata
        )