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
- KernelSchema: Complete kernel definition with inputs and outputs
"""

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union, Dict, Any, TYPE_CHECKING
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .constraints import InterfaceConstraint
from .dimension_sources import DimensionSource
from .datatype_sources import DatatypeSource
from .relationships import InterfaceRelationship
from .types import DerivedDim, ScaledDim  # Backward compatibility

# Type aliases for better clarity
TilingSpec = Sequence[Union[int, str, DimensionSource]]


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

    The datatype field can be:
    - None: Use default naming (input{i}Datatype/output{i}Datatype), value from tensor context
    - str: Custom nodeattr name override, value from tensor context
    - DatatypeSource: Use default naming, value derived from other interfaces
    """

    name: str
    constraints: List[InterfaceConstraint] = field(default_factory=list)
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None
    optional: bool = False

    # Unified datatype field (replaces datatype_attr + datatype_source)
    datatype: Union[str, DatatypeSource, None] = None


@dataclass
class InputSchema(InterfaceSchema):
    """Schema for an input interface.

    Extends InterfaceSchema with input-specific fields like streaming
    configuration and weight marking.
    """

    # Input-specific fields
    is_weight: bool = False  # Explicitly mark weight inputs for FINN

    def __repr__(self) -> str:
        """String representation of InputSchema."""
        return _build_repr(
            "InputSchema",
            self.name,
            constraints=self.constraints,
            datatype=self.datatype,
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

    The datatype field allows output datatypes to be derived from input datatypes
    using patterns like DerivedDatatype, WidenedDatatype, etc.
    """

    def __repr__(self) -> str:
        """String representation of OutputSchema."""
        return _build_repr(
            "OutputSchema",
            self.name,
            constraints=self.constraints,
            datatype=self.datatype,
            block_tiling=self.block_tiling,
            stream_tiling=self.stream_tiling,
            optional=self.optional
        )


@dataclass
class KernelSchema:
    """Schema for a complete kernel definition.

    Defines a kernel with its input/output interfaces. This is a pure schema
    definition - model creation happens in AutoHWCustomOp.

    The relationships field allows cross-interface validation using patterns
    like DatatypesEqual, DimensionsEqual, or custom validation logic.
    """

    name: str
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    relationships: List[InterfaceRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate schema structure."""
        # Check unique input names
        input_names = [inp.name for inp in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Duplicate input names in kernel '{self.name}'")

        # Check unique output names
        output_names = [out.name for out in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Duplicate output names in kernel '{self.name}'")

        # Validate constraint targets match interface names
        for inp in self.inputs:
            for constraint in inp.constraints:
                if constraint.interface_name != inp.name:
                    raise ValueError(
                        f"Input '{inp.name}' constraint targets wrong interface '{constraint.interface_name}'"
                    )

        for out in self.outputs:
            for constraint in out.constraints:
                if constraint.interface_name != out.name:
                    raise ValueError(
                        f"Output '{out.name}' constraint targets wrong interface '{constraint.interface_name}'"
                    )

        # Validate DimensionSource references
        self._validate_dimension_references(input_names, output_names)

        # Validate relationship interface references
        self._validate_relationships(input_names, output_names)

    def _validate_dimension_references(self, input_names: List[str], output_names: List[str]) -> None:
        """Validate DimensionSource references in tiling specs.

        Args:
            input_names: List of input interface names
            output_names: List of output interface names

        Raises:
            ValueError: If dimension references are invalid

        Note:
            This validation is basic - only checks DerivedDim/ScaledDim which have
            source_interface attributes. Other DimensionSource types (SumDims, etc.)
            are validated at runtime when resolve() is called.
        """
        all_interface_names = set(input_names + output_names)

        # Check inputs
        for inp in self.inputs:
            for spec_name, spec in [("block_tiling", inp.block_tiling), ("stream_tiling", inp.stream_tiling)]:
                if spec is None:
                    continue

                for dim_idx, dim in enumerate(spec):
                    # Only validate DimensionSource types with source_interface attribute
                    if isinstance(dim, DimensionSource) and hasattr(dim, 'source_interface'):
                        # Validate source interface exists
                        if dim.source_interface not in all_interface_names:
                            raise ValueError(
                                f"Input '{inp.name}' {spec_name}[{dim_idx}] references "
                                f"unknown interface '{dim.source_interface}'"
                            )

                        # Prevent input-to-input dependencies (ordering issues)
                        if dim.source_interface in input_names:
                            raise ValueError(
                                f"Input '{inp.name}' {spec_name}[{dim_idx}] cannot reference "
                                f"another input '{dim.source_interface}' (dependency ordering not supported)"
                            )

        # Check outputs
        for out in self.outputs:
            for spec_name, spec in [("block_tiling", out.block_tiling), ("stream_tiling", out.stream_tiling)]:
                if spec is None:
                    continue

                for dim_idx, dim in enumerate(spec):
                    # Only validate DimensionSource types with source_interface attribute
                    if isinstance(dim, DimensionSource) and hasattr(dim, 'source_interface'):
                        # Validate source interface exists
                        if dim.source_interface not in all_interface_names:
                            raise ValueError(
                                f"Output '{out.name}' {spec_name}[{dim_idx}] references "
                                f"unknown interface '{dim.source_interface}'"
                            )

                        # Prevent output-to-output dependencies (dependency chains)
                        if dim.source_interface in output_names:
                            raise ValueError(
                                f"Output '{out.name}' {spec_name}[{dim_idx}] cannot reference "
                                f"another output '{dim.source_interface}' (dependency chains not allowed)"
                            )

    def _validate_relationships(self, input_names: List[str], output_names: List[str]) -> None:
        """Validate InterfaceRelationship references.

        Args:
            input_names: List of input interface names
            output_names: List of output interface names

        Raises:
            ValueError: If relationship references are invalid

        Note:
            This validation is basic - only checks that interface names exist.
            Full validation happens at runtime when check() is called.
        """
        all_interface_names = set(input_names + output_names)

        for rel_idx, rel in enumerate(self.relationships):
            # Check if relationship has interface_names attribute (most do)
            if hasattr(rel, 'interface_names'):
                for interface_name in rel.interface_names:
                    if interface_name not in all_interface_names:
                        raise ValueError(
                            f"Relationship {rel_idx} ({type(rel).__name__}) references "
                            f"unknown interface '{interface_name}'"
                        )

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
        code because they control datatype constraints.

        Returns:
            List of protected attribute names as strings
        """
        protected = []

        # Add input datatype attrs (custom names or defaults)
        for i, inp in enumerate(self.inputs):
            if isinstance(inp.datatype, str):
                protected.append(inp.datatype)  # Custom name
            else:
                protected.append(f"input{i}Datatype")  # Default

        # Add output datatype attrs (custom names or defaults)
        for i, out in enumerate(self.outputs):
            if isinstance(out.datatype, str):
                protected.append(out.datatype)  # Custom name
            else:
                protected.append(f"output{i}Datatype")  # Default

        return protected

    def get_datatype_attr(self, index: int, is_input: bool = True) -> str:
        """Get the nodeattr name for an interface's datatype.

        Args:
            index: Index of the interface
            is_input: True for input, False for output

        Returns:
            Node attribute name (custom or default)
        """
        interface = (self.inputs[index] if is_input else self.outputs[index])

        # Custom name (str) takes precedence
        if isinstance(interface.datatype, str):
            return interface.datatype

        # Default naming
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