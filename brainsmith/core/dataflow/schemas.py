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

# Type aliases for better clarity
TilingSpec = Sequence[Union[int, str, DimensionSource]]


@dataclass
class InterfaceSchema:
    """Base class for input/output interface schemas.

    Provides common fields and validation for all interface types.
    Inputs always get their datatypes from the ONNX graph.
    Outputs can derive datatypes from inputs or internal datatypes.
    """

    name: str
    block_tiling: Optional[TilingSpec] = None
    stream_tiling: Optional[TilingSpec] = None
    constraints: List[InterfaceConstraint] = field(default_factory=list)
    optional: bool = False


@dataclass
class InputSchema(InterfaceSchema):
    """Schema for an input interface."""

    is_weight: bool = False  # Explicitly mark weight inputs for FINN


@dataclass
class OutputSchema(InterfaceSchema):
    """Schema for an output interface.

    The datatype field specifies how the output datatype is determined:
    - None: Use datatype from ONNX graph (pass-through/validation only)
    - DatatypeSource: Derive datatype from inputs or internal datatypes
    """

    datatype: Optional[DatatypeSource] = None


@dataclass
class KernelSchema:
    """Schema for a complete kernel definition.

    Defines a kernel with its input/output interfaces. This is a pure schema
    definition - model creation happens in AutoHWCustomOp.

    The relationships field allows cross-interface validation using patterns
    like DatatypesEqual, DimensionsEqual, or custom validation logic.

    Internal datatypes represent intermediate computation datatypes not attached
    to ONNX tensors (e.g., accumulators, bias values). They are derived from
    inputs or other internals using DatatypeSource patterns.
    """

    name: str
    inputs: List[InputSchema] = field(default_factory=list)
    outputs: List[OutputSchema] = field(default_factory=list)
    internal_datatypes: Dict[str, DatatypeSource] = field(default_factory=dict)
    relationships: List[InterfaceRelationship] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate schema structure."""
        # Validate constraint targets match interface names
        self.validate()

    def validate(self) -> None:
        """Validate the schema structure."""

        # Check unique input names
        input_names = [inp.name for inp in self.inputs]
        if len(input_names) != len(set(input_names)):
            raise ValueError(f"Duplicate input names in kernel '{self.name}'")

        # Check unique output names
        output_names = [out.name for out in self.outputs]
        if len(output_names) != len(set(output_names)):
            raise ValueError(f"Duplicate output names in kernel '{self.name}'")

        # Check internal datatype names don't conflict with interfaces
        all_interface_names = set(input_names + output_names)
        for internal_name in self.internal_datatypes.keys():
            if internal_name in all_interface_names:
                raise ValueError(
                    f"Internal datatype '{internal_name}' conflicts with interface name in kernel '{self.name}'"
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
        """Get all node attribute names that are protected (set by datatype resolution).

        These are node attribute names that should not be modified by external
        code because they control datatype constraints.

        Returns:
            List of protected attribute names as strings
        """
        protected = []

        # Add input datatype attrs (always use default naming)
        for i in range(len(self.inputs)):
            protected.append(f"input{i}Datatype")

        # Add output datatype attrs (always use default naming)
        for i in range(len(self.outputs)):
            protected.append(f"output{i}Datatype")

        # Add internal datatype attrs
        for internal_name in self.internal_datatypes.keys():
            protected.append(f"{internal_name}Datatype")

        return protected

    def get_datatype_attr(self, index: int, is_input: bool = True) -> str:
        """Get the nodeattr name for an interface's datatype.

        Always uses default naming pattern: input{i}Datatype or output{i}Datatype.

        Args:
            index: Index of the interface
            is_input: True for input, False for output

        Returns:
            Node attribute name following default naming pattern
        """
        if is_input:
            return f"input{index}Datatype"
        else:
            return f"output{index}Datatype"

    def _extract_template_params(self, spec: Optional[TilingSpec]) -> List[str]:
        """Extract template parameter names from a tiling spec.

        Template params are string values that are not ":" (copy dimension).

        Args:
            spec: Tiling specification (block or stream)

        Returns:
            List of unique template parameter names
        """
        if spec is None:
            return []

        params = []
        for dim in spec:
            if isinstance(dim, str) and dim != ":":
                params.append(dim)

        return params

    def get_nodeattr_types(self) -> Dict[str, tuple]:
        """Generate node attribute type registry from schema.

        Extracts:
        1. Datatype attributes from inputs/outputs/internals (string nodeattrs)
        2. Template parameters from tiling specs (integer nodeattrs)

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # 1. Add datatype attributes (string type)
        # Input datatypes
        for i, inp in enumerate(self.inputs):
            attr_name = self.get_datatype_attr(i, is_input=True)
            attrs[attr_name] = ("s", True, "")  # String, required, empty default

        # Output datatypes
        for i, out in enumerate(self.outputs):
            attr_name = self.get_datatype_attr(i, is_input=False)
            attrs[attr_name] = ("s", True, "")  # String, required, empty default

        # Internal datatypes
        for internal_name in self.internal_datatypes.keys():
            attr_name = f"{internal_name}Datatype"
            attrs[attr_name] = ("s", True, "")  # String, required, empty default

        # 2. Extract template parameters from tiling specs (integer type)
        template_params = set()

        for inp in self.inputs:
            template_params.update(self._extract_template_params(inp.block_tiling))
            template_params.update(self._extract_template_params(inp.stream_tiling))

        for out in self.outputs:
            template_params.update(self._extract_template_params(out.block_tiling))
            template_params.update(self._extract_template_params(out.stream_tiling))

        for param in template_params:
            attrs[param] = ("i", True, 1)  # Integer, required, default 1

        return attrs
