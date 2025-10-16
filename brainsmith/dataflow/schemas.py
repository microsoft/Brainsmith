############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Consolidated schema definitions for dataflow kernels.

This module contains all schema classes for defining kernel interfaces:
- InterfaceSchema: Base class for input/output interfaces
- InputSchema: Schema for input interfaces
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
from .types import FULL_DIM

# Type aliases for better clarity
TilingSpec = Sequence[Union[int, str, type(FULL_DIM), DimensionSource]]


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

    @property
    def tiling_attrs(self) -> List[str]:
        """Extract unique template parameter names from tiling specs."""
        params = set()
        for spec in (self.block_tiling, self.stream_tiling):
            if spec:
                params.update(dim for dim in spec if isinstance(dim, str))
        return list(params)


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
    definition - model creation happens in KernelOp.

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

    def get_nodeattr_types(self) -> Dict[str, tuple]:
        """Generate complete nodeattr registry from schema.

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # ================================================================
        # 1. Interface Datatypes (protected, _ prefix)
        # ================================================================

        for i in range(len(self.inputs)):
            attrs[f"_input{i}Datatype"] = ("s", False, "")

        for i in range(len(self.outputs)):
            attrs[f"_output{i}Datatype"] = ("s", False, "")

        # ================================================================
        # 2. Internal Datatypes (prefix "_")
        # ================================================================

        for internal_name in self.internal_datatypes.keys():
            attr_name = f"_{internal_name}Datatype"
            attrs[attr_name] = ("s", False, "")

        # ================================================================
        # 3. Protected Shape Attributes (prefix "_")
        # ================================================================

        for i in range(len(self.inputs)):
            attrs[f"_input{i}TensorShape"] = ("ints", False, [])
            attrs[f"_input{i}BlockShape"] = ("ints", False, [])
            attrs[f"_input{i}StreamShape"] = ("ints", False, [])

        for i in range(len(self.outputs)):
            attrs[f"_output{i}TensorShape"] = ("ints", False, [])
            attrs[f"_output{i}BlockShape"] = ("ints", False, [])
            attrs[f"_output{i}StreamShape"] = ("ints", False, [])

        # ================================================================
        # 4. Template Parameters (user-configurable, no prefix)
        # ================================================================

        template_params = self._extract_template_params()
        for param in template_params:
            attrs[param] = ("i", True, 1)

        return attrs

    def _extract_template_params(self) -> set:
        """Extract unique template parameter names from tiling specs.

        Returns:
            Set of template parameter names (strings found in tiling specs)
        """
        params = set()
        for interface in self.inputs + self.outputs:
            params.update(interface.tiling_attrs)
        return params

    @property
    def protected_attr_names(self) -> set:
        """Attributes that cannot be modified by users.

        Protected attributes are managed by refresh_df_model() and
        internal resolution logic.

        Returns:
            Set of protected attribute names
        """
        protected = set()

        # All attributes with "_" prefix
        for i in range(len(self.inputs)):
            protected.add(f"_input{i}TensorShape")
            protected.add(f"_input{i}BlockShape")
            protected.add(f"_input{i}StreamShape")
            protected.add(f"_input{i}Datatype")

        for i in range(len(self.outputs)):
            protected.add(f"_output{i}TensorShape")
            protected.add(f"_output{i}BlockShape")
            protected.add(f"_output{i}StreamShape")
            protected.add(f"_output{i}Datatype")

        # Internal datatypes
        for internal_name in self.internal_datatypes.keys():
            protected.add(f"_{internal_name}Datatype")

        return protected

    def get_datatype_attr(self, index: int, is_input: bool = True) -> str:
        """Get nodeattr name for interface datatype.

        Args:
            index: Interface index (0-based)
            is_input: True for inputs, False for outputs

        Returns:
            Nodeattr name (e.g., "_input0Datatype", "_output2Datatype")

        Raises:
            IndexError: If index out of range
        """
        if is_input:
            if index >= len(self.inputs):
                raise IndexError(
                    f"Input index {index} out of range (have {len(self.inputs)})"
                )
            return f"_input{index}Datatype"
        else:
            if index >= len(self.outputs):
                raise IndexError(
                    f"Output index {index} out of range (have {len(self.outputs)})"
                )
            return f"_output{index}Datatype"
    
