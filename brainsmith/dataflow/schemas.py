############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Consolidated schema definitions for dataflow kernels.

Schemas define STRUCTURE, not STORAGE. They specify:
- Interface definitions (inputs, outputs)
- Tiling templates (block_tiling, stream_tiling)
- Validation rules (constraints, relationships)

Schemas are stateless - they never define what gets stored in nodeattrs.
Storage decisions are implementation details of operator classes.

Key classes:
- InterfaceSchema: Base class for input/output interfaces
- InputSchema: Schema for input interfaces
- OutputSchema: Schema for output interfaces
- KernelSchema: Complete kernel definition with inputs and outputs
"""

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union, Dict, Any, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

from qonnx.core.datatype import BaseDataType
from .constraints import InterfaceConstraint
from .dimension_sources import DimensionSource
from .datatype_sources import DatatypeSource
from .relationships import InterfaceRelationship
from .types import FULL_DIM

if TYPE_CHECKING:
    from .models import KernelModel

logger = logging.getLogger(__name__)

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

    Defines kernel STRUCTURE:
    - Input/output interfaces with tiling templates
    - Validation rules (constraints, relationships)
    - Internal datatype derivation patterns

    Does NOT define STORAGE:
    - Shapes are extracted from ModelWrapper or computed from templates
    - Only datatypes and user parameters persist in nodeattrs
    - KernelOp handles storage implementation

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

    def validate_model(
        self,
        model: 'KernelModel',
        param_getter: Callable[[str], Any]
    ) -> None:
        """Validate a KernelModel against this schema's constraints and relationships.

        This is where validation logic belongs - with the schema definition that
        specifies the rules. The schema knows what constraints and relationships
        must hold, so it should validate them.

        Args:
            model: KernelModel to validate
            param_getter: Function to retrieve nodeattr values (for constraint checking)

        Raises:
            ValueError: If any constraint or relationship validation fails

        Example:
            >>> schema = KernelSchema(name="LayerNorm", ...)
            >>> model = builder.build(ctx)
            >>> schema.validate_model(model, get_nodeattr)  # Validates model
        """
        logger.debug(f"Validating KernelModel against schema '{self.name}'")

        # Validate input constraints
        for i, input_model in enumerate(model.inputs):
            schema = self.inputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    interface_name=schema.name,
                    interface_model=input_model,
                    constraints=schema.constraints,
                    param_getter=param_getter
                )

        # Validate output constraints
        for i, output_model in enumerate(model.outputs):
            schema = self.outputs[i]
            if schema.constraints:
                self._validate_interface_constraints(
                    interface_name=schema.name,
                    interface_model=output_model,
                    constraints=schema.constraints,
                    param_getter=param_getter
                )

        # Validate relationships (cross-interface)
        if self.relationships:
            logger.debug(f"Validating {len(self.relationships)} relationships")
            for relationship in self.relationships:
                error = relationship.check(model, param_getter)
                if error:
                    raise ValueError(f"Relationship validation failed: {error}")

    def _validate_interface_constraints(
        self,
        interface_name: str,
        interface_model: Any,
        constraints: List[InterfaceConstraint],
        param_getter: Callable[[str], Any]
    ) -> None:
        """Validate constraints for a single interface.

        Args:
            interface_name: Interface name for error messages
            interface_model: InputModel or OutputModel to validate
            constraints: List of InterfaceConstraint instances to check
            param_getter: Function to retrieve nodeattr values

        Raises:
            ValueError: If any constraint fails
        """
        for constraint in constraints:
            error = constraint.check(interface_model, param_getter)
            if error:
                raise ValueError(
                    f"Constraint validation failed for '{interface_name}': {error}"
                )

    def get_nodeattr_types(self) -> Dict[str, tuple]:
        """Generate nodeattr registry from schema.

        Schemas define STRUCTURE, not STORAGE. Returns only attributes
        that need persistence:
        - Datatypes (for interfaces and internals)
        - User parameters (SIMD, PE, etc.)

        Shapes are NEVER stored in nodeattrs. They are either:
        - Tensor shapes: extracted from ModelWrapper (ONNX graph)
        - Block/stream shapes: computed from schema templates

        Returns:
            Dict mapping nodeattr name to (type, required, default_value)
            Format: {"attrName": ("i"|"s"|"f", True|False, default)}
        """
        attrs = {}

        # ================================================================
        # 1. Interface Datatypes (datatypes only, NO shapes)
        # ================================================================

        for i in range(len(self.inputs)):
            attrs[f"input{i}Datatype"] = ("s", False, "")

        for i in range(len(self.outputs)):
            attrs[f"output{i}Datatype"] = ("s", False, "")

        # ================================================================
        # 2. Internal Datatypes
        # ================================================================

        for internal_name in self.internal_datatypes.keys():
            attr_name = f"{internal_name}Datatype"
            attrs[attr_name] = ("s", False, "")

        # ================================================================
        # 3. User Parameters (template params from tiling specs)
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
