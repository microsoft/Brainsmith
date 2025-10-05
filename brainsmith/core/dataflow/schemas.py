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
from .relationships import DimensionRelationship, RelationType, ValidationResult, ConstraintViolation

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
    def validate_structure(self) -> ValidationResult:
        """Validate the schema structure for internal consistency.

        This validates the schema definition itself, not a model instance.
        Used during schema construction to catch errors early.

        Returns:
            ValidationResult with any violations found
        """
        pass

    @abstractmethod
    def validate(self, model) -> ValidationResult:
        """Validate a model instance against this schema.

        Args:
            model: Model instance to validate

        Returns:
            ValidationResult with any constraint violations
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

    def validate_structure(self) -> ValidationResult:
        """Validate schema structure (internal consistency check).

        This validates the schema definition itself, not a model instance.
        Used during schema construction to catch errors early.

        Returns:
            ValidationResult with any violations found
        """
        violations = []

        if not self.name:
            violations.append(ConstraintViolation(
                constraint_type="interface_structure",
                message=f"{self.__class__.__name__} name cannot be empty",
                severity="error"
            ))

        if self.block_tiling:
            # Validate tiling spec format
            for i, item in enumerate(self.block_tiling):
                if not isinstance(item, (int, str)):
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_template",
                        message=f"Invalid tiling item {item!r} at position {i} in block_tiling - must be int or str",
                        severity="error"
                    ))
                elif isinstance(item, int) and item <= 0:
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_value",
                        message=f"Invalid tiling value {item} at position {i} in block_tiling - must be positive",
                        severity="error"
                    ))

        return ValidationResult(violations=violations)

    def validate(self, model) -> ValidationResult:
        """Validate a model instance against this schema.

        Args:
            model: InterfaceModel (InputModel or OutputModel) to validate

        Returns:
            ValidationResult with any constraint violations
        """
        from .models import InputModel, OutputModel

        violations = []

        # Validate name matches
        if model.name != self.name:
            violations.append(ConstraintViolation(
                constraint_type="interface_name",
                message=f"Model name '{model.name}' doesn't match schema name '{self.name}'",
                severity="error",
                expected=self.name,
                actual=model.name
            ))

        # Validate datatype constraints
        if self.datatype_constraints:
            if not self.validate_datatype(model.datatype):
                violations.append(ConstraintViolation(
                    constraint_type="datatype_constraint",
                    message=f"Datatype {model.datatype.get_canonical_name()} violates constraints for interface '{self.name}'",
                    severity="error",
                    expected=[str(c) for c in self.datatype_constraints],
                    actual=model.datatype.get_canonical_name(),
                    details={"constraints": [str(c) for c in self.datatype_constraints]}
                ))

        # Validate dimensions match block_tiling template if defined
        if self.block_tiling:
            expected_ndim = len(self.block_tiling)
            actual_ndim = len(model.block_shape)
            if expected_ndim != actual_ndim:
                violations.append(ConstraintViolation(
                    constraint_type="dimension_count",
                    message=f"Interface '{self.name}': block dimensions count mismatch",
                    severity="error",
                    expected=expected_ndim,
                    actual=actual_ndim
                ))

        return ValidationResult(violations=violations)
    
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
    
    def validate_structure(self) -> ValidationResult:
        """Validate input schema structure (internal consistency check)."""
        # Get base violations
        result = super().validate_structure()
        violations = list(result.violations)

        if self.stream_tiling:
            # Validate stream tiling spec
            for i, item in enumerate(self.stream_tiling):
                if not isinstance(item, (int, str)):
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_template",
                        message=f"Invalid tiling item {item!r} at position {i} in stream_tiling - must be int or str",
                        severity="error"
                    ))
                elif isinstance(item, int) and item <= 0:
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_value",
                        message=f"Invalid tiling value {item} at position {i} in stream_tiling - must be positive",
                        severity="error"
                    ))

        return ValidationResult(violations=violations)

    def validate(self, model) -> ValidationResult:
        """Validate an InputModel against this schema.

        Args:
            model: InputModel to validate

        Returns:
            ValidationResult with any constraint violations
        """
        from .models import InputModel

        # Get base interface violations
        result = super().validate(model)
        violations = list(result.violations)

        # Type check
        if not isinstance(model, InputModel):
            violations.append(ConstraintViolation(
                constraint_type="model_type",
                message=f"Expected InputModel, got {type(model).__name__}",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # Validate stream dimensions if stream_tiling is defined
        if self.stream_tiling:
            expected_ndim = len(self.stream_tiling)
            actual_ndim = len(model.stream_shape)
            if expected_ndim != actual_ndim:
                violations.append(ConstraintViolation(
                    constraint_type="dimension_count",
                    message=f"Input '{self.name}': stream dimensions count mismatch",
                    severity="error",
                    expected=expected_ndim,
                    actual=actual_ndim
                ))

            # Validate stream shape don't exceed block shape
            for i, (stream_dim, block_dim) in enumerate(zip(model.stream_shape, model.block_shape)):
                if stream_dim > block_dim:
                    violations.append(ConstraintViolation(
                        constraint_type="streaming_constraint",
                        message=f"Input '{self.name}': stream_shape[{i}] exceeds block_shape[{i}]",
                        severity="error",
                        expected=f"<= {block_dim}",
                        actual=stream_dim
                    ))

        # Validate is_weight flag matches
        if model.is_weight != self.is_weight:
            violations.append(ConstraintViolation(
                constraint_type="weight_flag",
                message=f"Input '{self.name}': is_weight flag mismatch",
                severity="warning",
                expected=self.is_weight,
                actual=model.is_weight
            ))

        return ValidationResult(violations=violations)
    
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
    
    def validate(self, model) -> ValidationResult:
        """Validate an OutputModel against this schema.

        Args:
            model: OutputModel to validate

        Returns:
            ValidationResult with any constraint violations
        """
        from .models import OutputModel

        # Get base interface violations
        result = super().validate(model)
        violations = list(result.violations)

        # Type check
        if not isinstance(model, OutputModel):
            violations.append(ConstraintViolation(
                constraint_type="model_type",
                message=f"Expected OutputModel, got {type(model).__name__}",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # NOTE: streaming_rate is now computed by KernelModel, not stored in OutputModel

        return ValidationResult(violations=violations)

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




    def _validate_schema_structures(self,
                         schemas: List[BaseSchema],
                         schema_type: str) -> List[ConstraintViolation]:
        """Validate schema structures (internal consistency check).

        Args:
            schemas: List of schemas to validate
            schema_type: Type name for error messages

        Returns:
            List of constraint violations
        """
        violations = []
        for idx, schema in enumerate(schemas):
            result = schema.validate_structure()
            for violation in result.violations:
                # Add context to the violation message
                new_violation = ConstraintViolation(
                    constraint_type=violation.constraint_type,
                    message=f"{schema_type} '{schema.name}': {violation.message}",
                    severity=violation.severity,
                    details=violation.details
                )
                violations.append(new_violation)
        return violations

    def _validate_relationship_structure(self) -> List[ConstraintViolation]:
        """Validate relationship definitions reference existing interfaces.

        Returns:
            List of constraint violations
        """
        violations = []
        all_names = ({inp.name for inp in self.inputs} |
                    {out.name for out in self.outputs})

        for rel in self.relationships:
            if rel.source_interface not in all_names:
                violations.append(ConstraintViolation(
                    constraint_type="relationship_reference",
                    message=f"Relationship source '{rel.source_interface}' not found",
                    severity="error",
                    details={"available_interfaces": list(all_names)}
                ))
            if rel.target_interface not in all_names:
                violations.append(ConstraintViolation(
                    constraint_type="relationship_reference",
                    message=f"Relationship target '{rel.target_interface}' not found",
                    severity="error",
                    details={"available_interfaces": list(all_names)}
                ))

        return violations

    def validate_structure(self) -> ValidationResult:
        """Validate kernel schema structure (internal consistency check).

        This validates the schema definition itself, not a model instance.
        Used during schema construction to catch errors early.

        Returns:
            ValidationResult with any violations found
        """
        violations = []

        # Must have at least one input and output
        if not self.inputs:
            violations.append(ConstraintViolation(
                constraint_type="schema_structure",
                message="Kernel must have at least one input",
                severity="error"
            ))
        if not self.outputs:
            violations.append(ConstraintViolation(
                constraint_type="schema_structure",
                message="Kernel must have at least one output",
                severity="error"
            ))

        # Validate individual schema structures
        violations.extend(self._validate_schema_structures(self.inputs, "Input"))
        violations.extend(self._validate_schema_structures(self.outputs, "Output"))

        # Validate relationship structure
        violations.extend(self._validate_relationship_structure())

        return ValidationResult(violations=violations)

    def validate(self, model) -> ValidationResult:
        """Validate a KernelModel against this schema.

        Args:
            model: KernelModel to validate

        Returns:
            ValidationResult with any constraint violations
        """
        from .models import KernelModel

        violations = []

        # Type check
        if not isinstance(model, KernelModel):
            violations.append(ConstraintViolation(
                constraint_type="model_type",
                message=f"Expected KernelModel, got {type(model).__name__}",
                severity="error"
            ))
            return ValidationResult(violations=violations)

        # Validate name matches
        if model.name != self.name:
            violations.append(ConstraintViolation(
                constraint_type="kernel_name",
                message=f"Model name '{model.name}' doesn't match schema name '{self.name}'",
                severity="error",
                expected=self.name,
                actual=model.name
            ))

        # Validate input/output counts
        if len(model.inputs) != len(self.inputs):
            violations.append(ConstraintViolation(
                constraint_type="interface_count",
                message=f"Input count mismatch",
                severity="error",
                expected=len(self.inputs),
                actual=len(model.inputs)
            ))

        if len(model.outputs) != len(self.outputs):
            violations.append(ConstraintViolation(
                constraint_type="interface_count",
                message=f"Output count mismatch",
                severity="error",
                expected=len(self.outputs),
                actual=len(model.outputs)
            ))

        # Validate each input model against its schema
        for i, (input_schema, input_model) in enumerate(zip(self.inputs, model.inputs)):
            result = input_schema.validate(input_model)
            violations.extend(result.violations)

        # Validate each output model against its schema
        for i, (output_schema, output_model) in enumerate(zip(self.outputs, model.outputs)):
            result = output_schema.validate(output_model)
            violations.extend(result.violations)

        # Validate dimension relationships
        if self.relationships:
            interface_map = {inp.name: inp for inp in model.inputs}
            interface_map.update({out.name: out for out in model.outputs})

            for rel in self.relationships:
                try:
                    is_satisfied = rel.evaluate(interface_map)
                    if not is_satisfied:
                        violations.append(ConstraintViolation(
                            constraint_type="dimension_relationship",
                            message=f"Relationship constraint violated: {rel.describe()}",
                            severity="error",
                            description=rel.describe()
                        ))
                except (ValueError, KeyError) as e:
                    violations.append(ConstraintViolation(
                        constraint_type="relationship_evaluation",
                        message=f"Error evaluating relationship: {str(e)}",
                        severity="error",
                        description=rel.describe()
                    ))

        return ValidationResult(violations=violations)
    
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