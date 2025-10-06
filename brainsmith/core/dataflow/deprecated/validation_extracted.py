############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""
EXTRACTED VALIDATION CODE - FOR REFERENCE ONLY

This file contains validation code extracted from schemas.py for review.
Not meant to be imported or used directly.
"""

# ===========================================================================
# InterfaceSchema validation methods
# ===========================================================================

def InterfaceSchema_validate_datatype(self, datatype):
    """Check if datatype satisfies constraints."""
    if not self.datatype_constraints:
        return True  # No constraints = allow any
    return validate_datatype_against_constraints(datatype, self.datatype_constraints)

def InterfaceSchema_validate_structure(self):
    """Validate schema structure (internal consistency check)."""
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

def InterfaceSchema_validate(self, model):
    """Validate a model instance against this schema."""
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


# ===========================================================================
# InputSchema validation methods
# ===========================================================================

def InputSchema_validate_structure(self):
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

def InputSchema_validate(self, model):
    """Validate an InputModel against this schema."""
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


# ===========================================================================
# OutputSchema validation methods
# ===========================================================================

def OutputSchema_validate(self, model):
    """Validate an OutputModel against this schema."""
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


# ===========================================================================
# KernelSchema validation methods
# ===========================================================================

def KernelSchema_validate_schema_structures(self, schemas, schema_type):
    """Validate schema structures (internal consistency check)."""
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

def KernelSchema_validate_relationship_structure(self):
    """Validate relationship definitions reference existing interfaces."""
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

def KernelSchema_validate_structure(self):
    """Validate kernel schema structure (internal consistency check)."""
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

def KernelSchema_validate(self, model):
    """Validate a KernelModel against this schema."""
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
