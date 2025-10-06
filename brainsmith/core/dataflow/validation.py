############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Centralized validation layer for dataflow modeling.

Provides a unified validation pipeline with clear phases:
1. Schema validation - structural correctness
2. Config validation - parameter resolution and constraints
3. Model validation - complete runtime validation

All validation returns ValidationResult for proper error collection.
"""

from typing import List, Dict, Any, Optional, Protocol, Set
from abc import ABC, abstractmethod

from .relationships import ValidationResult, ConstraintViolation, DimensionRelationship
from .schemas import KernelSchema, InterfaceSchema
from .models import KernelModel, InputModel, OutputModel
from .types import Shape
from .tensor_context import TensorContext
from .constraint_types import validate_datatype_against_constraints
from qonnx.core.datatype import BaseDataType


class Validator(Protocol):
    """Protocol for phase-specific validators."""

    def validate(self, target: Any) -> ValidationResult:
        """Validate target and return result."""
        ...


class SchemaValidator:
    """Validates kernel schemas for structural correctness.

    Delegates to schema.validate_structure() which performs internal
    consistency checks on the schema definition itself.
    """

    def validate(self, schema: KernelSchema) -> ValidationResult:
        """Validate schema structure and consistency.

        Args:
            schema: KernelSchema to validate

        Returns:
            ValidationResult with structural violations
        """
        # Delegate to schema's validate_structure() method
        result = schema.validate_structure()
        violations = list(result.violations)

        # Add additional high-level checks
        # Check for duplicate interface names across inputs and outputs
        interface_names = set()
        for inp in schema.inputs:
            if inp.name in interface_names:
                violations.append(ConstraintViolation(
                    constraint_type="name_uniqueness",
                    message=f"Duplicate interface name: {inp.name}",
                    severity="error"
                ))
            interface_names.add(inp.name)

        for out in schema.outputs:
            if out.name in interface_names:
                violations.append(ConstraintViolation(
                    constraint_type="name_uniqueness",
                    message=f"Duplicate interface name: {out.name}",
                    severity="error"
                ))
            interface_names.add(out.name)

        return ValidationResult(violations=violations)


class ModelValidator:
    """Validates complete kernel models.

    Delegates to schema.validate(model) for schema-based validation,
    and adds additional context-aware checks.
    """

    def __init__(self, schema: KernelSchema):
        self.schema = schema

    def validate(self, model: KernelModel, tensor_context: TensorContext) -> ValidationResult:
        """Validate complete model including shapes and datatypes.

        Args:
            model: KernelModel to validate
            tensor_context: Tensor context with expected shapes

        Returns:
            ValidationResult with all violations
        """
        # Delegate to schema for core validation
        result = self.schema.validate(model)
        violations = list(result.violations)

        # Add context-aware validation
        shape_result = self._validate_shapes(model, tensor_context)
        violations.extend(shape_result.violations)

        # Validate dimension constraints (NEW)
        constraint_result = self._validate_dimension_constraints(model)
        violations.extend(constraint_result.violations)

        # Validate performance metrics
        perf_result = self._validate_performance(model)
        violations.extend(perf_result.violations)

        return ValidationResult(violations=violations)

    def _validate_shapes(self, model: KernelModel, tensor_context: TensorContext) -> ValidationResult:
        """Validate tensor shapes against context."""
        violations = []

        # Validate input shapes
        for i, inp in enumerate(model.inputs):
            # Check tensor shape matches context
            if i < len(tensor_context.input_shapes):
                expected_shape = tensor_context.input_shapes[i]
                if inp.tensor_shape != expected_shape:
                    violations.append(ConstraintViolation(
                        constraint_type="shape_mismatch",
                        message=f"Input[{i}] shape mismatch",
                        severity="error",
                        expected=expected_shape,
                        actual=inp.tensor_shape
                    ))

            # Validate tiling divides evenly
            tiling_result = self._validate_tiling(inp.tensor_shape, inp.block_shape, f"input[{i}]")
            violations.extend(tiling_result.violations)

        # Validate output shapes
        for i, out in enumerate(model.outputs):
            if i < len(tensor_context.output_shapes):
                expected_shape = tensor_context.output_shapes[i]
                if out.tensor_shape != expected_shape:
                    violations.append(ConstraintViolation(
                        constraint_type="shape_mismatch",
                        message=f"Output[{i}] shape mismatch",
                        severity="error",
                        expected=expected_shape,
                        actual=out.tensor_shape
                    ))

            # Validate tiling
            tiling_result = self._validate_tiling(out.tensor_shape, out.block_shape, f"output[{i}]")
            violations.extend(tiling_result.violations)

        return ValidationResult(violations=violations)

    def _validate_tiling(self, tensor_shape: Shape, block_shape: Shape, context: str) -> ValidationResult:
        """Validate tiling divides tensor dimensions evenly."""
        violations = []

        if len(tensor_shape) != len(block_shape):
            violations.append(ConstraintViolation(
                constraint_type="tiling_dimension",
                message=f"{context}: Tiling dimension mismatch",
                severity="error",
                expected=len(tensor_shape),
                actual=len(block_shape),
                details={
                    "tensor_shape": tensor_shape,
                    "block_shape": block_shape
                }
            ))
            return ValidationResult(violations=violations)

        for i, (tensor_dim, block_dim) in enumerate(zip(tensor_shape, block_shape)):
            if block_dim <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_value",
                    message=f"{context}: Block dimension [{i}] must be positive",
                    severity="error",
                    expected="> 0",
                    actual=block_dim
                ))
            elif tensor_dim % block_dim != 0:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_divisibility",
                    message=f"{context}: Dimension [{i}] not evenly divisible: {tensor_dim} % {block_dim} = {tensor_dim % block_dim}",
                    severity="error"
                ))

        return ValidationResult(violations=violations)

    def _validate_performance(self, model: KernelModel) -> ValidationResult:
        """Validate performance metrics are reasonable."""
        violations = []

        # Check initiation interval
        if model.initiation_interval <= 0:
            violations.append(ConstraintViolation(
                constraint_type="performance_metric",
                message=f"Invalid initiation interval",
                severity="error",
                expected="> 0",
                actual=model.initiation_interval
            ))

        # Check clock frequency
        if model.clock_freq_mhz <= 0:
            violations.append(ConstraintViolation(
                constraint_type="performance_metric",
                message=f"Invalid clock frequency",
                severity="error",
                expected="> 0 MHz",
                actual=model.clock_freq_mhz
            ))

        # Validate bandwidth calculations
        for i, inp in enumerate(model.inputs):
            if inp.streaming_bandwidth <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="performance_metric",
                    message=f"Input[{i}] invalid streaming bandwidth",
                    severity="warning",
                    expected="> 0",
                    actual=inp.streaming_bandwidth
                ))

        for i, out in enumerate(model.outputs):
            streaming_rate = model.output_streaming_rate(i)
            if streaming_rate <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="performance_metric",
                    message=f"Output[{i}] invalid streaming rate",
                    severity="warning",
                    expected="> 0",
                    actual=streaming_rate
                ))

        return ValidationResult(violations=violations)

    def _validate_dimension_constraints(self, model: KernelModel) -> ValidationResult:
        """Validate dimension constraints on interfaces.

        Uses the new two-phase validation approach:
        1. Atomic constraints validated per-interface with check_interface()
        2. Cross-interface constraints validated with check_relationship()

        Args:
            model: KernelModel to validate

        Returns:
            ValidationResult with constraint violations
        """
        violations = []

        # Build interface map for cross-interface validation
        interfaces = {inp.name: inp for inp in model.inputs}
        interfaces.update({out.name: out for out in model.outputs})

        # Dummy nodeattr getter (validation.py doesn't have access to nodeattrs)
        # In actual AutoHWCustomOp usage, this would be self.get_nodeattr
        def dummy_nodeattr_getter(name: str) -> Any:
            raise AttributeError(f"Nodeattr '{name}' not available in validation context")

        # Phase 1: Validate atomic constraints per-interface
        for i, inp in enumerate(model.inputs):
            schema = self.schema.inputs[i]
            for constraint in schema.dimension_constraints:
                # Try atomic validation first
                error = constraint.check_interface(
                    inp.name,
                    inp,
                    dummy_nodeattr_getter
                )
                if error:
                    violations.append(ConstraintViolation(
                        constraint_type="dimension_constraint",
                        message=error,
                        severity="error"
                    ))

        for i, out in enumerate(model.outputs):
            schema = self.schema.outputs[i]
            for constraint in schema.dimension_constraints:
                # Try atomic validation first
                error = constraint.check_interface(
                    out.name,
                    out,
                    dummy_nodeattr_getter
                )
                if error:
                    violations.append(ConstraintViolation(
                        constraint_type="dimension_constraint",
                        message=error,
                        severity="error"
                    ))

        # Phase 2: Validate cross-interface constraints
        # Collect all constraints from all interfaces
        all_constraints = []
        for schema in self.schema.inputs:
            all_constraints.extend(schema.dimension_constraints)
        for schema in self.schema.outputs:
            all_constraints.extend(schema.dimension_constraints)

        # Also get constraints from relationships
        for relationship in self.schema.relationships:
            try:
                all_constraints.extend(relationship.get_constraints())
            except Exception as e:
                violations.append(ConstraintViolation(
                    constraint_type="constraint_generation",
                    message=f"Failed to generate constraints from relationship: {str(e)}",
                    severity="warning",
                    details={"relationship": relationship.describe()}
                ))

        # Validate cross-interface constraints
        for constraint in all_constraints:
            error = constraint.check_relationship(interfaces)
            if error:
                violations.append(ConstraintViolation(
                    constraint_type="dimension_relationship",
                    message=error,
                    severity="error"
                ))

        return ValidationResult(violations=violations)


class KernelValidator:
    """Centralized validation with clear phases."""

    def __init__(self):
        self._schema_validator = SchemaValidator()

    def validate_schema(self, schema: KernelSchema) -> ValidationResult:
        """Compile-time schema validation."""
        return self._schema_validator.validate(schema)

    def validate_model(
        self,
        model: KernelModel,
        schema: KernelSchema,
        tensor_context: TensorContext
    ) -> ValidationResult:
        """Phase 2 validation (complete model)."""
        model_validator = ModelValidator(schema)
        return model_validator.validate(model, tensor_context)


# Convenience functions for common validation tasks
def validate_kernel_schema(schema: KernelSchema) -> ValidationResult:
    """Validate a kernel schema."""
    validator = KernelValidator()
    return validator.validate_schema(schema)


def validate_kernel_model(
    model: KernelModel,
    schema: KernelSchema,
    tensor_context: TensorContext
) -> ValidationResult:
    """Validate a complete kernel model."""
    validator = KernelValidator()
    return validator.validate_model(model, schema, tensor_context)
