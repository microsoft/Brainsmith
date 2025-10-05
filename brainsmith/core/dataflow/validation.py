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
from .resolved_config import ResolvedKernelConfig, ResolvedInterfaceConfig
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


class ConfigValidator:
    """Validates resolved configurations."""

    def __init__(self, schema: KernelSchema):
        self.schema = schema

    def validate(self, config: ResolvedKernelConfig) -> ValidationResult:
        """Validate resolved configuration against schema."""
        violations = []

        # Validate config matches schema structure
        if config.kernel_name != self.schema.name:
            violations.append(ConstraintViolation(
                constraint_type="config_schema_mismatch",
                message=f"Config kernel name '{config.kernel_name}' doesn't match schema '{self.schema.name}'",
                severity="error"
            ))

        # Validate interface counts
        if len(config.inputs) != len(self.schema.inputs):
            violations.append(ConstraintViolation(
                constraint_type="interface_count",
                message=f"Input count mismatch: config has {len(config.inputs)}, schema has {len(self.schema.inputs)}",
                severity="error"
            ))

        if len(config.outputs) != len(self.schema.outputs):
            violations.append(ConstraintViolation(
                constraint_type="interface_count",
                message=f"Output count mismatch: config has {len(config.outputs)}, schema has {len(self.schema.outputs)}",
                severity="error"
            ))

        # Validate resolved parameters
        param_result = self._validate_parameters(config.parameters)
        violations.extend(param_result.violations)

        # Validate each interface config
        for i, inp_config in enumerate(config.inputs):
            if i < len(self.schema.inputs):
                interface_result = self._validate_interface_config(
                    inp_config, self.schema.inputs[i], f"input[{i}]"
                )
                violations.extend(interface_result.violations)

        for i, out_config in enumerate(config.outputs):
            if i < len(self.schema.outputs):
                interface_result = self._validate_interface_config(
                    out_config, self.schema.outputs[i], f"output[{i}]"
                )
                violations.extend(interface_result.violations)

        return ValidationResult(violations=violations)

    def _validate_interface_config(
        self,
        config: ResolvedInterfaceConfig,
        schema: InterfaceSchema,
        context: str
    ) -> ValidationResult:
        """Validate interface configuration against schema."""
        violations = []

        # Name must match
        if config.name != schema.name:
            violations.append(ConstraintViolation(
                constraint_type="interface_name",
                message=f"{context}: Name mismatch - config '{config.name}' vs schema '{schema.name}'",
                severity="error"
            ))

        # Validate resolved tiling parameters
        if config.block_params:
            for i, param in enumerate(config.block_params):
                if not isinstance(param, int) or param <= 0:
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_parameter",
                        message=f"{context}.block_params[{i}]: Must be positive integer, got {param}",
                        severity="error"
                    ))

        if config.stream_params:
            for i, param in enumerate(config.stream_params):
                if not isinstance(param, int) or param <= 0:
                    violations.append(ConstraintViolation(
                        constraint_type="tiling_parameter",
                        message=f"{context}.stream_params[{i}]: Must be positive integer, got {param}",
                        severity="error"
                    ))

        return ValidationResult(violations=violations)

    def _validate_parameters(self, parameters: Dict[str, Any]) -> ValidationResult:
        """Validate kernel parameters."""
        violations = []

        # Check for required parameters based on schema
        # This would be expanded based on actual parameter requirements

        for name, value in parameters.items():
            if value is None:
                violations.append(ConstraintViolation(
                    constraint_type="parameter_value",
                    message=f"Parameter '{name}' cannot be None",
                    severity="warning"
                ))

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


class KernelValidator:
    """Centralized validation with clear phases."""

    def __init__(self):
        self._schema_validator = SchemaValidator()

    def validate_schema(self, schema: KernelSchema) -> ValidationResult:
        """Compile-time schema validation."""
        return self._schema_validator.validate(schema)

    def validate_config(self, config: ResolvedKernelConfig, schema: KernelSchema) -> ValidationResult:
        """Phase 1 validation (nodeattrs)."""
        config_validator = ConfigValidator(schema)
        return config_validator.validate(config)

    def validate_model(
        self,
        model: KernelModel,
        schema: KernelSchema,
        tensor_context: TensorContext
    ) -> ValidationResult:
        """Phase 2 validation (complete model)."""
        model_validator = ModelValidator(schema)
        return model_validator.validate(model, tensor_context)

    def validate_full_pipeline(
        self,
        schema: KernelSchema,
        config: ResolvedKernelConfig,
        model: KernelModel,
        tensor_context: TensorContext
    ) -> ValidationResult:
        """Validate entire pipeline and collect all violations."""
        all_violations = []

        # Schema validation
        schema_result = self.validate_schema(schema)
        all_violations.extend(schema_result.violations)

        # Config validation
        config_result = self.validate_config(config, schema)
        all_violations.extend(config_result.violations)

        # Model validation
        model_result = self.validate_model(model, schema, tensor_context)
        all_violations.extend(model_result.violations)

        return ValidationResult(violations=all_violations)


# Convenience functions for common validation tasks
def validate_kernel_schema(schema: KernelSchema) -> ValidationResult:
    """Validate a kernel schema."""
    validator = KernelValidator()
    return validator.validate_schema(schema)


def validate_kernel_config(config: ResolvedKernelConfig, schema: KernelSchema) -> ValidationResult:
    """Validate a resolved kernel configuration."""
    validator = KernelValidator()
    return validator.validate_config(config, schema)


def validate_kernel_model(
    model: KernelModel,
    schema: KernelSchema,
    tensor_context: TensorContext
) -> ValidationResult:
    """Validate a complete kernel model."""
    validator = KernelValidator()
    return validator.validate_model(model, schema, tensor_context)
