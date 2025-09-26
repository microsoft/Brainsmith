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
    """Validates kernel schemas for structural correctness."""
    
    def validate(self, schema: KernelSchema) -> ValidationResult:
        """Validate schema structure and consistency."""
        violations = []
        
        # Check schema has required attributes
        if not schema.name:
            violations.append(ConstraintViolation(
                constraint_type="schema_structure",
                message="Kernel schema must have a name",
                severity="error"
            ))
        
        # Validate at least one input/output
        if not schema.inputs:
            violations.append(ConstraintViolation(
                constraint_type="schema_structure", 
                message="Kernel must have at least one input",
                severity="error"
            ))
            
        if not schema.outputs:
            violations.append(ConstraintViolation(
                constraint_type="schema_structure",
                message="Kernel must have at least one output", 
                severity="error"
            ))
        
        # Validate interface names are unique
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
        
        # Validate relationships reference valid interfaces
        for rel in schema.relationships:
            if not self._validate_relationship_interfaces(rel, interface_names):
                violations.append(ConstraintViolation(
                    constraint_type="relationship_reference",
                    message=f"Relationship references unknown interface",
                    severity="error",
                    details={
                        "relationship": str(rel),
                        "known_interfaces": list(interface_names)
                    }
                ))
        
        # Validate individual interfaces
        for i, inp in enumerate(schema.inputs):
            interface_result = self._validate_interface(inp, f"input[{i}]")
            violations.extend(interface_result.violations)
            
        for i, out in enumerate(schema.outputs):
            interface_result = self._validate_interface(out, f"output[{i}]")
            violations.extend(interface_result.violations)
        
        return ValidationResult(violations=violations)
    
    def _validate_interface(self, interface: InterfaceSchema, context: str) -> ValidationResult:
        """Validate individual interface schema."""
        violations = []
        
        # Check required attributes
        if not interface.name:
            violations.append(ConstraintViolation(
                constraint_type="interface_structure",
                message=f"{context}: Interface must have a name",
                severity="error"
            ))
        
        # Validate tiling templates if present
        if hasattr(interface, 'block_tiling') and interface.block_tiling:
            tiling_errors = self._validate_tiling_template(interface.block_tiling)
            for error in tiling_errors:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_template",
                    message=f"{context}.block_tiling: {error}",
                    severity="error"
                ))
        
        if hasattr(interface, 'stream_tiling') and interface.stream_tiling:
            tiling_errors = self._validate_tiling_template(interface.stream_tiling)
            for error in tiling_errors:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_template",
                    message=f"{context}.stream_tiling: {error}",
                    severity="error"
                ))
        
        return ValidationResult(violations=violations)
    
    def _validate_tiling_template(self, template: List[Any]) -> List[str]:
        """Validate tiling template syntax."""
        errors = []
        
        if not template:
            errors.append("Tiling template cannot be empty")
            return errors
            
        for i, elem in enumerate(template):
            if isinstance(elem, int):
                if elem <= 0:
                    errors.append(f"Element {i}: Integer must be positive, got {elem}")
            elif isinstance(elem, str):
                if elem == ":":
                    continue  # Valid full dimension marker
                elif not elem.isidentifier():
                    errors.append(f"Element {i}: Invalid parameter name '{elem}'")
            else:
                errors.append(f"Element {i}: Must be int, str, or ':', got {type(elem).__name__}")
        
        return errors
    
    def _validate_relationship_interfaces(self, rel: DimensionRelationship, interfaces: Set[str]) -> bool:
        """Check if relationship references valid interfaces."""
        # Extract interface names from relationship
        # This is simplified - actual implementation would parse the relationship structure
        return True  # Placeholder for now


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
    """Validates complete kernel models."""
    
    def __init__(self, schema: KernelSchema):
        self.schema = schema
    
    def validate(self, model: KernelModel, tensor_context: TensorContext) -> ValidationResult:
        """Validate complete model including shapes and datatypes."""
        violations = []
        
        # Validate basic structure
        structure_result = self._validate_structure(model)
        violations.extend(structure_result.violations)
        
        # Validate shapes
        shape_result = self._validate_shapes(model, tensor_context)
        violations.extend(shape_result.violations)
        
        # Validate datatypes
        dtype_result = self._validate_datatypes(model)
        violations.extend(dtype_result.violations)
        
        # Validate relationships
        rel_result = self._validate_relationships(model)
        violations.extend(rel_result.violations)
        
        # Validate performance metrics
        perf_result = self._validate_performance(model)
        violations.extend(perf_result.violations)
        
        return ValidationResult(violations=violations)
    
    def _validate_structure(self, model: KernelModel) -> ValidationResult:
        """Validate model structure matches schema."""
        violations = []
        
        if len(model.inputs) != len(self.schema.inputs):
            violations.append(ConstraintViolation(
                constraint_type="model_structure",
                message=f"Input count mismatch: {len(model.inputs)} vs {len(self.schema.inputs)}",
                severity="error"
            ))
            
        if len(model.outputs) != len(self.schema.outputs):
            violations.append(ConstraintViolation(
                constraint_type="model_structure",
                message=f"Output count mismatch: {len(model.outputs)} vs {len(self.schema.outputs)}",
                severity="error"
            ))
        
        return ValidationResult(violations=violations)
    
    def _validate_shapes(self, model: KernelModel, tensor_context: TensorContext) -> ValidationResult:
        """Validate tensor shapes and tiling."""
        violations = []
        
        # Validate input shapes
        for i, inp in enumerate(model.inputs):
            # Check tensor shape matches context
            if i < len(tensor_context.input_shapes):
                expected_shape = tensor_context.input_shapes[i]
                if inp.tensor_dims != expected_shape:
                    violations.append(ConstraintViolation(
                        constraint_type="shape_mismatch",
                        message=f"Input[{i}] shape mismatch: {inp.tensor_dims} vs expected {expected_shape}",
                        severity="error"
                    ))
            
            # Validate tiling divides evenly
            tiling_result = self._validate_tiling(inp.tensor_dims, inp.block_dims, f"input[{i}]")
            violations.extend(tiling_result.violations)
        
        # Validate output shapes
        for i, out in enumerate(model.outputs):
            if i < len(tensor_context.output_shapes):
                expected_shape = tensor_context.output_shapes[i]
                if out.tensor_dims != expected_shape:
                    violations.append(ConstraintViolation(
                        constraint_type="shape_mismatch",
                        message=f"Output[{i}] shape mismatch: {out.tensor_dims} vs expected {expected_shape}",
                        severity="error"
                    ))
            
            # Validate tiling
            tiling_result = self._validate_tiling(out.tensor_dims, out.block_dims, f"output[{i}]")
            violations.extend(tiling_result.violations)
        
        return ValidationResult(violations=violations)
    
    def _validate_tiling(self, tensor_dims: Shape, block_dims: Shape, context: str) -> ValidationResult:
        """Validate tiling divides tensor dimensions evenly."""
        violations = []
        
        if len(tensor_dims) != len(block_dims):
            violations.append(ConstraintViolation(
                constraint_type="tiling_dimension",
                message=f"{context}: Tiling dimension mismatch",
                severity="error",
                details={
                    "tensor_dims": tensor_dims,
                    "block_dims": block_dims
                }
            ))
            return ValidationResult(violations=violations)
        
        for i, (tensor_dim, block_dim) in enumerate(zip(tensor_dims, block_dims)):
            if block_dim <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_value",
                    message=f"{context}: Block dimension [{i}] must be positive",
                    severity="error"
                ))
            elif tensor_dim % block_dim != 0:
                violations.append(ConstraintViolation(
                    constraint_type="tiling_divisibility",
                    message=f"{context}: Dimension [{i}] not evenly divisible: {tensor_dim} % {block_dim} = {tensor_dim % block_dim}",
                    severity="error"
                ))
        
        return ValidationResult(violations=violations)
    
    def _validate_datatypes(self, model: KernelModel) -> ValidationResult:
        """Validate datatypes against constraints."""
        violations = []
        
        # Validate input datatypes
        for i, (inp, schema) in enumerate(zip(model.inputs, self.schema.inputs)):
            if schema.datatype_constraints:
                if not validate_datatype_against_constraints(inp.datatype, schema.datatype_constraints):
                    violations.append(ConstraintViolation(
                        constraint_type="datatype_constraint",
                        message=f"Input[{i}] datatype {inp.datatype.get_canonical_name()} violates constraints",
                        severity="error",
                        details={"constraints": [str(c) for c in schema.datatype_constraints]}
                    ))
        
        # Validate output datatypes  
        for i, (out, schema) in enumerate(zip(model.outputs, self.schema.outputs)):
            if schema.datatype_constraints:
                if not validate_datatype_against_constraints(out.datatype, schema.datatype_constraints):
                    violations.append(ConstraintViolation(
                        constraint_type="datatype_constraint",
                        message=f"Output[{i}] datatype {out.datatype.get_canonical_name()} violates constraints",
                        severity="error",
                        details={"constraints": [str(c) for c in schema.datatype_constraints]}
                    ))
        
        return ValidationResult(violations=violations)
    
    def _validate_relationships(self, model: KernelModel) -> ValidationResult:
        """Validate dimension relationships between interfaces."""
        violations = []
        
        # Create context for relationship evaluation
        context = self._build_relationship_context(model)
        
        # Evaluate each relationship
        for rel in self.schema.relationships:
            result = rel.evaluate(context)
            if not result:
                violations.append(ConstraintViolation(
                    constraint_type="dimension_relationship",
                    message=f"Relationship constraint violated: {rel}",
                    severity="error",
                    details={"context": context}
                ))
        
        return ValidationResult(violations=violations)
    
    def _build_relationship_context(self, model: KernelModel) -> Dict[str, Any]:
        """Build context for relationship evaluation."""
        context = {}
        
        # Add interface dimensions
        for inp in model.inputs:
            for i, dim in enumerate(inp.tensor_dims):
                context[f"{inp.name}.shape[{i}]"] = dim
            for i, dim in enumerate(inp.block_dims):
                context[f"{inp.name}.block[{i}]"] = dim
        
        for out in model.outputs:
            for i, dim in enumerate(out.tensor_dims):
                context[f"{out.name}.shape[{i}]"] = dim
            for i, dim in enumerate(out.block_dims):
                context[f"{out.name}.block[{i}]"] = dim
        
        # Add parameters
        context.update(model.parameters)
        
        return context
    
    def _validate_performance(self, model: KernelModel) -> ValidationResult:
        """Validate performance metrics are reasonable."""
        violations = []
        
        # Check initiation interval
        if model.initiation_interval <= 0:
            violations.append(ConstraintViolation(
                constraint_type="performance_metric",
                message=f"Invalid initiation interval: {model.initiation_interval}",
                severity="error"
            ))
        
        # Check clock frequency
        if model.clock_freq_mhz <= 0:
            violations.append(ConstraintViolation(
                constraint_type="performance_metric",
                message=f"Invalid clock frequency: {model.clock_freq_mhz} MHz",
                severity="error"
            ))
        
        # Validate bandwidth calculations
        for i, inp in enumerate(model.inputs):
            if inp.streaming_bandwidth <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="performance_metric",
                    message=f"Input[{i}] invalid streaming bandwidth: {inp.streaming_bandwidth}",
                    severity="warning"
                ))
        
        for i, out in enumerate(model.outputs):
            if out.streaming_rate <= 0:
                violations.append(ConstraintViolation(
                    constraint_type="performance_metric",
                    message=f"Output[{i}] invalid streaming rate: {out.streaming_rate}",
                    severity="warning"
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