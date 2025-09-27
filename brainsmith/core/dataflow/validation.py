############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Centralized validation for contextualized dataflow modeling.

Provides unified validation with clear phases:
1. Schema validation - structural correctness
2. Context validation - tensor constraints satisfied
3. NodeAttr validation - parameters valid against context
4. Model validation - complete runtime validation
"""

from typing import List, Dict, Any, Optional, Set

from .relationships import ValidationResult, ConstraintViolation, DimensionRelationship, RelationType
from .schemas import KernelSchema
from .models import KernelModel
from .types import Shape


class KernelValidator:
    """Unified validation for kernel modeling."""
    
    def validate_schema(self, schema: KernelSchema) -> ValidationResult:
        """Validate schema structure and consistency."""
        violations = []
        
        # Check schema has required attributes
        if not schema.name:
            violations.append(ConstraintViolation(
                path="schema.name",
                message="Kernel schema must have a name",
                severity="error"
            ))
        
        # Validate at least one input/output
        if not schema.inputs:
            violations.append(ConstraintViolation(
                path="schema.inputs", 
                message="Kernel must have at least one input",
                severity="error"
            ))
            
        if not schema.outputs:
            violations.append(ConstraintViolation(
                path="schema.outputs",
                message="Kernel must have at least one output", 
                severity="error"
            ))
        
        # Validate interface names are unique
        interface_names = set()
        for inp in schema.inputs:
            if inp.name in interface_names:
                violations.append(ConstraintViolation(
                    path=f"schema.inputs.{inp.name}",
                    message=f"Duplicate interface name: {inp.name}",
                    severity="error"
                ))
            interface_names.add(inp.name)
            
        for out in schema.outputs:
            if out.name in interface_names:
                violations.append(ConstraintViolation(
                    path=f"schema.outputs.{out.name}",
                    message=f"Duplicate interface name: {out.name}",
                    severity="error"
                ))
            interface_names.add(out.name)
        
        # Validate relationships reference valid interfaces
        for rel in schema.relationships:
            if rel.source_interface not in interface_names:
                violations.append(ConstraintViolation(
                    path=f"schema.relationships",
                    message=f"Relationship source '{rel.source_interface}' not found",
                    severity="error"
                ))
            if rel.target_interface not in interface_names:
                violations.append(ConstraintViolation(
                    path=f"schema.relationships",
                    message=f"Relationship target '{rel.target_interface}' not found",
                    severity="error"
                ))
        
        # Validate each interface
        for inp in schema.inputs:
            result = inp.validate()
            violations.extend(result.violations)
            
        for out in schema.outputs:
            result = out.validate()
            violations.extend(result.violations)
        
        result = ValidationResult()
        for violation in violations:
            result.add_violation(violation)
        return result
    
    def validate_model(
        self, 
        model: KernelModel,
        schema: KernelSchema,
        tensor_context: Any = None  # Not used in new flow
    ) -> ValidationResult:
        """Validate complete kernel model."""
        violations = []
        
        # Validate model has expected interfaces
        model_input_names = {inp.name for inp in model.inputs}
        schema_input_names = {inp.name for inp in schema.inputs if not inp.optional}
        
        missing_inputs = schema_input_names - model_input_names
        for name in missing_inputs:
            violations.append(ConstraintViolation(
                path=f"model.inputs.{name}",
                message=f"Required input '{name}' missing from model",
                severity="error"
            ))
        
        # Validate relationships
        violations.extend(self._validate_relationships(model, schema))
        
        # Validate streaming constraints
        violations.extend(self._validate_streaming(model))
        
        # Validate performance
        if model.initiation_interval <= 0:
            violations.append(ConstraintViolation(
                path="model.initiation_interval",
                message=f"Initiation interval must be positive, got {model.initiation_interval}",
                severity="error"
            ))
        
        result = ValidationResult()
        for violation in violations:
            result.add_violation(violation)
        return result
    
    def _validate_relationships(
        self, 
        model: KernelModel, 
        schema: KernelSchema
    ) -> List[ConstraintViolation]:
        """Validate dimension relationships in model."""
        violations = []
        
        # Build interface lookup
        interfaces = {}
        for inp in model.inputs:
            interfaces[inp.name] = inp.tensor_dims
        for out in model.outputs:
            interfaces[out.name] = out.tensor_dims
        
        # Check each relationship
        for rel in schema.relationships:
            source_shape = interfaces.get(rel.source_interface)
            target_shape = interfaces.get(rel.target_interface)
            
            if not source_shape or not target_shape:
                continue  # Already caught by other validation
            
            # Validate dimension exists
            if rel.source_dim is not None and rel.source_dim >= len(source_shape):
                violations.append(ConstraintViolation(
                    path=f"relationships.{rel.source_interface}",
                    message=f"Source dimension {rel.source_dim} out of bounds",
                    severity="error"
                ))
                continue
                
            if rel.target_dim is not None and rel.target_dim >= len(target_shape):
                violations.append(ConstraintViolation(
                    path=f"relationships.{rel.target_interface}",
                    message=f"Target dimension {rel.target_dim} out of bounds",
                    severity="error"
                ))
                continue
            
            # Validate relationship holds
            if rel.source_dim is not None:
                source_val = source_shape[rel.source_dim]
            else:
                # None means total size
                source_val = 1
                for dim in source_shape:
                    source_val *= dim
                    
            if rel.target_dim is not None:
                target_val = target_shape[rel.target_dim]
            else:
                # None means total size
                target_val = 1
                for dim in target_shape:
                    target_val *= dim
            
            # Evaluate relationship
            valid = False
            if rel.relation == RelationType.EQUAL:
                valid = source_val == target_val
            elif rel.relation == RelationType.MULTIPLE:
                valid = source_val == rel.factor * target_val
            elif rel.relation == RelationType.DEPENDENT:
                # For DEPENDENT, we just validate it exists
                valid = True
                
            if not valid:
                violations.append(ConstraintViolation(
                    path=f"relationships",
                    message=(
                        f"Relationship {rel.relation.name} violated: "
                        f"{rel.source_interface}[{rel.source_dim}]={source_val} "
                        f"vs {rel.target_interface}[{rel.target_dim}]={target_val}"
                    ),
                    severity="error"
                ))
        
        return violations
    
    def _validate_streaming(self, model: KernelModel) -> List[ConstraintViolation]:
        """Validate streaming configuration."""
        violations = []
        
        # Check input streaming
        for inp in model.inputs:
            if hasattr(inp, 'stream_dims') and inp.stream_dims:
                # Validate stream divides block evenly
                for i, (stream, block) in enumerate(zip(inp.stream_dims, inp.block_dims)):
                    if block % stream != 0:
                        violations.append(ConstraintViolation(
                            path=f"model.inputs.{inp.name}.stream_dims[{i}]",
                            message=(
                                f"Stream dimension {stream} does not divide "
                                f"block dimension {block} evenly"
                            ),
                            severity="error"
                        ))
        
        # Check bandwidth constraints
        for inp in model.inputs:
            if hasattr(inp, 'bandwidth_bits') and inp.bandwidth_bits > 1024:
                violations.append(ConstraintViolation(
                    path=f"model.inputs.{inp.name}.bandwidth",
                    message=f"Input bandwidth {inp.bandwidth_bits} bits exceeds 1024 bit limit",
                    severity="warning"
                ))
        
        return violations


# Convenience functions for backward compatibility
def validate_kernel_schema(schema: KernelSchema) -> ValidationResult:
    """Validate kernel schema."""
    return KernelValidator().validate_schema(schema)


def validate_kernel_model(
    model: KernelModel,
    schema: KernelSchema,
    tensor_context: Any = None
) -> ValidationResult:
    """Validate kernel model."""
    return KernelValidator().validate_model(model, schema, tensor_context)


# Dummy for backward compatibility
def validate_kernel_config(*args, **kwargs) -> ValidationResult:
    """Config validation is now part of contextualized schema validation."""
    return ValidationResult(violations=[])