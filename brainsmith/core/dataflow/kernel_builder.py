############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Unified kernel building with fluent interface using direct factory.

Provides a clean builder pattern for kernel model creation using the
direct factory approach.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .schemas import KernelSchema
from .models import KernelModel
from .validation import KernelValidator, ValidationResult, ConstraintViolation
from .tensor_context import TensorContext
from .direct_factory import DirectKernelFactory


@dataclass
class BuilderState:
    """Tracks builder state for direct flow."""
    schema: Optional[KernelSchema] = None
    tensor_context: Optional[TensorContext] = None
    nodeattrs: Dict[str, Any] = field(default_factory=dict)
    validation_errors: List[ConstraintViolation] = field(default_factory=list)


class KernelBuilder:
    """Unified kernel building with direct factory flow.
    
    Example usage:
        ```python
        # Using ModelWrapper
        model = (KernelBuilder()
            .from_schema_and_model(schema, node, model_wrapper)
            .with_nodeattrs({"PE": 16, "SIMD": 8})
            .build())
        
        # Or with TensorContext
        model = (KernelBuilder()
            .from_schema(schema)
            .with_tensor_context(context)
            .with_nodeattrs({"PE": 16})
            .build())
        ```
    """
    
    def __init__(self):
        """Initialize empty builder."""
        self._state = BuilderState()
        self._validator = KernelValidator()
    
    # =============================================================================
    # Builder Methods - Schema and Context
    # =============================================================================
    
    
    def from_schema_and_model(self, schema: KernelSchema, node, model: ModelWrapper) -> 'KernelBuilder':
        """Initialize using schema and model.
        
        Args:
            schema: Kernel schema
            node: ONNX node
            model: ModelWrapper with tensor info
            
        Returns:
            Self for chaining
        """
        self._state.schema = schema
        self._state.tensor_context = TensorContext.from_model_wrapper(node, model)
        
        # Validate schema
        result = self._validator.validate_schema(schema)
        self._state.validation_errors.extend(result.violations)
        
        return self
    
    def from_schema(self, schema: KernelSchema) -> 'KernelBuilder':
        """Set kernel schema.
        
        Args:
            schema: Kernel schema defining structure
            
        Returns:
            Self for chaining
        """
        self._state.schema = schema
        
        # Validate schema
        result = self._validator.validate_schema(schema)
        self._state.validation_errors.extend(result.violations)
        
        return self
    
    def with_tensor_context(self, context: TensorContext) -> 'KernelBuilder':
        """Set tensor context.
        
        Args:
            context: Tensor context from ONNX graph
            
        Returns:
            Self for chaining
        """
        if not self._state.schema:
            raise ValueError("Must set schema before tensor context")
            
        self._state.tensor_context = context
        return self
    
    # =============================================================================
    # Builder Methods - NodeAttrs
    # =============================================================================
    
    def with_nodeattrs(self, attrs: Dict[str, Any]) -> 'KernelBuilder':
        """Set node attributes.
        
        Args:
            attrs: Dictionary of node attributes
            
        Returns:
            Self for chaining
        """
        self._state.nodeattrs.update(attrs)
        return self
    
    def with_nodeattr(self, name: str, value: Any) -> 'KernelBuilder':
        """Set a single node attribute.
        
        Args:
            name: Attribute name
            value: Attribute value
            
        Returns:
            Self for chaining
        """
        return self.with_nodeattrs({name: value})
    
    # =============================================================================
    # Validation Methods
    # =============================================================================
    
    def validate(self) -> ValidationResult:
        """Validate current builder state.
        
        Returns:
            ValidationResult with all violations found
        """
        violations = list(self._state.validation_errors)
        
        # Add state-specific validation
        if not self._state.schema:
            violations.append(ConstraintViolation(
                path="builder.schema",
                message="No schema set",
                severity="error"
            ))
        
        # Direct flow validation
        if not self._state.tensor_context:
            violations.append(ConstraintViolation(
                path="builder.tensor_context",
                message="No tensor context - call from_schema_and_model() or with_tensor_context()",
                severity="error"
            ))
        
        if not self._state.nodeattrs:
            violations.append(ConstraintViolation(
                path="builder.nodeattrs",
                message="No node attributes set",
                severity="warning"
            ))
        
        result = ValidationResult()
        for violation in violations:
            result.add_violation(violation)
        return result
    
    def can_build(self) -> bool:
        """Check if builder has enough information to build.
        
        Returns:
            True if build() will succeed
        """
        has_errors = len([v for v in self._state.validation_errors if v.severity == "error"]) > 0
        
        return (
            self._state.schema is not None and
            self._state.tensor_context is not None and
            not has_errors
        )
    
    # =============================================================================
    # Build Methods
    # =============================================================================
    
    def build(self) -> KernelModel:
        """Build the kernel model.
        
        Returns:
            Complete KernelModel instance
            
        Raises:
            ValueError: If validation fails
        """
        # Check readiness
        if not self.can_build():
            result = self.validate()
            raise ValueError(
                f"Cannot build model: {result.get_error_summary()}"
            )
        
        # Direct flow: Schema + TensorContext + NodeAttrs → Model
        model = DirectKernelFactory.create_model(
            self._state.schema,
            self._state.tensor_context,
            self._state.nodeattrs
        )
        
        return model
    


# =============================================================================
# Convenience Functions
# =============================================================================

def build_kernel_model(
    schema: KernelSchema,
    node,
    model: ModelWrapper,
    nodeattrs: Dict[str, Any]
) -> KernelModel:
    """Build kernel model using direct factory.
    
    Args:
        schema: Kernel schema
        node: ONNX node
        model: ModelWrapper with tensor info
        nodeattrs: Node attributes
        
    Returns:
        Complete KernelModel
        
    Raises:
        ValueError: If validation fails
    """
    return (KernelBuilder()
        .from_schema_and_model(schema, node, model)
        .with_nodeattrs(nodeattrs)
        .build())


def build_kernel_model_direct(
    schema: KernelSchema,
    tensor_context: TensorContext,
    nodeattrs: Dict[str, Any]
) -> KernelModel:
    """Build kernel model using direct factory flow.
    
    Args:
        schema: Kernel schema
        tensor_context: Tensor context from ONNX graph
        nodeattrs: Node attributes
        
    Returns:
        Complete KernelModel
        
    Raises:
        ValueError: If validation fails
    """
    return (KernelBuilder()
        .from_schema(schema)
        .with_tensor_context(tensor_context)
        .with_nodeattrs(nodeattrs)
        .build())