############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Unified kernel building with fluent interface.

Simplifies kernel model creation by providing a single builder pattern
that replaces multiple factory functions. Integrates validation at each
build step.
"""

from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field

from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper

from .schemas import KernelSchema
from .resolved_config import ResolvedKernelConfig, ResolvedInterfaceConfig
from .tensor_context import TensorContext, TensorInfo
from .models import KernelModel, create_kernel_model
from .validation import KernelValidator, ValidationResult, ConstraintViolation
from .template_utils import resolve_template_params
from .model_factory import KernelModelFactory
from .types import Shape


@dataclass
class BuilderState:
    """Tracks builder state and progress."""
    schema: Optional[KernelSchema] = None
    nodeattrs: Dict[str, Any] = field(default_factory=dict)
    resolved_config: Optional[ResolvedKernelConfig] = None
    tensor_context: Optional[TensorContext] = None
    datatype_resolver: Dict[str, DataType] = field(default_factory=dict)
    validation_errors: List[ConstraintViolation] = field(default_factory=list)


class KernelBuilder:
    """Unified kernel building with fluent interface.
    
    Example usage:
        ```python
        model = (KernelBuilder()
            .from_schema(my_schema)
            .with_nodeattrs({"PE": 16, "SIMD": 8})
            .with_tensor_context(tensor_ctx)
            .build())
        ```
    
    The builder validates at each step and collects all violations.
    Call validate() to check if build will succeed before calling build().
    """
    
    def __init__(self):
        """Initialize empty builder."""
        self._state = BuilderState()
        self._validator = KernelValidator()
    
    # =============================================================================
    # Fluent Interface Methods
    # =============================================================================
    
    def from_schema(self, schema: KernelSchema) -> 'KernelBuilder':
        """Set the kernel schema.
        
        Args:
            schema: The kernel schema to build from
            
        Returns:
            Self for chaining
            
        Raises:
            ValueError: If schema is invalid
        """
        # Validate schema
        result = self._validator.validate_schema(schema)
        if not result.is_valid():
            raise ValueError(
                f"Invalid schema: {result.get_error_summary()}"
            )
        
        self._state.schema = schema
        return self
    
    def with_nodeattrs(self, attrs: Dict[str, Any]) -> 'KernelBuilder':
        """Set node attributes and resolve configuration.
        
        This triggers Phase 1 resolution, creating ResolvedKernelConfig.
        
        Args:
            attrs: Dictionary of node attributes
            
        Returns:
            Self for chaining
        """
        if not self._state.schema:
            raise RuntimeError("Must call from_schema() before with_nodeattrs()")
        
        self._state.nodeattrs.update(attrs)
        
        # Resolve configuration (Phase 1)
        self._state.resolved_config = self._resolve_config()
        
        # Validate resolved config
        result = self._validator.validate_config(
            self._state.resolved_config,
            self._state.schema
        )
        self._state.validation_errors.extend(result.violations)
        
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
    
    def with_tensor_context(self, context: TensorContext) -> 'KernelBuilder':
        """Set tensor context for Phase 2 resolution.
        
        Args:
            context: Tensor context with shapes and datatypes
            
        Returns:
            Self for chaining
        """
        self._state.tensor_context = context
        
        # Build datatype resolver from context
        self._build_datatype_resolver()
        
        return self
    
    def with_model_wrapper(self, onnx_node, model: ModelWrapper) -> 'KernelBuilder':
        """Extract tensor context from ModelWrapper.
        
        Args:
            onnx_node: The ONNX node
            model: The ModelWrapper instance
            
        Returns:
            Self for chaining
        """
        context = TensorContext.from_model_wrapper(onnx_node, model)
        return self.with_tensor_context(context)
    
    # =============================================================================
    # Validation Methods
    # =============================================================================
    
    def validate(self) -> ValidationResult:
        """Validate current builder state.
        
        Returns:
            ValidationResult with all violations found so far
        """
        violations = list(self._state.validation_errors)
        
        # Add state-specific validation
        if not self._state.schema:
            violations.append(ConstraintViolation(
                constraint_type="builder_state",
                message="No schema set - call from_schema() first",
                severity="error"
            ))
        
        if not self._state.resolved_config:
            violations.append(ConstraintViolation(
                constraint_type="builder_state",
                message="No configuration resolved - call with_nodeattrs() first",
                severity="error"
            ))
        
        if not self._state.tensor_context:
            violations.append(ConstraintViolation(
                constraint_type="builder_state",
                message="No tensor context - call with_tensor_context() first",
                severity="error"
            ))
        
        return ValidationResult(violations=violations)
    
    def can_build(self) -> bool:
        """Check if builder has enough information to build.
        
        Returns:
            True if build() will succeed
        """
        return (
            self._state.schema is not None and
            self._state.resolved_config is not None and
            self._state.tensor_context is not None and
            len(self._state.validation_errors) == 0
        )
    
    # =============================================================================
    # Build Methods
    # =============================================================================
    
    def build(self) -> KernelModel:
        """Build the kernel model.
        
        Returns:
            Complete KernelModel instance
            
        Raises:
            RuntimeError: If builder state is incomplete
            ValueError: If validation fails
        """
        # Check readiness
        if not self.can_build():
            result = self.validate()
            raise ValueError(
                f"Cannot build model: {result.get_error_summary()}"
            )
        
        # Create model using factory
        model = KernelModelFactory.create_model(
            self._state.resolved_config,
            self._state.tensor_context,
            self._state.datatype_resolver
        )
        
        # Final validation
        result = self._validator.validate_model(
            model,
            self._state.schema,
            self._state.tensor_context
        )
        
        if not result.is_valid():
            raise ValueError(
                f"Model validation failed: {result.get_error_summary()}"
            )
        
        return model
    
    def build_partial(self) -> Union[ResolvedKernelConfig, KernelModel]:
        """Build as far as possible with current state.
        
        Returns:
            ResolvedKernelConfig if only Phase 1 complete, or
            KernelModel if fully resolved
            
        Raises:
            RuntimeError: If insufficient state to build anything
        """
        if self.can_build():
            return self.build()
        
        if self._state.resolved_config:
            return self._state.resolved_config
        
        raise RuntimeError("Insufficient state to build - need at least schema and nodeattrs")
    
    # =============================================================================
    # Private Methods
    # =============================================================================
    
    def _resolve_config(self) -> ResolvedKernelConfig:
        """Resolve configuration from schema and nodeattrs."""
        schema = self._state.schema
        
        # Helper to get nodeattr with error handling
        def get_attr(name: str, default=None):
            return self._state.nodeattrs.get(name, default)
        
        # Helper to get nodeattr types (for template resolution)
        def get_attr_types() -> Dict[str, type]:
            return {k: type(v) for k, v in self._state.nodeattrs.items()}
        
        # Resolve interfaces
        inputs = []
        for i, input_schema in enumerate(schema.inputs):
            inputs.append(self._resolve_interface(
                input_schema, i, True, get_attr, get_attr_types
            ))

        outputs = []
        for i, output_schema in enumerate(schema.outputs):
            outputs.append(self._resolve_interface(
                output_schema, i, False, get_attr, get_attr_types
            ))
        
        # Extract parameters
        parameters = self._extract_parameters(get_attr)
        
        # Get clock frequency
        clock_freq_mhz = get_attr("clock_freq_mhz", 100.0)
        
        return ResolvedKernelConfig(
            kernel_name=schema.name,
            inputs=inputs,
            outputs=outputs,
            parameters=parameters,
            clock_freq_mhz=clock_freq_mhz
        )
    
    def _resolve_interface(
        self,
        schema,
        position: int,
        is_input: bool,
        get_attr,
        get_attr_types
    ) -> ResolvedInterfaceConfig:
        """Resolve single interface configuration."""
        # Resolve block tiling
        block_params = None
        if schema.block_tiling:
            block_params = resolve_template_params(
                schema.block_tiling,
                get_attr,
                get_attr_types()
            )
        
        # Resolve stream tiling (inputs only)
        stream_params = None
        if is_input and hasattr(schema, 'stream_tiling') and schema.stream_tiling:
            stream_params = resolve_template_params(
                schema.stream_tiling,
                get_attr,
                get_attr_types()
            )
        
        return ResolvedInterfaceConfig(
            name=schema.name,
            position=position,
            block_params=block_params or [":"],
            stream_params=stream_params,
            datatype_attr=schema.get_datatype_attr(position),
            is_weight=getattr(schema, 'is_weight', False),
            optional=schema.optional
        )
    
    def _extract_parameters(self, get_attr) -> Dict[str, Any]:
        """Extract kernel parameters from nodeattrs."""
        params = {}
        
        # Get all parameter names from schema
        # This is simplified - would need to extract from templates
        param_names = set()
        for inp in self._state.schema.inputs:
            if inp.block_tiling:
                for item in inp.block_tiling:
                    if isinstance(item, str) and item != ":":
                        param_names.add(item)
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                for item in inp.stream_tiling:
                    if isinstance(item, str) and item != ":":
                        param_names.add(item)
        
        # Extract values
        for name in param_names:
            value = get_attr(name)
            if value is not None:
                # Unwrap single-element lists
                if isinstance(value, list) and len(value) == 1:
                    value = value[0]
                params[name] = value
        
        return params
    
    def _build_datatype_resolver(self) -> None:
        """Build datatype resolver from nodeattrs and context."""
        resolver = {}
        
        if not self._state.resolved_config:
            return
        
        # Get datatypes from nodeattrs
        for config in self._state.resolved_config.inputs + self._state.resolved_config.outputs:
            if config.datatype_attr:
                dtype_str = self._state.nodeattrs.get(config.datatype_attr)
                if dtype_str:
                    try:
                        resolver[config.datatype_attr] = DataType[dtype_str]
                    except KeyError:
                        self._state.validation_errors.append(ConstraintViolation(
                            constraint_type="datatype_resolution",
                            message=f"Invalid datatype '{dtype_str}' for {config.datatype_attr}",
                            severity="error"
                        ))
        
        self._state.datatype_resolver = resolver


# =============================================================================
# Convenience Functions
# =============================================================================

def build_kernel_model(
    schema: KernelSchema,
    nodeattrs: Dict[str, Any],
    tensor_context: TensorContext
) -> KernelModel:
    """Build a kernel model in one call.
    
    Args:
        schema: Kernel schema
        nodeattrs: Node attributes
        tensor_context: Tensor context
        
    Returns:
        Complete KernelModel
        
    Raises:
        ValueError: If validation fails at any stage
    """
    return (KernelBuilder()
        .from_schema(schema)
        .with_nodeattrs(nodeattrs)
        .with_tensor_context(tensor_context)
        .build())