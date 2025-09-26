############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Centralized datatype resolution with clear precedence rules.

Provides a unified interface for resolving datatypes from multiple sources
with a clear fallback chain and caching for performance.
"""

from typing import Dict, Any, Optional, List, Protocol, Callable
from dataclasses import dataclass, field
from enum import Enum

from qonnx.core.datatype import DataType, BaseDataType
from .resolved_config import ResolvedInterfaceConfig
from .tensor_context import TensorContext
from .schemas import InterfaceSchema


class ResolutionSource(Enum):
    """Source of datatype resolution."""
    NODEATTR = "nodeattr"
    TENSOR_CONTEXT = "tensor_context" 
    DEFAULT = "default"
    SCHEMA_CONSTRAINT = "schema_constraint"


@dataclass
class ResolutionResult:
    """Result of datatype resolution."""
    datatype: DataType
    source: ResolutionSource
    source_details: Optional[str] = None
    
    def __repr__(self) -> str:
        """String representation."""
        details = f" ({self.source_details})" if self.source_details else ""
        return f"ResolutionResult({self.datatype.name} from {self.source.value}{details})"


@dataclass 
class ResolutionContext:
    """Context for datatype resolution."""
    interface_config: ResolvedInterfaceConfig
    interface_schema: InterfaceSchema
    position: int
    is_input: bool
    nodeattrs: Dict[str, Any] = field(default_factory=dict)
    tensor_context: Optional[TensorContext] = None


class ResolutionStrategy(Protocol):
    """Protocol for datatype resolution strategies."""
    
    def resolve(self, context: ResolutionContext) -> Optional[ResolutionResult]:
        """Try to resolve datatype from this source.
        
        Args:
            context: Resolution context
            
        Returns:
            ResolutionResult if successful, None otherwise
        """
        ...


class NodeAttrStrategy:
    """Resolve datatypes from node attributes."""
    
    def resolve(self, context: ResolutionContext) -> Optional[ResolutionResult]:
        """Resolve from nodeattr."""
        if not context.interface_config.datatype_attr:
            return None
            
        dtype_str = context.nodeattrs.get(context.interface_config.datatype_attr)
        if not dtype_str:
            return None
            
        try:
            datatype = DataType[dtype_str]
            return ResolutionResult(
                datatype=datatype,
                source=ResolutionSource.NODEATTR,
                source_details=context.interface_config.datatype_attr
            )
        except KeyError:
            # Invalid datatype string
            return None


class TensorContextStrategy:
    """Resolve datatypes from tensor context."""
    
    def resolve(self, context: ResolutionContext) -> Optional[ResolutionResult]:
        """Resolve from tensor context."""
        if not context.tensor_context:
            return None
            
        # Get datatype from appropriate tensor list
        tensors = (context.tensor_context.inputs if context.is_input 
                  else context.tensor_context.outputs)
        
        if context.position >= len(tensors):
            return None
            
        tensor_info = tensors[context.position]
        if not tensor_info.datatype:
            return None
            
        return ResolutionResult(
            datatype=tensor_info.datatype,
            source=ResolutionSource.TENSOR_CONTEXT,
            source_details=f"{'input' if context.is_input else 'output'}[{context.position}]"
        )


class DefaultStrategy:
    """Resolve to default datatypes."""
    
    def __init__(self, default_input: DataType = DataType["INT8"],
                 default_output: DataType = DataType["INT8"]):
        """Initialize with default datatypes.
        
        Args:
            default_input: Default datatype for inputs
            default_output: Default datatype for outputs
        """
        self.default_input = default_input
        self.default_output = default_output
    
    def resolve(self, context: ResolutionContext) -> Optional[ResolutionResult]:
        """Resolve to default datatype."""
        default = self.default_input if context.is_input else self.default_output
        
        return ResolutionResult(
            datatype=default,
            source=ResolutionSource.DEFAULT,
            source_details=f"default_{'input' if context.is_input else 'output'}"
        )


class SchemaConstraintStrategy:
    """Resolve to first valid datatype from schema constraints."""
    
    def resolve(self, context: ResolutionContext) -> Optional[ResolutionResult]:
        """Find first valid datatype from constraints."""
        if not context.interface_schema.datatype_constraints:
            return None
            
        # Try common datatypes in order of preference
        candidates = [
            DataType["INT8"], DataType["UINT8"],
            DataType["INT16"], DataType["UINT16"], 
            DataType["INT32"], DataType["UINT32"],
            DataType["FLOAT32"], DataType["FLOAT16"]
        ]
        
        # Check each candidate against constraints
        for dtype in candidates:
            if context.interface_schema.validate_datatype(dtype):
                return ResolutionResult(
                    datatype=dtype,
                    source=ResolutionSource.SCHEMA_CONSTRAINT,
                    source_details=f"first valid from constraints"
                )
        
        return None


class DatatypeResolver:
    """Centralized datatype resolution with clear precedence.
    
    Default precedence chain:
    1. Node attributes (explicit user configuration)
    2. Tensor context (from ONNX graph)
    3. Schema constraints (first valid)
    4. Default values
    
    Usage:
        ```python
        resolver = DatatypeResolver()
        result = resolver.resolve(
            interface_config, 
            interface_schema,
            position=0,
            is_input=True,
            nodeattrs={"inputDataType": "INT16"},
            tensor_context=ctx
        )
        print(result.datatype)  # INT16
        print(result.source)    # ResolutionSource.NODEATTR
        ```
    """
    
    def __init__(
        self,
        strategies: Optional[List[ResolutionStrategy]] = None,
        default_input: DataType = DataType["INT8"],
        default_output: DataType = DataType["INT8"],
        enable_caching: bool = True
    ):
        """Initialize resolver with strategies.
        
        Args:
            strategies: Optional custom resolution strategies
            default_input: Default datatype for inputs
            default_output: Default datatype for outputs  
            enable_caching: Whether to cache resolution results
        """
        if strategies is None:
            # Default precedence chain
            strategies = [
                NodeAttrStrategy(),
                TensorContextStrategy(),
                SchemaConstraintStrategy(),
                DefaultStrategy(default_input, default_output)
            ]
        
        self.strategies = strategies
        self.enable_caching = enable_caching
        self._cache: Dict[str, ResolutionResult] = {}
    
    def resolve(
        self,
        interface_config: ResolvedInterfaceConfig,
        interface_schema: InterfaceSchema,
        position: int,
        is_input: bool,
        nodeattrs: Optional[Dict[str, Any]] = None,
        tensor_context: Optional[TensorContext] = None
    ) -> ResolutionResult:
        """Resolve datatype for an interface.
        
        Args:
            interface_config: Resolved interface configuration
            interface_schema: Interface schema with constraints
            position: Position in input/output list
            is_input: Whether this is an input interface
            nodeattrs: Optional node attributes
            tensor_context: Optional tensor context
            
        Returns:
            ResolutionResult with datatype and source
        """
        # Check cache
        cache_key = f"{interface_config.name}_{position}_{is_input}"
        if self.enable_caching and cache_key in self._cache:
            return self._cache[cache_key]
        
        # Build context
        context = ResolutionContext(
            interface_config=interface_config,
            interface_schema=interface_schema,
            position=position,
            is_input=is_input,
            nodeattrs=nodeattrs or {},
            tensor_context=tensor_context
        )
        
        # Try each strategy in order
        for strategy in self.strategies:
            result = strategy.resolve(context)
            if result is not None:
                # Validate against constraints
                if interface_schema.validate_datatype(result.datatype):
                    # Cache and return
                    if self.enable_caching:
                        self._cache[cache_key] = result
                    return result
        
        # This should not happen with default strategies
        raise RuntimeError(
            f"Failed to resolve datatype for {interface_config.name} - "
            f"no valid datatype found"
        )
    
    def resolve_all(
        self,
        inputs: List[ResolvedInterfaceConfig],
        outputs: List[ResolvedInterfaceConfig],
        input_schemas: List[InterfaceSchema],
        output_schemas: List[InterfaceSchema],
        nodeattrs: Optional[Dict[str, Any]] = None,
        tensor_context: Optional[TensorContext] = None
    ) -> Dict[str, DataType]:
        """Resolve datatypes for all interfaces.
        
        Returns:
            Dictionary mapping interface names to datatypes
        """
        result = {}
        
        # Resolve inputs
        for i, (config, schema) in enumerate(zip(inputs, input_schemas)):
            resolution = self.resolve(
                config, schema, i, True, nodeattrs, tensor_context
            )
            result[config.name] = resolution.datatype
        
        # Resolve outputs
        for i, (config, schema) in enumerate(zip(outputs, output_schemas)):
            resolution = self.resolve(
                config, schema, i, False, nodeattrs, tensor_context
            )
            result[config.name] = resolution.datatype
        
        return result
    
    def clear_cache(self) -> None:
        """Clear resolution cache."""
        self._cache.clear()
    
    def get_resolution_chain(self) -> List[str]:
        """Get human-readable resolution chain."""
        return [strategy.__class__.__name__ for strategy in self.strategies]


# Convenience functions
def create_datatype_resolver(
    default_input: str = "INT8",
    default_output: str = "INT8"
) -> DatatypeResolver:
    """Create a standard datatype resolver.
    
    Args:
        default_input: Default input datatype name
        default_output: Default output datatype name
        
    Returns:
        Configured DatatypeResolver
    """
    return DatatypeResolver(
        default_input=DataType[default_input],
        default_output=DataType[default_output]
    )


def resolve_interface_datatypes(
    kernel_config,
    kernel_schema,
    nodeattrs: Optional[Dict[str, Any]] = None,
    tensor_context: Optional[TensorContext] = None
) -> Dict[str, DataType]:
    """Resolve datatypes for all kernel interfaces.
    
    Args:
        kernel_config: ResolvedKernelConfig
        kernel_schema: KernelSchema
        nodeattrs: Optional node attributes
        tensor_context: Optional tensor context
        
    Returns:
        Dictionary mapping interface names to datatypes
    """
    resolver = DatatypeResolver()
    return resolver.resolve_all(
        kernel_config.inputs,
        kernel_config.outputs,
        kernel_schema.inputs,
        kernel_schema.outputs,
        nodeattrs,
        tensor_context
    )