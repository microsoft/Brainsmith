############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Core dataflow modeling components

This module provides the core classes for modeling dataflow kernels using
the contextualized architecture where tensor context is applied before nodeattrs.

Key Components:
- KernelSchema: Defines kernel structure and constraints
- ContextualizedSchema: Schema with tensor context applied
- KernelModel: Immutable runtime model
- KernelBuilder: Fluent API for model creation
"""

# Core types
from .types import Shape

# Relationships
from .relationships import DimensionRelationship, RelationType

# QONNX types (direct from QONNX)
from qonnx.core.datatype import DataType, BaseDataType

# Constraint types
from .constraint_types import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints,
)

# Core architecture - schemas
from .schemas import InputSchema, OutputSchema, KernelSchema


# Immutable models and factory functions
from .models import (
    InputModel,
    OutputModel,
    KernelModel,
    create_kernel_model,
    create_input_model,
    create_output_model,
    update_kernel_stream_config
)

# Resolution structures (used internally)
from .resolved_config import ResolvedInterfaceConfig, ResolvedKernelConfig

# Tensor context
from .tensor_context import TensorContext, TensorInfo

# Direct factory
from .direct_factory import DirectKernelFactory

# Schema compilation for performance
from .schema_compiler import CompiledSchema, SchemaCompiler

# Validation layer
from .validation import (
    KernelValidator,
    validate_kernel_schema,
    validate_kernel_model
)

# Also export ValidationResult from relationships
from .relationships import ValidationResult

# Simplified factory pattern
from .kernel_builder import KernelBuilder, build_kernel_model

# Reactive cache management
from .reactive import (
    ReactiveDict,
    ComputedProperty,
    ReactiveState,
    reactive_method
)

# Utility modules
from . import template_utils
from . import shape_utils


__all__ = [
    # Core types
    'Shape',
    
    # Relationships
    'DimensionRelationship', 'RelationType',
    
    # QONNX types
    'DataType', 'BaseDataType',
    
    # Constraint types
    'DatatypeConstraintGroup', 'validate_datatype_against_constraints',
    
    # Core architecture
    'InputSchema', 'OutputSchema', 'KernelSchema',
    
    
    # Immutable models and factory functions
    'InputModel', 'OutputModel', 'KernelModel',
    'create_kernel_model', 'create_input_model', 'create_output_model',
    'update_kernel_stream_config',
    
    # Supporting structures
    'ResolvedInterfaceConfig', 'ResolvedKernelConfig',
    'TensorContext', 'TensorInfo', 'DirectKernelFactory',
    
    # Schema compilation
    'CompiledSchema', 'SchemaCompiler',
    
    # Validation
    'KernelValidator', 'validate_kernel_schema', 'validate_kernel_model',
    'ValidationResult',
    
    # Builder pattern
    'KernelBuilder', 'build_kernel_model',
    
    # Reactive cache management
    'ReactiveDict', 'ComputedProperty', 'ReactiveState', 'reactive_method',
    
    # Utility modules
    'template_utils', 'shape_utils',
]