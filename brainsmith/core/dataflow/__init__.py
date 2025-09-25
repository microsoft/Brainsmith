############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Core dataflow modeling components

This module provides the core classes for modeling dataflow kernels.

Key Components:
- InputSchema/OutputSchema: Schema definitions for kernel interfaces
- KernelSchema: Kernel definition with input/output schemas
- Immutable models: Created via factory functions in models.py
- RelationType: Dimension relationships and constraints
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

# Core architecture - schemas consolidated in schemas.py
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

# Two-phase model creation components
from .resolved_config import ResolvedInterfaceConfig, ResolvedKernelConfig
from .tensor_context import TensorContext, TensorInfo
from .model_factory import KernelModelFactory

# Schema compilation for performance
from .schema_compiler import CompiledSchema, SchemaCompiler

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
    
    # Two-phase model creation
    'ResolvedInterfaceConfig', 'ResolvedKernelConfig',
    'TensorContext', 'TensorInfo', 'KernelModelFactory',
    
    # Schema compilation
    'CompiledSchema', 'SchemaCompiler',
    
    # Utility modules
    'template_utils', 'shape_utils',
]