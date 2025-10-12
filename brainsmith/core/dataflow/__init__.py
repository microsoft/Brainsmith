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
- Constraint system: InterfaceConstraint for single-interface validation
- Dimension sources: DerivedDim/ScaledDim for explicit dimension derivation
"""

# Core types
from .types import Shape, ShapeHierarchy, DerivedDim, ScaledDim

# QONNX types (direct from QONNX)
from qonnx.core.datatype import DataType, BaseDataType

# Constraint system
from .constraints import (
    # Base classes
    InterfaceConstraint,
    DimensionConstraint,
    # Concrete constraints
    DatatypeConstraint,
    DimensionDivisible,
    DimensionMinValue,
    DimensionMaxValue,
)

# Core architecture - schemas consolidated in schemas.py
from .schemas import InputSchema, OutputSchema, KernelSchema

# Immutable models
from .models import (
    InterfaceModel,
    InputModel,
    OutputModel,
    KernelModel,
)

# Tensor context extraction
from .tensor_context import TensorContext, TensorInfo

# Template resolution
from .template_resolution import resolve_template

# Validation
from .validation import ValidationError



__all__ = [
    # Core types
    'Shape', 'DerivedDim', 'ScaledDim',

    # QONNX types
    'DataType', 'BaseDataType',

    # Unified constraint system
    'ShapeHierarchy', 'InterfaceConstraint', 'DimensionConstraint',
    # Interface constraints
    'DatatypeConstraint', 'DimensionDivisible', 'DimensionMinValue', 'DimensionMaxValue',

    # Core architecture
    'InputSchema', 'OutputSchema', 'KernelSchema',

    # Immutable models
    'InterfaceModel', 'InputModel', 'OutputModel', 'KernelModel',

    # Tensor context extraction
    'TensorContext', 'TensorInfo',

    # Template resolution
    'resolve_template',

    # Validation
    'ValidationError',
]