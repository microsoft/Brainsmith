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
- Relationship system: InterfaceRelationship for cross-interface constraints
"""

# Core types
from .types import Shape, ShapeHierarchy

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

# Relationship system
from .relationships import (
    # Base class
    InterfaceRelationship,
    # Concrete relationships
    DimensionEquality,
    DimensionDivisibleBy,
    DimensionScaled,
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



__all__ = [
    # Core types
    'Shape',

    # QONNX types
    'DataType', 'BaseDataType',

    # Unified constraint system
    'ShapeHierarchy', 'InterfaceConstraint', 'DimensionConstraint', 'InterfaceRelationship',
    # Interface constraints
    'DatatypeConstraint', 'DimensionDivisible', 'DimensionMinValue', 'DimensionMaxValue',
    # Interface relationships
    'DimensionEquality', 'DimensionDivisibleBy', 'DimensionScaled',

    # Core architecture
    'InputSchema', 'OutputSchema', 'KernelSchema',

    # Immutable models
    'InterfaceModel', 'InputModel', 'OutputModel', 'KernelModel',

    # Tensor context extraction
    'TensorContext', 'TensorInfo',
]