############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Core dataflow modeling components

This module provides the core classes for modeling dataflow kernels.

**Key Principles:**
- Schemas define STRUCTURE, not storage
- ModelWrapper is the single source of truth for shapes
- Only datatypes and user parameters persist in nodeattrs

Key Components:
- InputSchema/OutputSchema: Schema definitions for kernel interfaces
- KernelSchema: Kernel definition with input/output schemas + validates models
- KernelModelBuilder: Constructs immutable models from schemas + context
- KernelOp: Base class for all kernels (FINN adapter, delegates to builder)
- Immutable models: InputModel, OutputModel, KernelModel
- Constraint system: InterfaceConstraint for single-interface validation
- Derivation system: DimensionSource/DatatypeSource for cross-interface derivation
- Relationship system: InterfaceRelationship for cross-interface validation

Architecture:
    KernelSchema (defines + validates) → KernelModelBuilder (constructs) → KernelModel
"""

# Core types
from .types import Shape, ShapeHierarchy, FULL_DIM

# QONNX types (direct from QONNX)
from qonnx.core.datatype import DataType, BaseDataType

# Extensible derivation system
from .dimension_sources import (
    DimensionSource,
    # Built-in patterns
    DerivedDim,
    ScaledDim,
    SumDims,
    MaxDim,
    ComputedDim,
)

from .datatype_sources import (
    DatatypeSource,
    DerivedDatatype,
    WidenedDatatype,
    UnionDatatype,
    ComputedDatatype,
)

# Extensible relationship validation system
from .relationships import (
    InterfaceRelationship,
    DatatypesEqual,
    DimensionsEqual,
    CustomRelationship,
)

# Constraint system (single-interface validation)
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

# Builder (constructs models from schemas + context)
from .builder import (
    BuildContext,
    KernelModelBuilder,
)

# Template resolution
from .template_resolution import resolve_template

# Validation
from .validation import ValidationError

# Kernel operator base class
from .kernel_op import KernelOp, KernelOpError


__all__ = [
    # Core types
    'Shape', 'ShapeHierarchy', 'FULL_DIM',

    # QONNX types
    'DataType', 'BaseDataType',

    # Extensible derivation system - dimension sources
    'DimensionSource',
    'DerivedDim', 'ScaledDim',  # Most common patterns
    'SumDims', 'MaxDim', 'ComputedDim',  # Additional patterns

    # Extensible derivation system - datatype sources
    'DatatypeSource',
    'DerivedDatatype', 'WidenedDatatype', 'UnionDatatype', 'ComputedDatatype',

    # Extensible relationship validation system
    'InterfaceRelationship',
    'DatatypesEqual', 'DimensionsEqual', 'CustomRelationship',

    # Constraint system (single-interface validation)
    'InterfaceConstraint', 'DimensionConstraint',
    'DatatypeConstraint', 'DimensionDivisible', 'DimensionMinValue', 'DimensionMaxValue',

    # Core architecture
    'InputSchema', 'OutputSchema', 'KernelSchema',

    # Immutable models
    'InterfaceModel', 'InputModel', 'OutputModel', 'KernelModel',

    # Builder
    'BuildContext', 'KernelModelBuilder',

    # Template resolution
    'resolve_template',

    # Validation
    'ValidationError',

    # Kernel operator base class
    'KernelOp', 'KernelOpError',
]