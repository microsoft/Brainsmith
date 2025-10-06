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
from .relationships import (
    DimensionRelationship,
    RelationType,
    # Relationship builder functions
    equal_shapes,
    equal_dimension,
    divisible_dimension,
    scaled_dimension,
)

# QONNX types (direct from QONNX)
from qonnx.core.datatype import DataType, BaseDataType

# Constraint types
from .constraint_types import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints,
)

# Dimension constraints
from .dimension_constraints import (
    DimensionConstraint,
    # Atomic constraints
    DivisibleConstraint,
    MinValueConstraint,
    MaxValueConstraint,
    RangeConstraint,
    PowerOfTwoConstraint,
    # Cross-interface constraints
    EqualityConstraint,
    DivisibleByDimensionConstraint,
    ScaledEqualityConstraint,
)

# Core architecture - schemas consolidated in schemas.py
from .schemas import InputSchema, OutputSchema, KernelSchema

# Immutable models
from .models import (
    InputModel,
    OutputModel,
    KernelModel,
)

# Tensor context extraction
from .tensor_context import TensorContext, TensorInfo

# Validation layer
from .validation import (
    KernelValidator,
    validate_kernel_schema,
    validate_kernel_model
)



__all__ = [
    # Core types
    'Shape',

    # Relationships
    'DimensionRelationship', 'RelationType',
    # Relationship builders
    'equal_shapes', 'equal_dimension', 'divisible_dimension', 'scaled_dimension',

    # QONNX types
    'DataType', 'BaseDataType',

    # Constraint types
    'DatatypeConstraintGroup', 'validate_datatype_against_constraints',

    # Dimension constraints
    'DimensionConstraint',
    # Atomic constraints
    'DivisibleConstraint', 'MinValueConstraint', 'MaxValueConstraint',
    'RangeConstraint', 'PowerOfTwoConstraint',
    # Cross-interface constraints
    'EqualityConstraint', 'DivisibleByDimensionConstraint', 'ScaledEqualityConstraint',

    # Core architecture
    'InputSchema', 'OutputSchema', 'KernelSchema',

    # Immutable models
    'InputModel', 'OutputModel', 'KernelModel',

    # Tensor context extraction
    'TensorContext', 'TensorInfo',

    # Validation
    'KernelValidator', 'validate_kernel_schema', 'validate_kernel_model',
]