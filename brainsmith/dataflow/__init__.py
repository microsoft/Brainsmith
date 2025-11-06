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
- KernelSchema: Kernel definition with unified constraints
- DesignSpaceBuilder: Constructs immutable models from schemas + context
- KernelOp: Base class for all kernels (FINN adapter, delegates to builder)
- Immutable models (two-phase construction):
  - Design Space: InterfaceDesignSpace, KernelDesignSpace
  - Design Point: InterfaceDesignPoint, KernelDesignPoint
- Unified Constraint system: Single abstraction for all validation (ONNX + kernel)
- Union type system: Simple derivation via string shorthand, tuples, and callables
- Schema helpers: Dimension/datatype derivation and arithmetic range computation (spec_helpers module)
- Inference system: InferencePattern for ONNX → HW layer inference

Architecture:
    KernelSchema (defines + validates) → DesignSpaceBuilder (constructs) → KernelDesignSpace → KernelDesignPoint
"""

# Core types
from .types import (
    Shape,
    ShapeHierarchy,
    FULL_DIM,
    FULL_SHAPE,
    VALUE_OPTIMIZED,
)

# Schema helpers (for building kernel schemas)
from .spec_helpers import (
    # Dimension/datatype derivation
    derive_dim,
    derive_datatype,
    constant_datatype,
    value_optimized_datatype,
    # Arithmetic range computation
    compute_add_range,
    compute_sub_range,
    compute_mul_range,
    compute_min_range,
    compute_max_range,
    smallest_datatype_for_range,
    # Context-aware datatype builders
    add_datatype,
    sub_datatype,
    mul_datatype,
    min_datatype,
    max_datatype,
)

# Broadcasting helpers (for elementwise operations)
from .broadcast_helpers import (
    BroadcastInfo,
    compute_broadcast_info,
)

# Inference helpers (for kernel inference preprocessing)
from .inference_helpers import (
    lift_scalar_to_rank1,
)

# QONNX types (direct from QONNX)
from qonnx.core.datatype import DataType, BaseDataType

# Unified Constraint system (works in both ONNX and kernel contexts)
from .constraints import (
    # Base class
    Constraint,
    # Datatype constraints
    DatatypeInteger,
    DatatypeFloat,
    DatatypeInRange,
    DatatypesEqual,
    # Shape constraints
    ShapesEqual,
    DimensionDivisible,
    DimensionInRange,
    DimensionEquals,
    TensorDimMatches,
    TensorSizeMatches,
    # ONNX-specific constraints
    IsDynamic,
    IsStatic,
    HasLayout,
    NodeAttributeEquals,
    # Attribute comparison
    AttrCompare,
    # Custom constraint
    CustomConstraint,
)

# Backwards compatibility alias
Custom = CustomConstraint

# Core architecture - unified schemas
from .schemas import (
    DSEDimension,
    InputSchema,
    OutputSchema,
    KernelSchema,
)

# DSE navigation
from .ordered_dimension import OrderedDimension

# Immutable models (design space exploration)
from .dse_models import (
    InterfaceDesignSpace,
    KernelDesignSpace,
    InterfaceDesignPoint,
    KernelDesignPoint,
)

# Builder (constructs models from schemas + context)
from .builder import (
    BuildContext,
    DesignSpaceBuilder,
)

# Template resolution
from .template_resolution import resolve_template, normalize_template

# Validation
from .validation import (
    ValidationError,
    DesignSpaceValidationContext,
    ConfigurationValidationContext,
)

# Kernel operator base class
from .kernel_op import KernelOp, KernelOpError

# Transformation system (unified)
from .transformation import TransformationResult


__all__ = [
    # === Core Public API ===
    # Main classes
    'KernelOp',
    'KernelOpError',
    'KernelSchema',
    'InputSchema',
    'OutputSchema',

    # Builder (for DSE and kernel construction)
    'DesignSpaceBuilder',
    'BuildContext',

    # Immutable models (design space exploration)
    'KernelDesignSpace',
    'KernelDesignPoint',
    'InterfaceDesignSpace',
    'InterfaceDesignPoint',
    'OrderedDimension',  # For ordered dimension navigation

    # Validation
    'Constraint',
    'ValidationError',
    'DesignSpaceValidationContext',
    'ConfigurationValidationContext',

    # Transformation
    'TransformationResult',

    # Inference helpers
    'lift_scalar_to_rank1',  # For scalar input normalization in kernel inference

    # Essential types
    'Shape',
    'ShapeHierarchy',
    'FULL_DIM',
    'FULL_SHAPE',

    # QONNX type re-exports (for convenience)
    'DataType',
    'BaseDataType',

    # === Advanced API ===
    # For advanced users - specific constraint classes require explicit import:
    #   from brainsmith.dataflow.constraints import DatatypeInteger, ShapesEqual, ...
    # For schema helpers (dimension/datatype derivation, range computation) - require explicit import:
    #   from brainsmith.dataflow.spec_helpers import derive_dim, add_datatype, compute_add_range, ...
    # For template resolution - require explicit import:
    #   from brainsmith.dataflow.template_resolution import resolve_template, normalize_template
]
