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
# QONNX types (direct from QONNX)
from qonnx.core.datatype import BaseDataType, DataType

# Builder (constructs models from schemas + context)
from .builder import (
    BuildContext,
    DesignSpaceBuilder,
)

# Unified Constraint system (works in both ONNX and kernel contexts)
from .constraints import (
    # Base class
    Constraint,
)

# Immutable models (design space exploration)
from .dse_models import (
    InterfaceDesignPoint,
    InterfaceDesignSpace,
    KernelDesignPoint,
    KernelDesignSpace,
)

# Schema helpers (for building kernel schemas)
# Broadcasting helpers (for elementwise operations)
# Inference helpers (for kernel inference preprocessing)
from .inference_helpers import (
    lift_scalar_to_rank1,
)

# Kernel operator base class
from .kernel_op import KernelOp, KernelOpError

# DSE navigation
from .ordered_parameter import OrderedParameter

# Core architecture - unified schemas
from .schemas import (
    InputSchema,
    KernelSchema,
    OutputSchema,
    ParameterSpec,
)

# Transformation system (unified)
from .transformation import TransformationResult
from .types import (
    FULL_DIM,
    FULL_SHAPE,
    Shape,
    ShapeHierarchy,
)

# Template resolution
# Validation
from .validation import (
    ConfigurationValidationContext,
    DesignSpaceValidationContext,
    ValidationError,
)

__all__ = [
    # === Core Public API ===
    # Main classes
    'KernelOp',
    'KernelOpError',
    'KernelSchema',
    'InputSchema',
    'OutputSchema',
    'ParameterSpec',

    # Builder (for DSE and kernel construction)
    'DesignSpaceBuilder',
    'BuildContext',

    # Immutable models (design space exploration)
    'KernelDesignSpace',
    'KernelDesignPoint',
    'InterfaceDesignSpace',
    'InterfaceDesignPoint',
    'OrderedParameter',  # For ordered parameter navigation

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
