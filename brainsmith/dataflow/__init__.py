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
- KernelModelBuilder: Constructs immutable models from schemas + context
- KernelOp: Base class for all kernels (FINN adapter, delegates to builder)
- Immutable models: InputModel, OutputModel, KernelModel
- Unified Constraint system: Single abstraction for all validation (ONNX + kernel)
- Derivation system: DimensionSource/DatatypeSource for cross-interface derivation
- Inference system: InferencePattern for ONNX → HW layer inference

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
    # ONNX-specific constraints
    IsDynamic,
    IsStatic,
    HasLayout,
    NodeAttributeEquals,
    # Custom constraint
    Custom,
)

# Core architecture - unified schemas
from .schemas import (
    InputSchema,
    OutputSchema,
    KernelSchema,
)

# Immutable models (design space exploration)
from .models import (
    InterfaceDesignSpace,
    KernelDesignSpace,
    InterfaceConfiguration,
    KernelConfiguration,
)

# Builder (constructs models from schemas + context)
from .builder import (
    BuildContext,
    KernelModelBuilder,
)

# Template resolution
from .template_resolution import resolve_template, normalize_template

# Validation
from .validation import (
    ValidationError,
    ValidationContext,
    OnnxValidationContext,
    KernelValidationContext,
    DesignSpaceValidationContext,
    ConfigurationValidationContext,
)

# Kernel operator base class
from .kernel_op import KernelOp, KernelOpError

# Transformation system (unified)
from .transformation import (
    TransformationResult,
    transform_onnx_to_kernel,
)

# Inference infrastructure
from .inference import InferenceHelper


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

    # Unified Constraint system (works in both ONNX and kernel contexts)
    'Constraint',
    # Datatype constraints
    'DatatypeInteger', 'DatatypeFloat', 'DatatypeInRange', 'DatatypesEqual',
    # Shape constraints
    'ShapesEqual', 'DimensionDivisible', 'DimensionInRange', 'DimensionEquals',
    # ONNX-specific constraints
    'IsDynamic', 'IsStatic', 'HasLayout', 'NodeAttributeEquals',
    # Custom constraint
    'Custom',

    # Core architecture (unified schemas)
    'InputSchema', 'OutputSchema', 'KernelSchema',

    # Immutable models (design space exploration)
    'InterfaceDesignSpace', 'KernelDesignSpace',
    'InterfaceConfiguration', 'KernelConfiguration',

    # Builder
    'BuildContext', 'KernelModelBuilder',

    # Template resolution
    'resolve_template', 'normalize_template',

    # Validation
    'ValidationError',
    'ValidationContext',
    'OnnxValidationContext',
    'KernelValidationContext',
    'DesignSpaceValidationContext',
    'ConfigurationValidationContext',

    # Kernel operator base class
    'KernelOp', 'KernelOpError',

    # Transformation system (unified)
    'TransformationResult',
    'transform_onnx_to_kernel',

    # Inference infrastructure
    'InferenceHelper',
]