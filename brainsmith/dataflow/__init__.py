############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Core dataflow modeling components

This module provides the core classes for modeling dataflow kernels with
the SDIM (Streaming Dimensions) architecture.

Key Components:
- InputDefinition/OutputDefinition: Schema for interfaces
- InputInterface/OutputInterface: Runtime models with SDIM support
- KernelDefinition/KernelModel: Kernel-level abstractions
- RelationType: Including DEPENDENT relationship for dimension constraints
- Tiling functions: For block dimension configuration
"""

# Core types
from .types import Shape

# Relationships
from .relationships import DimensionRelationship, RelationType

# QONNX types (unified type system)
from .qonnx_types import (
    BaseDataType,
    create_simple_datatype,
    datatype_from_string,
)

# Constraint types
from .constraint_types import (
    DatatypeConstraintGroup,
    validate_datatype_against_constraints,
)

# Core architecture
from .input_definition import InputDefinition
from .output_definition import OutputDefinition
from .input_interface import InputInterface
from .output_interface import OutputInterface
from .kernel_definition import KernelDefinition
from .kernel_model import KernelModel

# Tiling functions and configuration
from .tiling_functions import (
    fixed_tiles,
    adaptive_parameterized_tiles,
    parameterized_tiles,
    adaptive_tiles,
    full_tensor
)
from .tiling_spec import TilingSpec, TilingExpr, TilingExprType
from .tiling_strategy import TilingStrategy, TilingOrder, TilingResult



__all__ = [
    # Core types
    'Shape',
    
    # Relationships
    'DimensionRelationship', 'RelationType',
    
    # QONNX types (unified type system)
    'BaseDataType', 'create_simple_datatype', 'datatype_from_string',
    
    # Constraint types
    'DatatypeConstraintGroup', 'validate_datatype_against_constraints',
    
    # Core architecture
    'InputDefinition', 'OutputDefinition',
    'InputInterface', 'OutputInterface',
    'KernelDefinition', 'KernelModel',
    
    # Tiling functions
    'fixed_tiles', 'adaptive_parameterized_tiles', 'parameterized_tiles', 'adaptive_tiles',
    'full_tensor',
    'TilingSpec', 'TilingExpr', 'TilingExprType',
    'TilingStrategy', 'TilingOrder', 'TilingResult',
]