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
from .types import DataType, Shape, InterfaceDirection, RelationType

# Relationships
from .relationships import DimensionRelationship

# Core architecture
from .input_definition import InputDefinition
from .output_definition import OutputDefinition
from .input_interface import InputInterface
from .output_interface import OutputInterface
from .kernel_definition import KernelDefinition
from .kernel_model import KernelModel

# Tiling functions
from .tiling_functions import (
    fixed_tiles,
    parameterized_tiles,
    adaptive_tiles,
    create_tiling_function
)

# Expression support
from .expressions import Expression, ParameterExpression

# Validation
from .validators import validate_shape, validate_block_dims

__all__ = [
    # Core types
    'DataType', 'Shape', 'InterfaceDirection', 'RelationType',
    
    # Relationships
    'DimensionRelationship',
    
    # Core architecture
    'InputDefinition', 'OutputDefinition',
    'InputInterface', 'OutputInterface',
    'KernelDefinition', 'KernelModel',
    
    # Tiling
    'fixed_tiles', 'parameterized_tiles', 'adaptive_tiles',
    'create_tiling_function',
    
    # Expressions
    'Expression', 'ParameterExpression',
    
    # Validation
    'validate_shape', 'validate_block_dims'
]