############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Unified Kernel Modeling Framework for FPGA AI Accelerators"""

from .core.types import Shape, RaggedShape, InterfaceDirection, DataType
from .core.interface import Interface
from .core.relationships import (
    RelationType, DimensionRelationship, ArchitecturalConstraint, 
    ParameterDependency, ConstraintViolation, ValidationResult
)
from .core.kernel import Kernel
from .core.graph import DataflowGraph, DataflowEdge

# New Definition/Model architecture
from .core.base import (
    BaseDefinition, BaseModel, ParameterBinding, ValidationContext,
    ModelFactory, DefinitionRegistry, DEFINITION_REGISTRY
)
from .core.interface_definition import InterfaceDefinition
from .core.interface_model import InterfaceModel
from .core.kernel_definition import KernelDefinition
from .core.kernel_model import KernelModel

__version__ = "0.1.0"

__all__ = [
    # Types
    "Shape",
    "RaggedShape", 
    "InterfaceDirection",
    "DataType",
    # Legacy core classes
    "Interface",
    "Kernel",
    "DataflowGraph",
    "DataflowEdge",
    # Relationships and constraints
    "RelationType",
    "DimensionRelationship",
    "ArchitecturalConstraint",
    "ParameterDependency",
    "ConstraintViolation",
    "ValidationResult",
    # New Definition/Model architecture
    "BaseDefinition",
    "BaseModel",
    "ParameterBinding",
    "ValidationContext",
    "ModelFactory",
    "DefinitionRegistry",
    "DEFINITION_REGISTRY",
    "InterfaceDefinition",
    "InterfaceModel", 
    "KernelDefinition",
    "KernelModel",
]