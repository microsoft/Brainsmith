"""
Core dataflow modeling components.

This module provides the core components for interface-wise dataflow modeling,
including base classes for auto-generated hardware operators.
"""

from .interface_types import InterfaceType
from .dataflow_interface import (
    DataflowInterface,
    DataTypeConstraint,
    ConstraintType,
    Constraint,
    DivisibilityConstraint,
    RangeConstraint,
)
from .dataflow_model import DataflowModel
from .block_chunking import (
    ChunkingType,
    BlockChunkingStrategy,
    DefaultChunkingStrategy,
    block_chunking,
    default_chunking,
    get_default_block_shape,
)
from .validation import (
    ValidationResult,
    ValidationSeverity,
    validate_dataflow_model,
)

# Auto-generated base classes
from .auto_hw_custom_op import AutoHWCustomOp
from .auto_rtl_backend import AutoRTLBackend
from .class_naming import (
    generate_class_name,
    generate_test_class_name,
    generate_backend_class_name,
)

__all__ = [
    # Interface types
    "InterfaceType",
    # Interfaces
    "DataflowInterface",
    "DataTypeConstraint",
    "ConstraintType",
    "Constraint",
    "DivisibilityConstraint",
    "RangeConstraint",
    # Model
    "DataflowModel",
    # Chunking
    "ChunkingType",
    "BlockChunkingStrategy",
    "DefaultChunkingStrategy",
    "block_chunking",
    "default_chunking",
    "get_default_block_shape",
    # Validation
    "ValidationResult",
    "ValidationSeverity",
    "validate_dataflow_model",
    # Auto-generated base classes
    "AutoHWCustomOp",
    "AutoRTLBackend",
    # Class naming utilities
    "generate_class_name",
    "generate_test_class_name",
    "generate_backend_class_name",
]
