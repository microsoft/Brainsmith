"""
Core dataflow modeling components.

This module provides the core components for interface-wise dataflow modeling,
including base classes for auto-generated hardware operators.
"""

from .dataflow_interface import (
    DataflowInterface,
    DataflowInterfaceType,
    DataflowDataType,
    DataTypeConstraint,
    ConstraintType,
    Constraint,
    DivisibilityConstraint,
    RangeConstraint,
)
from .dataflow_model import DataflowModel
from .tensor_chunking import (
    TensorChunk,
    ChunkingStrategy,
    calculate_tensor_chunks,
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
    # Interfaces
    "DataflowInterface",
    "DataflowInterfaceType",
    "DataflowDataType",
    "DataTypeConstraint",
    "ConstraintType",
    "Constraint",
    "DivisibilityConstraint",
    "RangeConstraint",
    # Model
    "DataflowModel",
    # Chunking
    "TensorChunk",
    "ChunkingStrategy",
    "calculate_tensor_chunks",
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
