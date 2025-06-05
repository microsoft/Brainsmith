"""
Core dataflow framework components

This module contains the foundational classes and data structures for the
Interface-Wise Dataflow Modeling framework.
"""

from .dataflow_interface import (
    DataflowInterface,
    DataflowInterfaceType,
    DataflowDataType,
    DataTypeConstraint,
    Constraint,
    ConstraintType,
    DivisibilityConstraint,
    RangeConstraint
)

from .dataflow_model import (
    DataflowModel,
    InitiationIntervals,
    ParallelismBounds,
    ParallelismConfiguration
)

from .validation import (
    ValidationError,
    ValidationResult,
    ValidationSeverity,
    ConstraintViolation
)

__all__ = [
    # Interface classes
    "DataflowInterface",
    "DataflowInterfaceType",
    "DataflowDataType",
    "DataTypeConstraint",
    "Constraint",
    "ConstraintType", 
    "DivisibilityConstraint",
    "RangeConstraint",
    
    # Model classes
    "DataflowModel",
    "InitiationIntervals",
    "ParallelismBounds",
    "ParallelismConfiguration",
    
    # Validation classes
    "ValidationError",
    "ValidationResult",
    "ValidationSeverity",
    "ConstraintViolation"
]
