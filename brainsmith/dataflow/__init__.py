"""
Interface-Wise Dataflow Modeling Framework

This framework provides a unified abstraction layer for hardware kernel design that 
simplifies the complexity of integrating custom RTL implementations into the 
FINN/Brainsmith ecosystem through standardized interface-based modeling and 
automated code generation.
"""

__version__ = "0.1.0"

# Core framework components
from .core.dataflow_interface import DataflowInterface, DataflowInterfaceType, DataflowDataType
from .core.dataflow_model import DataflowModel, InitiationIntervals, ParallelismBounds
from .core.validation import ValidationError, ValidationResult, ValidationSeverity

__all__ = [
    "DataflowInterface",
    "DataflowInterfaceType", 
    "DataflowDataType",
    "DataflowModel",
    "InitiationIntervals",
    "ParallelismBounds",
    "ValidationError",
    "ValidationResult", 
    "ValidationSeverity"
]
