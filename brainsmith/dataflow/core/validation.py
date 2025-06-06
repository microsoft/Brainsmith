"""
Validation utilities for dataflow modeling.

This module provides validation framework for dataflow models and interfaces.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum


class ValidationSeverity(Enum):
    """Severity levels for validation results"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Represents a validation error"""
    component: str
    error_type: str
    message: str
    severity: ValidationSeverity
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Container for validation results"""
    errors: List[ValidationError] = field(default_factory=list)
    warnings: List[ValidationError] = field(default_factory=list)
    
    def add_error(self, error: ValidationError):
        """Add an error to the validation result"""
        if error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
            self.errors.append(error)
        else:
            self.warnings.append(error)
    
    def is_valid(self) -> bool:
        """Check if validation passed (no errors)"""
        return len(self.errors) == 0
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0
    
    @property
    def success(self) -> bool:
        """Alias for is_valid() to match test expectations"""
        return self.is_valid()
    
    def merge(self, other: 'ValidationResult'):
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)


def create_validation_result() -> ValidationResult:
    """Create a new validation result"""
    return ValidationResult()


def create_divisibility_error(interface_name: str, param_name: str, 
                             value: int, divisor: int) -> ValidationError:
    """Create a divisibility constraint error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="divisibility_constraint",
        message=f"{param_name} ({value}) must be divisible by {divisor}",
        severity=ValidationSeverity.ERROR,
        context={"value": value, "divisor": divisor}
    )


def create_range_error(interface_name: str, param_name: str,
                      value: int, min_val: int, max_val: int) -> ValidationError:
    """Create a range constraint error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="range_constraint",
        message=f"{param_name} ({value}) must be between {min_val} and {max_val}",
        severity=ValidationSeverity.ERROR,
        context={"value": value, "min": min_val, "max": max_val}
    )


def create_datatype_error(interface_name: str, datatype: str,
                         allowed_types: List[str]) -> ValidationError:
    """Create a datatype constraint error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="datatype_constraint",
        message=f"Datatype {datatype} not allowed. Must be one of: {allowed_types}",
        severity=ValidationSeverity.ERROR,
        context={"datatype": datatype, "allowed": allowed_types}
    )


def validate_dataflow_model(model) -> ValidationResult:
    """
    Validate a complete dataflow model.
    
    Args:
        model: DataflowModel to validate
        
    Returns:
        ValidationResult with any errors or warnings
    """
    result = create_validation_result()
    
    # Placeholder - actual validation would go here
    # This would validate all interfaces, constraints, etc.
    
    return result
