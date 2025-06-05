"""
Validation framework for dataflow modeling

This module provides standardized validation error handling and constraint
violation tracking for the Interface-Wise Dataflow Modeling framework.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Any, Union

class ValidationSeverity(Enum):
    """Severity levels for validation errors"""
    ERROR = "error"      # Blocks code generation
    WARNING = "warning"  # Should be addressed but not blocking
    INFO = "info"        # Informational only

@dataclass
class ValidationError:
    """Standardized validation error representation"""
    component: str              # Component where error occurred
    error_type: str            # Error classification
    message: str               # Human-readable error description
    severity: ValidationSeverity  # ERROR, WARNING, INFO
    context: Dict[str, Any]    # Additional context for debugging

    def __str__(self) -> str:
        return f"{self.severity.value.upper()}: {self.component}: {self.message}"

@dataclass
class ValidationResult:
    """Complete validation result set"""
    success: bool
    errors: List[ValidationError]
    warnings: List[ValidationError]
    info: List[ValidationError]
    
    def has_blocking_errors(self) -> bool:
        """Check if there are any blocking errors"""
        return any(err.severity == ValidationSeverity.ERROR for err in self.errors)
    
    def add_error(self, error: ValidationError) -> None:
        """Add an error to the appropriate list based on severity"""
        if error.severity == ValidationSeverity.ERROR:
            self.errors.append(error)
        elif error.severity == ValidationSeverity.WARNING:
            self.warnings.append(error)
        else:
            self.info.append(error)
        
        # Update success status if we have blocking errors
        if self.has_blocking_errors():
            self.success = False
    
    def merge(self, other: 'ValidationResult') -> None:
        """Merge another validation result into this one"""
        self.errors.extend(other.errors)
        self.warnings.extend(other.warnings)
        self.info.extend(other.info)
        
        # Update success status
        if self.has_blocking_errors():
            self.success = False
    
    def summary(self) -> str:
        """Generate a summary string of validation results"""
        total_errors = len(self.errors)
        total_warnings = len(self.warnings)
        total_info = len(self.info)
        
        status = "PASS" if self.success else "FAIL"
        return f"Validation {status}: {total_errors} errors, {total_warnings} warnings, {total_info} info"

@dataclass
class ConstraintViolation:
    """Specialized constraint violation representation"""
    interface_name: str
    constraint_type: str
    constraint_details: Dict[str, Any]
    violation_message: str
    
    def to_validation_error(self) -> ValidationError:
        """Convert constraint violation to validation error"""
        return ValidationError(
            component=f"interface.{self.interface_name}",
            error_type="constraint_violation",
            message=f"Constraint violation ({self.constraint_type}): {self.violation_message}",
            severity=ValidationSeverity.ERROR,
            context={
                "interface": self.interface_name,
                "constraint_type": self.constraint_type,
                "constraint_details": self.constraint_details
            }
        )

# Standard validation error creators
def create_divisibility_error(interface_name: str, dimension: str, dividend: int, divisor: int) -> ValidationError:
    """Create a standardized divisibility constraint violation error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="divisibility_violation",
        message=f"{dimension} ({dividend}) must be divisible by divisor ({divisor})",
        severity=ValidationSeverity.ERROR,
        context={
            "interface": interface_name,
            "dimension": dimension,
            "dividend": dividend,
            "divisor": divisor
        }
    )

def create_range_error(interface_name: str, parameter: str, value: int, min_val: int, max_val: int) -> ValidationError:
    """Create a standardized range constraint violation error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="range_violation",
        message=f"{parameter} ({value}) must be in range [{min_val}, {max_val}]",
        severity=ValidationSeverity.ERROR,
        context={
            "interface": interface_name,
            "parameter": parameter,
            "value": value,
            "min_value": min_val,
            "max_value": max_val
        }
    )

def create_datatype_error(interface_name: str, target_type: str, allowed_types: List[str]) -> ValidationError:
    """Create a standardized datatype constraint violation error"""
    return ValidationError(
        component=f"interface.{interface_name}",
        error_type="datatype_violation",
        message=f"Datatype '{target_type}' not allowed. Allowed types: {allowed_types}",
        severity=ValidationSeverity.ERROR,
        context={
            "interface": interface_name,
            "target_type": target_type,
            "allowed_types": allowed_types
        }
    )

# Factory for creating empty validation results
def create_validation_result() -> ValidationResult:
    """Create an empty validation result with success=True"""
    return ValidationResult(
        success=True,
        errors=[],
        warnings=[],
        info=[]
    )
