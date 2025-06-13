############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# Validation Framework for Interface-Wise Dataflow Modeling
############################################################################

"""Comprehensive validation framework for dataflow modeling components.

This module provides the core validation infrastructure used throughout the
Interface-Wise Dataflow Modeling framework to ensure correctness, provide
actionable error messages, and prevent common configuration mistakes.

Components:
- ValidationResult: Comprehensive validation results with errors and warnings
- ValidationError: Exception type for validation failures  
- Validator: Base class for all validators in the system
- Legacy compatibility for existing ValidationError/ValidationSeverity classes
"""

from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
from abc import ABC, abstractmethod
import logging

# Set up logger for this module
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation results"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationError:
    """Legacy validation error class - kept for backward compatibility"""
    component: str
    error_type: str
    message: str
    severity: ValidationSeverity
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Enhanced validation result with comprehensive error and warning support.
    
    Supports both the new simplified API and legacy ValidationError objects
    for backward compatibility with existing tests.
    """
    # New simplified API - primary interface
    is_valid: bool = True
    errors: List[Union[str, ValidationError]] = field(default_factory=list)
    warnings: List[Union[str, ValidationError]] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def add_error(self, message_or_error: Union[str, ValidationError], 
                  context: Optional[Dict[str, Any]] = None) -> None:
        """Add validation error with optional context.
        
        Supports both string messages (new API) and ValidationError objects (legacy).
        
        Args:
            message_or_error: Error message string or ValidationError object
            context: Optional structured context data for debugging
        """
        if isinstance(message_or_error, ValidationError):
            # Legacy API support
            if message_or_error.severity in [ValidationSeverity.ERROR, ValidationSeverity.CRITICAL]:
                self.errors.append(message_or_error)
                self.is_valid = False
            else:
                self.warnings.append(message_or_error)
        else:
            # New simplified API
            self.errors.append(message_or_error)
            self.is_valid = False
            
            if context:
                error_key = f"error_{len(self.errors)}"
                self.context[error_key] = context
                
        logger.debug(f"Validation error added: {message_or_error}")
    
    def add_warning(self, message_or_warning: Union[str, ValidationError],
                   context: Optional[Dict[str, Any]] = None) -> None:
        """Add validation warning with optional context.
        
        Args:
            message_or_warning: Warning message string or ValidationError object  
            context: Optional structured context data for debugging
        """
        self.warnings.append(message_or_warning)
        
        if isinstance(message_or_warning, str) and context:
            warning_key = f"warning_{len(self.warnings)}"
            self.context[warning_key] = context
            
        logger.debug(f"Validation warning added: {message_or_warning}")
    
    def has_warnings(self) -> bool:
        """Check if there are any warnings"""
        return len(self.warnings) > 0
    
    @property
    def success(self) -> bool:
        """Alias for is_valid() to match test expectations"""
        return self.is_valid
    
    def merge(self, other: 'ValidationResult') -> 'ValidationResult':
        """Merge multiple validation results.
        
        Args:
            other: Another ValidationResult to merge
            
        Returns:
            New ValidationResult containing merged results
        """
        if hasattr(other, 'errors') and hasattr(other, 'warnings'):
            # Handle both new and legacy ValidationResult formats
            self.errors.extend(other.errors)
            self.warnings.extend(other.warnings)
            self.is_valid = self.is_valid and getattr(other, 'is_valid', other.success)
            
            # Merge context if available
            if hasattr(other, 'context'):
                for key, value in other.context.items():
                    if key in self.context:
                        self.context[f"merged_{key}"] = value
                    else:
                        self.context[key] = value
        
        return self
    
    def get_summary(self) -> str:
        """Get human-readable validation summary.
        
        Returns:
            Formatted summary string with errors and warnings
        """
        if self.is_valid and not self.warnings:
            return "✅ Validation passed with no issues"
            
        summary_lines = []
        
        if not self.is_valid:
            summary_lines.append(f"❌ Validation failed with {len(self.errors)} error(s):")
            for i, error in enumerate(self.errors, 1):
                error_msg = error.message if isinstance(error, ValidationError) else str(error)
                summary_lines.append(f"  {i}. {error_msg}")
        else:
            summary_lines.append("✅ Validation passed")
            
        if self.warnings:
            summary_lines.append(f"⚠️  {len(self.warnings)} warning(s):")
            for i, warning in enumerate(self.warnings, 1):
                warning_msg = warning.message if isinstance(warning, ValidationError) else str(warning)
                summary_lines.append(f"  {i}. {warning_msg}")
                
        return "\n".join(summary_lines)
    
    def get_errors(self) -> List[str]:
        """Get list of error messages as strings."""
        return [error.message if isinstance(error, ValidationError) else str(error) 
                for error in self.errors]
    
    def get_warnings(self) -> List[str]:
        """Get list of warning messages as strings."""
        return [warning.message if isinstance(warning, ValidationError) else str(warning)
                for warning in self.warnings]


class ValidationException(Exception):
    """Exception raised for validation failures with detailed context.
    
    Provides rich error information including the full ValidationResult
    that triggered the exception, enabling detailed error handling and
    debugging.
    """
    
    def __init__(self, result: ValidationResult, additional_message: Optional[str] = None):
        """Initialize ValidationException with ValidationResult.
        
        Args:
            result: ValidationResult containing detailed error information
            additional_message: Optional additional context message
        """
        self.result = result
        
        if additional_message:
            message = f"{additional_message}\n{result.get_summary()}"
        else:
            message = result.get_summary()
            
        super().__init__(message)


class Validator(ABC):
    """Base class for all validators in the system.
    
    Provides a common interface for validation components throughout
    the Interface-Wise Dataflow Modeling framework. Subclasses implement
    specific validation logic for different types of objects.
    """
    
    @abstractmethod
    def validate(self, obj: Any) -> ValidationResult:
        """Validate an object and return detailed results.
        
        Args:
            obj: Object to validate (type depends on validator implementation)
            
        Returns:
            ValidationResult with detailed validation feedback
        """
        pass
    
    def validate_and_raise(self, obj: Any) -> None:
        """Validate an object and raise ValidationException if invalid.
        
        Convenience method for cases where exceptions are preferred
        over return value checking.
        
        Args:
            obj: Object to validate
            
        Raises:
            ValidationException: If validation fails
        """
        result = self.validate(obj)
        if not result.is_valid:
            raise ValidationException(result)


def validate_positive_integers(values: List[int], name: str) -> ValidationResult:
    """Utility function to validate that all values are positive integers.
    
    Args:
        values: List of integers to validate
        name: Name of the values for error messages
        
    Returns:
        ValidationResult indicating if all values are positive
    """
    result = ValidationResult(True)
    
    if not values:
        result.add_error(f"{name} cannot be empty")
        return result
        
    for i, value in enumerate(values):
        if not isinstance(value, int):
            result.add_error(f"{name}[{i}] must be an integer, got {type(value).__name__}")
        elif value <= 0:
            result.add_error(f"{name}[{i}] must be positive, got {value}")
            
    return result


def validate_dimension_relationships(tensor_dims: List[int], 
                                   block_dims: List[int],
                                   stream_dims: List[int]) -> ValidationResult:
    """Utility function to validate dimension relationships follow axioms.
    
    Validates that the core relationship tensor_dims = num_blocks × block_dims
    and that streaming dimensions are valid.
    
    Args:
        tensor_dims: Full tensor dimensions
        block_dims: Block processing dimensions  
        stream_dims: Streaming dimensions per cycle
        
    Returns:
        ValidationResult indicating if relationships are valid
    """
    result = ValidationResult(True)
    
    # Check dimension count consistency
    if len(tensor_dims) != len(block_dims):
        result.add_error(
            f"Dimension count mismatch: tensor_dims has {len(tensor_dims)} dimensions, "
            f"block_dims has {len(block_dims)} dimensions"
        )
    
    if len(block_dims) != len(stream_dims):
        result.add_error(
            f"Dimension count mismatch: block_dims has {len(block_dims)} dimensions, "
            f"stream_dims has {len(stream_dims)} dimensions"
        )
        
    # If dimension counts match, validate relationships
    if len(tensor_dims) == len(block_dims) == len(stream_dims):
        for i in range(len(tensor_dims)):
            # Validate tensor_dims = num_blocks × block_dims relationship
            if tensor_dims[i] % block_dims[i] != 0:
                result.add_error(
                    f"Dimension {i}: tensor_dims[{i}]={tensor_dims[i]} is not divisible by "
                    f"block_dims[{i}]={block_dims[i]}"
                )
                
            # Validate stream_dims ≤ block_dims relationship  
            if stream_dims[i] > block_dims[i]:
                result.add_error(
                    f"Dimension {i}: stream_dims[{i}]={stream_dims[i]} cannot be larger than "
                    f"block_dims[{i}]={block_dims[i]}"
                )
                
            # Validate block_dims divisible by stream_dims
            if block_dims[i] % stream_dims[i] != 0:
                result.add_warning(
                    f"Dimension {i}: block_dims[{i}]={block_dims[i]} is not divisible by "
                    f"stream_dims[{i}]={stream_dims[i]}. This may cause inefficient processing."
                )
    
    return result


def create_validation_result(is_valid: bool = True) -> ValidationResult:
    """Create a new ValidationResult with the specified validity.
    
    Args:
        is_valid: Initial validation state
        
    Returns:
        New ValidationResult instance
    """
    return ValidationResult(is_valid=is_valid)


def create_divisibility_error(dimension: int, dividend: int, divisor: int, 
                            dimension_name: str = "dimension") -> ValidationError:
    """Create a validation error for divisibility issues.
    
    Args:
        dimension: Dimension index
        dividend: Value that should be divisible
        divisor: Value to divide by
        dimension_name: Name of the dimension for error context
        
    Returns:
        ValidationError for divisibility issue
    """
    return ValidationError(
        component="dimension_validation",
        error_type="divisibility_error",
        message=f"{dimension_name} {dimension}: {dividend} is not divisible by {divisor}",
        severity=ValidationSeverity.ERROR,
        context={
            "dimension": dimension,
            "dividend": dividend,
            "divisor": divisor,
            "dimension_name": dimension_name
        }
    )


def create_range_error(value: int, min_value: int, max_value: int,
                      value_name: str = "value") -> ValidationError:
    """Create a validation error for range violations.
    
    Args:
        value: Value that's out of range
        min_value: Minimum allowed value
        max_value: Maximum allowed value
        value_name: Name of the value for error context
        
    Returns:
        ValidationError for range violation
    """
    return ValidationError(
        component="range_validation",
        error_type="range_error", 
        message=f"{value_name} {value} is outside allowed range [{min_value}, {max_value}]",
        severity=ValidationSeverity.ERROR,
        context={
            "value": value,
            "min_value": min_value,
            "max_value": max_value,
            "value_name": value_name
        }
    )


def create_datatype_error(datatype: str, expected_types: List[str],
                         interface_name: str = "interface") -> ValidationError:
    """Create a validation error for datatype mismatches.
    
    Args:
        datatype: Actual datatype that was invalid
        expected_types: List of allowed datatypes
        interface_name: Name of the interface for error context
        
    Returns:
        ValidationError for datatype mismatch
    """
    return ValidationError(
        component="datatype_validation",
        error_type="datatype_error",
        message=f"{interface_name} datatype '{datatype}' not in allowed types: {expected_types}",
        severity=ValidationSeverity.ERROR,
        context={
            "datatype": datatype,
            "expected_types": expected_types,
            "interface_name": interface_name
        }
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
    
    # Enhanced validation implementation
    from .dataflow_interface import DataflowInterface
    from .dataflow_model import DataflowModel
    
    if hasattr(model, 'interfaces'):
        for interface in model.interfaces:
            if hasattr(interface, 'validate'):
                interface_result = interface.validate()
                result.merge(interface_result)
    
    return result
