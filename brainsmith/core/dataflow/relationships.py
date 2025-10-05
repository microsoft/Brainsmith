############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Native relationship and constraint types for kernel modeling"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Union, Optional, List, Callable
from abc import ABC, abstractmethod


class RelationType(Enum):
    """Types of relationships between interface dimensions
    
    Simplified to only include actively used relationship types:
    - EQUAL: Dimensions must be exactly equal
    - DEPENDENT: Target dimension depends on source (with optional scaling)
    - MULTIPLE: Target is a multiple of source (kept for future use)
    """
    EQUAL = "equal"
    MULTIPLE = "multiple"
    DEPENDENT = "dependent"  # Dimension-specific dependency


@dataclass(frozen=True)
class DimensionRelationship:
    """Relationship between dimensions of different interfaces
    
    Represents constraints like:
    - matrix[1] == vector[0] (matrix columns equal vector size)
    - output.total == 4 * input.total (output 4x larger than input)
    - data.total % burst_size == 0 (data divisible by burst size)
    """
    source_interface: str
    target_interface: str
    relation: RelationType
    source_dim: Optional[int] = None  # None means total size
    target_dim: Optional[int] = None  # None means total size
    factor: Optional[Union[int, float]] = None  # For MULTIPLE relations
    dependency_type: Optional[str] = None  # For DEPENDENT: "copy", "scaled", "min"
    description: str = ""
    
    def __post_init__(self):
        """Validate relationship configuration"""
        if self.relation == RelationType.MULTIPLE and self.factor is None:
            raise ValueError("MULTIPLE relationships require a factor")
        
        if self.source_interface == self.target_interface:
            if self.source_dim == self.target_dim:
                raise ValueError("Cannot relate interface dimension to itself")
    
    def describe(self) -> str:
        """Human-readable description of the relationship"""
        if self.description:
            return self.description
            
        src = self.source_interface
        if self.source_dim is not None:
            src += f"[{self.source_dim}]"
        else:
            src += ".total"
            
        tgt = self.target_interface
        if self.target_dim is not None:
            tgt += f"[{self.target_dim}]"
        else:
            tgt += ".total"
        
        if self.relation == RelationType.EQUAL:
            return f"{src} == {tgt}"
        elif self.relation == RelationType.MULTIPLE:
            return f"{src} == {self.factor} * {tgt}"
        elif self.relation == RelationType.DEPENDENT:
            if self.dependency_type == "scaled" and self.factor:
                return f"{tgt} = {src} * {self.factor}"
            else:
                return f"{tgt} depends on {src}"
        else:
            return f"{src} ? {tgt}"
    
    def evaluate(self, interfaces: Dict[str, Any]) -> bool:
        """Evaluate the relationship given interface objects
        
        Args:
            interfaces: Dict mapping interface names to Interface objects
            
        Returns:
            True if relationship is satisfied
        """
        if self.source_interface not in interfaces:
            raise ValueError(f"Source interface '{self.source_interface}' not found")
        if self.target_interface not in interfaces:
            raise ValueError(f"Target interface '{self.target_interface}' not found")
        
        src_intf = interfaces[self.source_interface]
        tgt_intf = interfaces[self.target_interface]
        
        # Get dimension values
        if self.source_dim is None:
            from .types import prod
            src_val = prod(src_intf.tensor_dims)
        else:
            if self.source_dim >= len(src_intf.tensor_dims):
                raise ValueError(f"Source dimension {self.source_dim} out of range")
            src_val = src_intf.tensor_dims[self.source_dim]
        
        if self.target_dim is None:
            from .types import prod
            tgt_val = prod(tgt_intf.tensor_dims)
        else:
            if self.target_dim >= len(tgt_intf.tensor_dims):
                raise ValueError(f"Target dimension {self.target_dim} out of range")
            tgt_val = tgt_intf.tensor_dims[self.target_dim]
        
        # Evaluate relationship
        if self.relation == RelationType.EQUAL:
            return src_val == tgt_val
        elif self.relation == RelationType.MULTIPLE:
            return src_val == self.factor * tgt_val
        elif self.relation == RelationType.DEPENDENT:
            # For DEPENDENT, we validate during SDIM propagation
            # Here we just return True as it's a valid relationship type
            return True
        else:
            raise ValueError(f"Unknown relation type: {self.relation}")


@dataclass(frozen=True)
class ParameterDependency:
    """Dependency between kernel parameters
    
    Represents computed parameters like:
    - buffer_size = max(input[0] * 64, 4096)
    - total_ops = matrix[0] * matrix[1] * vector[0]
    - cycles_per_block = (block_size + pipeline_depth - 1) / throughput
    """
    dependent: str
    expression: str
    description: str = ""
    
    def describe(self) -> str:
        """Human-readable description"""
        if self.description:
            return f"{self.dependent}: {self.description}"
        else:
            return f"{self.dependent} = {self.expression}"
    
    def evaluate(self, context: Dict[str, Any]) -> Union[int, float]:
        """Compute parameter value from expression
        
        Args:
            context: Dictionary with interfaces and parameters
            
        Returns:
            Computed parameter value
        """
        # This will be implemented with the expression evaluator
        # For now, just return 0 as placeholder
        return 0


@dataclass
class ConstraintViolation:
    """Detailed information about a constraint violation

    Unified violation class supporting both detailed (expected/actual) and
    simple (message-only) reporting styles.
    """
    constraint_type: str  # "relationship", "constraint", "dependency", etc.

    # Message-based reporting (simple style)
    message: Optional[str] = None

    # Detailed reporting (structured style)
    constraint_name: Optional[str] = None
    description: Optional[str] = None
    expected: Optional[Any] = None
    actual: Optional[Any] = None
    suggestion: str = ""

    # Severity level
    severity: str = "error"  # "error", "warning", "info"

    # Additional context
    details: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Ensure at least one reporting style is used"""
        if not self.message and not self.constraint_name and not self.description:
            raise ValueError("ConstraintViolation must have either message or constraint_name/description")

    def __str__(self) -> str:
        """Formatted error message"""
        # Use detailed format if available
        if self.constraint_name or (self.expected is not None and self.actual is not None):
            msg = f"{self.constraint_type.title()} violation"
            if self.constraint_name:
                msg += f": {self.constraint_name}"
            msg += "\n"

            if self.description:
                msg += f"  Description: {self.description}\n"
            elif self.message:
                msg += f"  Description: {self.message}\n"

            if self.expected is not None:
                msg += f"  Expected: {self.expected}\n"
            if self.actual is not None:
                msg += f"  Actual: {self.actual}"

            if self.suggestion:
                msg += f"\n  Suggestion: {self.suggestion}"

            if self.details:
                msg += f"\n  Details: {self.details}"
        else:
            # Use simple format
            msg = f"[{self.severity.upper()}] {self.constraint_type}: {self.message}"
            if self.details:
                msg += f"\n  Details: {self.details}"

        return msg


class ValidationResult:
    """Result of constraint validation with detailed error reporting

    Supports both constructor patterns:
    - ValidationResult() - empty result, add violations later
    - ValidationResult(violations=[...]) - initialize with violations
    """

    def __init__(self, violations: Optional[List[ConstraintViolation]] = None):
        """Initialize validation result

        Args:
            violations: Optional list of violations to initialize with
        """
        self.violations: List[ConstraintViolation] = violations if violations is not None else []
        self._is_valid = len(self.violations) == 0

    def add_violation(self, violation: ConstraintViolation):
        """Add a constraint violation"""
        self.violations.append(violation)
        self._is_valid = False

    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self._is_valid

    @property
    def has_errors(self) -> bool:
        """Check if result contains error-level violations"""
        return any(v.severity == "error" for v in self.violations)

    @property
    def has_warnings(self) -> bool:
        """Check if result contains warning-level violations"""
        return any(v.severity == "warning" for v in self.violations)

    def get_detailed_report(self) -> str:
        """Get detailed validation report"""
        if self.is_valid:
            return "All constraints satisfied"

        report = f"Validation failed with {len(self.violations)} violations:\n\n"
        for i, violation in enumerate(self.violations, 1):
            report += f"{i}. {violation}\n\n"

        return report

    def raise_if_invalid(self, kernel_name: str = ""):
        """Raise exception if validation failed"""
        if not self.is_valid:
            kernel_info = f" in kernel '{kernel_name}'" if kernel_name else ""
            raise ValueError(f"Constraint validation failed{kernel_info}:\n\n{self.get_detailed_report()}")