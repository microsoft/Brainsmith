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
    """Detailed information about a constraint violation"""
    constraint_type: str  # "relationship", "constraint", "dependency"
    constraint_name: str
    description: str
    expected: Any
    actual: Any
    suggestion: str = ""
    
    def __str__(self) -> str:
        """Formatted error message"""
        msg = f"{self.constraint_type.title()} violation: {self.constraint_name}\n"
        msg += f"  Description: {self.description}\n"
        msg += f"  Expected: {self.expected}\n"
        msg += f"  Actual: {self.actual}"
        if self.suggestion:
            msg += f"\n  Suggestion: {self.suggestion}"
        return msg


class ValidationResult:
    """Result of constraint validation with detailed error reporting"""
    
    def __init__(self):
        self.violations: List[ConstraintViolation] = []
        self._is_valid = True
    
    def add_violation(self, violation: ConstraintViolation):
        """Add a constraint violation"""
        self.violations.append(violation)
        self._is_valid = False
    
    @property
    def is_valid(self) -> bool:
        """Check if validation passed"""
        return self._is_valid
    
    def get_detailed_report(self) -> str:
        """Get detailed validation report"""
        if self.is_valid:
            return "All constraints satisfied"
        
        report = f"Validation failed with {len(self.violations)} violations:\n\n"
        for i, violation in enumerate(self.violations, 1):
            report += f"{i}. {violation}\n\n"
        
        return report
    
    def get_error_summary(self) -> str:
        """Get concise error summary"""
        errors = [v for v in self.violations if v.severity == "error"]
        if not errors:
            return "No errors"
        return "; ".join(v.message for v in errors)
    
    def raise_if_invalid(self, kernel_name: str = ""):
        """Raise exception if validation failed"""
        if not self.is_valid:
            kernel_info = f" in kernel '{kernel_name}'" if kernel_name else ""
            raise ValueError(f"Constraint validation failed{kernel_info}:\n\n{self.get_detailed_report()}")