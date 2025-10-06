############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Native relationship and constraint types for kernel modeling"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Any, Union, Optional, List, Callable, TYPE_CHECKING
from abc import ABC, abstractmethod

if TYPE_CHECKING:
    from .dimension_constraints import DimensionConstraint


class RelationType(Enum):
    """Types of relationships between interface dimensions

    Simplified to only include actively used relationship types:
    - EQUAL: Dimensions must be exactly equal
    - DIVISIBLE: Target must be divisible by source
    - SCALED: Target equals source * factor
    - DEPENDENT: Target dimension depends on source (with optional scaling) [DEPRECATED: use SCALED]
    - MULTIPLE: Target is a multiple of source [DEPRECATED: use SCALED]
    """
    EQUAL = "equal"
    DIVISIBLE = "divisible"
    SCALED = "scaled"
    MULTIPLE = "multiple"  # Deprecated: use SCALED
    DEPENDENT = "dependent"  # Deprecated: use SCALED or EQUAL


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
        elif self.relation == RelationType.DIVISIBLE:
            return tgt_val % src_val == 0
        elif self.relation == RelationType.SCALED:
            return tgt_val == src_val * self.factor
        elif self.relation == RelationType.MULTIPLE:
            # Deprecated: MULTIPLE means source == factor * target
            return src_val == self.factor * tgt_val
        elif self.relation == RelationType.DEPENDENT:
            # For DEPENDENT, we validate during SDIM propagation
            # Here we just return True as it's a valid relationship type
            return True
        else:
            raise ValueError(f"Unknown relation type: {self.relation}")

    def get_constraints(self) -> List['DimensionConstraint']:
        """Generate atomic constraints from this relationship.

        Relationships are convenience mechanisms that generate one or more
        dimension constraints. This method converts the relationship into
        its constituent constraints.

        Returns:
            List of DimensionConstraints that enforce this relationship
        """
        from .dimension_constraints import (
            EqualityConstraint,
            DivisibleByDimensionConstraint,
            ScaledEqualityConstraint
        )

        if self.relation == RelationType.EQUAL:
            # source[i] == target[j]
            return [
                EqualityConstraint(
                    source_interface=self.source_interface,
                    source_dim=self.source_dim,
                    target_interface=self.target_interface,
                    target_dim=self.target_dim
                )
            ]

        elif self.relation == RelationType.DIVISIBLE:
            # target[j] % source[i] == 0
            return [
                DivisibleByDimensionConstraint(
                    interface_name=self.target_interface,
                    dim_index=self.target_dim,
                    divisor_interface=self.source_interface,
                    divisor_dim=self.source_dim
                )
            ]

        elif self.relation == RelationType.SCALED:
            # target[j] == source[i] * factor
            return [
                ScaledEqualityConstraint(
                    target_interface=self.target_interface,
                    target_dim=self.target_dim,
                    source_interface=self.source_interface,
                    source_dim=self.source_dim,
                    scale_factor=self.factor
                )
            ]

        elif self.relation == RelationType.MULTIPLE:
            # Deprecated: source == factor * target => target == source / factor
            # Convert to SCALED constraint
            if self.factor == 0:
                raise ValueError("MULTIPLE relationship factor cannot be zero")
            return [
                ScaledEqualityConstraint(
                    target_interface=self.target_interface,
                    target_dim=self.target_dim,
                    source_interface=self.source_interface,
                    source_dim=self.source_dim,
                    scale_factor=1.0 / self.factor
                )
            ]

        elif self.relation == RelationType.DEPENDENT:
            # Deprecated: map to appropriate modern constraint
            if self.dependency_type == "copy":
                return [
                    EqualityConstraint(
                        source_interface=self.source_interface,
                        source_dim=self.source_dim,
                        target_interface=self.target_interface,
                        target_dim=self.target_dim
                    )
                ]
            elif self.dependency_type == "scaled":
                return [
                    ScaledEqualityConstraint(
                        target_interface=self.target_interface,
                        target_dim=self.target_dim,
                        source_interface=self.source_interface,
                        source_dim=self.source_dim,
                        scale_factor=self.factor or 1.0
                    )
                ]
            else:
                # For "min" and other types, just use equality for now
                return [
                    EqualityConstraint(
                        source_interface=self.source_interface,
                        source_dim=self.source_dim,
                        target_interface=self.target_interface,
                        target_dim=self.target_dim
                    )
                ]

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


# ===========================================================================
# Simple Relationship Builder Functions
# ===========================================================================

def equal_shapes(source: str, target: str, description: str = "") -> DimensionRelationship:
    """Create relationship: all dimensions must match (total size equality).

    Args:
        source: Source interface name
        target: Target interface name
        description: Optional human-readable description

    Returns:
        DimensionRelationship enforcing source.total == target.total

    Example:
        schema.relationships.append(equal_shapes("input", "output"))
    """
    return DimensionRelationship(
        source_interface=source,
        target_interface=target,
        relation=RelationType.EQUAL,
        source_dim=None,
        target_dim=None,
        description=description or f"{source} total equals {target} total"
    )


def equal_dimension(source: str, target: str,
                   source_dim: int, target_dim: int,
                   description: str = "") -> DimensionRelationship:
    """Create relationship: specific dimensions must match.

    Args:
        source: Source interface name
        target: Target interface name
        source_dim: Source dimension index
        target_dim: Target dimension index
        description: Optional human-readable description

    Returns:
        DimensionRelationship enforcing source[source_dim] == target[target_dim]

    Example:
        # Matrix-vector multiplication: matrix columns == vector length
        schema.relationships.append(
            equal_dimension("matrix", "vector", 1, 0)
        )
    """
    return DimensionRelationship(
        source_interface=source,
        target_interface=target,
        relation=RelationType.EQUAL,
        source_dim=source_dim,
        target_dim=target_dim,
        description=description or f"{source}[{source_dim}] equals {target}[{target_dim}]"
    )


def divisible_dimension(target: str, source: str,
                       target_dim: Optional[int] = None,
                       source_dim: Optional[int] = None,
                       description: str = "") -> DimensionRelationship:
    """Create relationship: target dimension divisible by source dimension.

    Args:
        target: Target interface name (must be divisible)
        source: Source interface name (divisor)
        target_dim: Target dimension index (None = total)
        source_dim: Source dimension index (None = total)
        description: Optional human-readable description

    Returns:
        DimensionRelationship enforcing target[target_dim] % source[source_dim] == 0

    Example:
        # Block size must divide tensor size
        schema.relationships.append(
            divisible_dimension("tensor", "block", 0, 0)
        )
    """
    return DimensionRelationship(
        source_interface=source,
        target_interface=target,
        relation=RelationType.DIVISIBLE,
        source_dim=source_dim,
        target_dim=target_dim,
        description=description or f"{target}[{target_dim}] divisible by {source}[{source_dim}]"
    )


def scaled_dimension(source: str, target: str,
                    scale_factor: Union[int, float, str],
                    source_dim: Optional[int] = None,
                    target_dim: Optional[int] = None,
                    description: str = "") -> DimensionRelationship:
    """Create relationship: target equals source * scale_factor.

    Args:
        source: Source interface name
        target: Target interface name
        scale_factor: Scaling factor (literal or nodeattr name)
        source_dim: Source dimension index (None = total)
        target_dim: Target dimension index (None = total)
        description: Optional human-readable description

    Returns:
        DimensionRelationship enforcing target[target_dim] == source[source_dim] * scale_factor

    Example:
        # Output is 4x input size
        schema.relationships.append(
            scaled_dimension("input", "output", 4)
        )

        # Output scaled by parameter
        schema.relationships.append(
            scaled_dimension("input", "output", "UPSAMPLE_FACTOR", 0, 0)
        )
    """
    return DimensionRelationship(
        source_interface=source,
        target_interface=target,
        relation=RelationType.SCALED,
        source_dim=source_dim,
        target_dim=target_dim,
        factor=scale_factor,
        description=description or f"{target}[{target_dim}] equals {source}[{source_dim}] * {scale_factor}"
    )