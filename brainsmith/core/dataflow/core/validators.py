############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Validation framework for kernel constraints and relationships"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Set
from .relationships import (
    ValidationResult, ConstraintViolation, DimensionRelationship, 
    ArchitecturalConstraint, ParameterDependency
)
from .expressions import evaluate_expression, validate_expression


class ConstraintValidator(ABC):
    """Base class for constraint validators"""
    
    @abstractmethod
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate constraints in given context
        
        Args:
            context: Dictionary containing interfaces, parameters, etc.
            
        Returns:
            ValidationResult with any violations
        """
        pass


class RelationshipValidator(ConstraintValidator):
    """Validator for dimension relationships between interfaces"""
    
    def __init__(self, relationships: List[DimensionRelationship]):
        self.relationships = relationships
    
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate all dimension relationships"""
        result = ValidationResult()
        interfaces = context.get('interfaces', {})
        
        for rel in self.relationships:
            try:
                if not rel.evaluate(interfaces):
                    # Get actual values for detailed error
                    src_intf = interfaces.get(rel.source_interface)
                    tgt_intf = interfaces.get(rel.target_interface)
                    
                    if src_intf and tgt_intf:
                        # Extract actual values
                        if rel.source_dim is None:
                            from .types import prod
                            src_val = prod(src_intf.tensor_dims)
                        else:
                            src_val = src_intf.tensor_dims[rel.source_dim]
                        
                        if rel.target_dim is None:
                            from .types import prod
                            tgt_val = prod(tgt_intf.tensor_dims)
                        else:
                            tgt_val = tgt_intf.tensor_dims[rel.target_dim]
                        
                        # Create violation
                        violation = ConstraintViolation(
                            constraint_type="relationship",
                            constraint_name=rel.describe(),
                            description=f"Dimension relationship not satisfied",
                            expected=rel.describe(),
                            actual=f"Source={src_val}, Target={tgt_val}",
                            suggestion=self._get_relationship_suggestion(rel, src_val, tgt_val)
                        )
                        result.add_violation(violation)
                    
            except Exception as e:
                violation = ConstraintViolation(
                    constraint_type="relationship",
                    constraint_name=rel.describe(),
                    description=f"Error evaluating relationship: {str(e)}",
                    expected="Valid relationship",
                    actual="Evaluation error"
                )
                result.add_violation(violation)
        
        return result
    
    def _get_relationship_suggestion(self, rel: DimensionRelationship, 
                                   src_val: int, tgt_val: int) -> str:
        """Generate helpful suggestion for fixing relationship violation"""
        from .relationships import RelationType
        
        if rel.relation == RelationType.EQUAL:
            if src_val > tgt_val:
                return f"Increase {rel.target_interface} dimension to {src_val}"
            else:
                return f"Increase {rel.source_interface} dimension to {tgt_val}"
        
        elif rel.relation == RelationType.MULTIPLE:
            expected = rel.factor * tgt_val
            return f"Set {rel.source_interface} to {expected} (= {rel.factor} * {tgt_val})"
        
        elif rel.relation == RelationType.DIVISIBLE:
            remainder = src_val % tgt_val if tgt_val != 0 else src_val
            aligned = src_val + (tgt_val - remainder) if remainder != 0 else src_val
            return f"Align {rel.source_interface} to multiple of {tgt_val} (try {aligned})"
        
        return "Adjust interface dimensions to satisfy relationship"


class ArchitecturalConstraintValidator(ConstraintValidator):
    """Validator for architectural constraints"""
    
    def __init__(self, constraints: List[ArchitecturalConstraint]):
        self.constraints = constraints
    
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate all architectural constraints"""
        result = ValidationResult()
        
        for constraint in self.constraints:
            try:
                # First validate the expression
                expr_errors = validate_expression(constraint.expression, context)
                if expr_errors:
                    violation = ConstraintViolation(
                        constraint_type="constraint",
                        constraint_name=constraint.name,
                        description=f"Invalid expression: {'; '.join(expr_errors)}",
                        expected="Valid expression",
                        actual=constraint.expression
                    )
                    result.add_violation(violation)
                    continue
                
                # Evaluate expression
                expr_value = evaluate_expression(constraint.expression, context)
                
                # Get constraint value
                if isinstance(constraint.value, str):
                    # Value is also an expression
                    constraint_value = evaluate_expression(constraint.value, context)
                else:
                    constraint_value = constraint.value
                
                # Check constraint
                satisfied = self._check_constraint(expr_value, constraint.operator, constraint_value)
                
                if not satisfied:
                    violation = ConstraintViolation(
                        constraint_type="constraint",
                        constraint_name=constraint.name,
                        description=constraint.describe(),
                        expected=f"{constraint.expression} {constraint.operator} {constraint_value}",
                        actual=f"{constraint.expression} = {expr_value}",
                        suggestion=self._get_constraint_suggestion(
                            constraint, expr_value, constraint_value
                        )
                    )
                    result.add_violation(violation)
                    
            except Exception as e:
                violation = ConstraintViolation(
                    constraint_type="constraint",
                    constraint_name=constraint.name,
                    description=f"Error evaluating constraint: {str(e)}",
                    expected="Valid constraint evaluation",
                    actual="Evaluation error"
                )
                result.add_violation(violation)
        
        return result
    
    def _check_constraint(self, expr_value: float, operator: str, 
                         constraint_value: float) -> bool:
        """Check if constraint is satisfied"""
        if operator == "==":
            return abs(expr_value - constraint_value) < 1e-9  # Handle floating point
        elif operator == "!=":
            return abs(expr_value - constraint_value) >= 1e-9
        elif operator == "<=":
            return expr_value <= constraint_value
        elif operator == ">=":
            return expr_value >= constraint_value
        elif operator == "<":
            return expr_value < constraint_value
        elif operator == ">":
            return expr_value > constraint_value
        elif operator == "%":
            return constraint_value != 0 and expr_value % constraint_value == 0
        else:
            raise ValueError(f"Unknown operator: {operator}")
    
    def _get_constraint_suggestion(self, constraint: ArchitecturalConstraint,
                                 actual: float, expected: float) -> str:
        """Generate helpful suggestion for fixing constraint violation"""
        if constraint.operator in ["<=", "<"]:
            return f"Reduce {constraint.expression} (current: {actual}, limit: {expected})"
        elif constraint.operator in [">=", ">"]:
            return f"Increase {constraint.expression} (current: {actual}, minimum: {expected})"
        elif constraint.operator == "==":
            return f"Set {constraint.expression} to exactly {expected}"
        elif constraint.operator == "%":
            remainder = actual % expected if expected != 0 else actual
            aligned = actual + (expected - remainder) if remainder != 0 else actual
            return f"Align {constraint.expression} to multiple of {expected} (try {aligned})"
        else:
            return f"Adjust {constraint.expression} to satisfy constraint"


class DependencyValidator(ConstraintValidator):
    """Validator for parameter dependencies"""
    
    def __init__(self, dependencies: List[ParameterDependency]):
        self.dependencies = dependencies
    
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate parameter dependencies for cycles and computability"""
        result = ValidationResult()
        
        # Build dependency graph
        dep_graph = self._build_dependency_graph()
        
        # Check for cycles
        cycles = self._find_cycles(dep_graph)
        if cycles:
            for cycle in cycles:
                violation = ConstraintViolation(
                    constraint_type="dependency",
                    constraint_name="circular_dependency",
                    description=f"Circular dependency detected: {' -> '.join(cycle)}",
                    expected="Acyclic dependency graph",
                    actual=f"Cycle: {' -> '.join(cycle)}",
                    suggestion="Remove one dependency to break the cycle"
                )
                result.add_violation(violation)
        
        # Check if all dependencies can be computed
        for dep in self.dependencies:
            try:
                # Validate expression
                expr_errors = validate_expression(dep.expression, context)
                if expr_errors:
                    violation = ConstraintViolation(
                        constraint_type="dependency",
                        constraint_name=dep.dependent,
                        description=f"Invalid dependency expression: {'; '.join(expr_errors)}",
                        expected="Valid expression",
                        actual=dep.expression
                    )
                    result.add_violation(violation)
                
            except Exception as e:
                violation = ConstraintViolation(
                    constraint_type="dependency",
                    constraint_name=dep.dependent,
                    description=f"Error validating dependency: {str(e)}",
                    expected="Valid dependency",
                    actual="Validation error"
                )
                result.add_violation(violation)
        
        return result
    
    def _build_dependency_graph(self) -> Dict[str, Set[str]]:
        """Build dependency graph from parameter dependencies"""
        from .expressions import extract_dependencies
        
        graph = {}
        for dep in self.dependencies:
            dependencies = extract_dependencies(dep.expression)
            graph[dep.dependent] = set(dependencies)
        
        return graph
    
    def _find_cycles(self, graph: Dict[str, Set[str]]) -> List[List[str]]:
        """Find cycles in dependency graph using DFS"""
        visited = set()
        rec_stack = set()
        cycles = []
        
        def dfs(node: str, path: List[str]) -> bool:
            if node in rec_stack:
                # Found cycle
                cycle_start = path.index(node)
                cycles.append(path[cycle_start:] + [node])
                return True
            
            if node in visited:
                return False
            
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in graph.get(node, set()):
                if neighbor in graph:  # Only follow dependencies that are also dependents
                    dfs(neighbor, path + [node])
            
            rec_stack.remove(node)
            return False
        
        for node in graph:
            if node not in visited:
                dfs(node, [])
        
        return cycles


class KernelValidator:
    """Comprehensive validator for entire kernel"""
    
    def __init__(self, relationships: List[DimensionRelationship],
                 constraints: List[ArchitecturalConstraint],
                 dependencies: List[ParameterDependency]):
        self.relationship_validator = RelationshipValidator(relationships)
        self.constraint_validator = ArchitecturalConstraintValidator(constraints)
        self.dependency_validator = DependencyValidator(dependencies)
    
    def validate(self, context: Dict[str, Any]) -> ValidationResult:
        """Validate all aspects of kernel"""
        result = ValidationResult()
        
        # Validate dependencies first (needed for other validations)
        dep_result = self.dependency_validator.validate(context)
        result.violations.extend(dep_result.violations)
        
        # Validate relationships
        rel_result = self.relationship_validator.validate(context)
        result.violations.extend(rel_result.violations)
        
        # Validate constraints
        const_result = self.constraint_validator.validate(context)
        result.violations.extend(const_result.violations)
        
        # Update valid flag
        result._is_valid = len(result.violations) == 0
        
        return result