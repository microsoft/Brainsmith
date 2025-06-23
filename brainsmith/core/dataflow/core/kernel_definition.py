############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Kernel definition for relationship specification and validation"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Union, Any
from .base import BaseDefinition, ValidationContext, ParameterBinding
from .interface_definition import InterfaceDefinition
from .relationships import (
    DimensionRelationship, ArchitecturalConstraint, ParameterDependency,
    ValidationResult, RelationType
)
from .validators import KernelValidator


@dataclass
class KernelDefinition(BaseDefinition):
    """Definition of a kernel with relationships, constraints, and validation rules
    
    Specifies the interfaces, relationships, and architectural requirements
    for a kernel type. Does not contain actual runtime values.
    """
    
    # Core specification
    name: str
    hw_module: Optional[str] = None  # SystemVerilog module name
    
    # Interface definitions
    interface_definitions: List[InterfaceDefinition] = field(default_factory=list)
    
    # Relationship specifications
    relationships: List[DimensionRelationship] = field(default_factory=list)
    constraints: List[ArchitecturalConstraint] = field(default_factory=list)
    dependencies: List[ParameterDependency] = field(default_factory=list)
    
    # Architectural requirements
    requires_burst_alignment: bool = False
    requires_power_of_two: Set[str] = field(default_factory=set)  # Interface names
    memory_architecture: Optional[str] = None  # "distributed", "HBM", "DDR", etc.
    pipeline_style: Optional[str] = None  # "streaming", "batch", "hybrid"
    
    # Timing specifications (for validation, not runtime calculation)
    min_latency_cycles: Optional[int] = None
    max_latency_cycles: Optional[int] = None
    min_ii: Optional[int] = None
    max_ii: Optional[int] = None
    
    # Resource constraints
    resource_bounds: Dict[str, Union[int, float]] = field(default_factory=dict)
    
    def __post_init__(self):
        """Initialize and validate kernel definition"""
        # Set hw_module to name if not specified
        if self.hw_module is None:
            self.hw_module = self.name
        
        # Build interface name mapping for quick lookup
        self._interface_map = {intf.name: intf for intf in self.interface_definitions}
    
    def validate(self) -> List[str]:
        """Validate the kernel definition for internal consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Validate basic properties
        if not isinstance(self.name, str) or not self.name:
            errors.append("Kernel name must be non-empty string")
        
        # Validate interface definitions
        interface_names = set()
        for intf_def in self.interface_definitions:
            # Check for duplicate names
            if intf_def.name in interface_names:
                errors.append(f"Duplicate interface name: {intf_def.name}")
            interface_names.add(intf_def.name)
            
            # Validate individual interface
            intf_errors = intf_def.validate()
            errors.extend([f"Interface {intf_def.name}: {err}" for err in intf_errors])
        
        # Validate relationships reference existing interfaces
        for rel in self.relationships:
            if rel.source_interface not in interface_names:
                errors.append(f"Relationship references unknown source interface: {rel.source_interface}")
            if rel.target_interface not in interface_names:
                errors.append(f"Relationship references unknown target interface: {rel.target_interface}")
        
        # Validate constraints reference valid interface expressions
        for constraint in self.constraints:
            # Extract interface references from expression
            from .expressions import extract_dependencies
            deps = extract_dependencies(constraint.expression)
            for dep in deps:
                if dep not in interface_names and not dep.isupper():  # Skip constants like BURST_SIZE
                    errors.append(f"Constraint {constraint.name} references unknown interface: {dep}")
        
        # Validate dependencies
        for dep in self.dependencies:
            from .expressions import extract_dependencies
            deps = extract_dependencies(dep.expression)
            for d in deps:
                if d not in interface_names and not d.isupper():
                    errors.append(f"Dependency {dep.dependent} references unknown interface: {d}")
        
        # Validate power-of-two requirements reference existing interfaces
        for intf_name in self.requires_power_of_two:
            if intf_name not in interface_names:
                errors.append(f"Power-of-two requirement references unknown interface: {intf_name}")
        
        # Validate timing constraints
        if (self.min_latency_cycles is not None and self.max_latency_cycles is not None and 
            self.min_latency_cycles > self.max_latency_cycles):
            errors.append(f"min_latency_cycles > max_latency_cycles")
        
        if (self.min_ii is not None and self.max_ii is not None and 
            self.min_ii > self.max_ii):
            errors.append(f"min_ii > max_ii")
        
        return errors
    
    def validate_model_configuration(self, interface_models: List['InterfaceModel'],
                                   parameter_binding: ParameterBinding) -> ValidationResult:
        """Validate that model configuration satisfies this definition
        
        Args:
            interface_models: List of interface models
            parameter_binding: Parameter values for this instance
            
        Returns:
            ValidationResult with any violations
        """
        # Create validator
        validator = KernelValidator(self.relationships, self.constraints, self.dependencies)
        
        # Build validation context
        context = ValidationContext()
        
        # Add interface models to context
        for model in interface_models:
            if model.definition:
                context.add_interface(model.definition.name, model)
        
        # Add parameters and constants
        for name, value in parameter_binding.parameters.items():
            context.add_parameter(name, value)
        for name, value in parameter_binding.constants.items():
            context.add_constant(name, value)
        
        # Validate using expression context
        return validator.validate(context.to_expression_context())
    
    def get_interface_definition(self, name: str) -> Optional[InterfaceDefinition]:
        """Get interface definition by name"""
        return self._interface_map.get(name)
    
    def add_interface_definition(self, interface_def: InterfaceDefinition):
        """Add an interface definition to this kernel"""
        if interface_def.name in self._interface_map:
            raise ValueError(f"Interface {interface_def.name} already exists")
        
        self.interface_definitions.append(interface_def)
        self._interface_map[interface_def.name] = interface_def
    
    def add_relationship(self, source: str, target: str,
                        relation: RelationType = RelationType.EQUAL,
                        source_dim: Optional[int] = None,
                        target_dim: Optional[int] = None,
                        factor: Optional[Union[int, float]] = None,
                        description: str = "") -> None:
        """Add a dimension relationship between interfaces
        
        Args:
            source: Source interface name
            target: Target interface name
            relation: Type of relationship
            source_dim: Source dimension index (None for total size)
            target_dim: Target dimension index (None for total size)
            factor: Factor for MULTIPLE relationships
            description: Human-readable description
        """
        rel = DimensionRelationship(
            source_interface=source,
            target_interface=target,
            relation=relation,
            source_dim=source_dim,
            target_dim=target_dim,
            factor=factor,
            description=description
        )
        self.relationships.append(rel)
        
        # Update interface dataflow metadata
        src_def = self.get_interface_definition(source)
        tgt_def = self.get_interface_definition(target)
        
        if src_def and tgt_def:
            # Add dataflow connections for certain relationship types
            if relation in [RelationType.EQUAL, RelationType.MULTIPLE]:
                src_def.add_produces(target)
                tgt_def.add_consumes(source)
    
    def add_constraint(self, name: str, expression: str,
                      operator: str, value: Union[int, float, str],
                      description: str = "") -> None:
        """Add an architectural constraint
        
        Args:
            name: Constraint name
            expression: Expression to evaluate
            operator: Comparison operator (==, <=, >=, etc.)
            value: Target value or expression
            description: Human-readable description
        """
        constraint = ArchitecturalConstraint(
            name=name,
            expression=expression,
            operator=operator,
            value=value,
            description=description
        )
        self.constraints.append(constraint)
    
    def add_dependency(self, dependent: str, expression: str,
                      description: str = "") -> None:
        """Add a parameter dependency
        
        Args:
            dependent: Name of dependent parameter
            expression: Expression to compute parameter value
            description: Human-readable description
        """
        dependency = ParameterDependency(
            dependent=dependent,
            expression=expression,
            description=description
        )
        self.dependencies.append(dependency)
    
    def get_relationships_for_interface(self, interface_name: str) -> List[DimensionRelationship]:
        """Get all relationships involving a specific interface"""
        return [rel for rel in self.relationships 
                if rel.source_interface == interface_name or rel.target_interface == interface_name]
    
    def get_dependent_interfaces(self, interface_name: str) -> Set[str]:
        """Get all interfaces that depend on the given interface"""
        dependents = set()
        for rel in self.relationships:
            if rel.source_interface == interface_name:
                dependents.add(rel.target_interface)
        return dependents
    
    def get_constraint_graph(self) -> Dict[str, Set[str]]:
        """Get constraint dependency graph
        
        Returns:
            Dict mapping interface names to their dependencies
        """
        graph = {}
        for rel in self.relationships:
            if rel.target_interface not in graph:
                graph[rel.target_interface] = set()
            graph[rel.target_interface].add(rel.source_interface)
        return graph
    
    def create_model(self, interface_models: List['InterfaceModel'],
                    parameter_binding: ParameterBinding,
                    **kwargs) -> 'KernelModel':
        """Create a kernel model from this definition
        
        Args:
            interface_models: List of interface models
            parameter_binding: Parameter values
            **kwargs: Additional model parameters
            
        Returns:
            KernelModel instance
            
        Raises:
            ValueError: If configuration violates definition constraints
        """
        # Validate model configuration against definition
        result = self.validate_model_configuration(interface_models, parameter_binding)
        if not result.is_valid:
            raise ValueError(f"Model configuration violates definition:\n{result.get_detailed_report()}")
        
        # Import here to avoid circular dependency
        from .kernel_model import KernelModel
        
        return KernelModel(
            definition=self,
            interface_models=interface_models,
            parameter_binding=parameter_binding,
            **kwargs
        )
    
    def get_definition_summary(self) -> Dict[str, Any]:
        """Get summary of kernel definition"""
        return {
            "name": self.name,
            "hw_module": self.hw_module,
            "n_interfaces": len(self.interface_definitions),
            "interface_names": [intf.name for intf in self.interface_definitions],
            "n_relationships": len(self.relationships),
            "n_constraints": len(self.constraints),
            "n_dependencies": len(self.dependencies),
            "requires_burst_alignment": self.requires_burst_alignment,
            "requires_power_of_two": list(self.requires_power_of_two),
            "memory_architecture": self.memory_architecture,
            "pipeline_style": self.pipeline_style,
            "resource_bounds": self.resource_bounds
        }
    
    def __repr__(self) -> str:
        """String representation"""
        n_intf = len(self.interface_definitions)
        n_rel = len(self.relationships)
        n_const = len(self.constraints)
        
        return (
            f"KernelDefinition(name='{self.name}', "
            f"interfaces={n_intf}, "
            f"relationships={n_rel}, "
            f"constraints={n_const})"
        )