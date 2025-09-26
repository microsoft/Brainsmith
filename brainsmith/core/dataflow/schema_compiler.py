############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""Schema compilation for optimized runtime performance.

This module provides pre-processing of KernelSchema objects to enable
efficient runtime operations. By extracting all metadata during initialization,
we can perform O(1) lookups and targeted cache invalidation during execution.

Key features:
- Pre-computed parameter sets for O(1) membership tests
- Dependency mapping for targeted cache invalidation
- Static resolution of non-parameterized templates
- Validation rules compilation
"""

from dataclasses import dataclass, field
from typing import Dict, Set, List, Optional, Union, FrozenSet, Tuple, Any
from functools import cached_property
from enum import Enum

from .schemas import KernelSchema, InputSchema, OutputSchema
from .template_utils import extract_tiling_parameters
from .relationships import RelationType, DimensionRelationship
from .validation import ValidationResult, ConstraintViolation


class ConstraintType(Enum):
    """Types of shape constraints."""
    FIXED = "fixed"  # Dimension has fixed value
    PARAMETER = "parameter"  # Dimension equals parameter
    MULTIPLE = "multiple"  # Dimension is multiple of parameter
    RANGE = "range"  # Dimension in range
    DEPENDENT = "dependent"  # Dimension depends on others


@dataclass
class ShapeConstraint:
    """Represents a constraint on a shape dimension."""
    
    interface: str
    dimension: int
    constraint_type: ConstraintType
    value: Optional[Union[int, str]] = None
    min_value: Optional[int] = None
    max_value: Optional[int] = None
    factor: Optional[int] = None
    expression: Optional[str] = None
    
    def validate(self, dim_value: int, params: Dict[str, int]) -> bool:
        """Validate dimension value against constraint."""
        if self.constraint_type == ConstraintType.FIXED:
            return dim_value == self.value
            
        elif self.constraint_type == ConstraintType.PARAMETER:
            param_value = params.get(self.value)
            return param_value is not None and dim_value == param_value
            
        elif self.constraint_type == ConstraintType.MULTIPLE:
            param_value = params.get(self.value)
            if param_value is None:
                return False
            return dim_value % param_value == 0
            
        elif self.constraint_type == ConstraintType.RANGE:
            if self.min_value is not None and dim_value < self.min_value:
                return False
            if self.max_value is not None and dim_value > self.max_value:
                return False
            return True
            
        elif self.constraint_type == ConstraintType.DEPENDENT:
            # Would need expression evaluation
            return True
            
        return False
    
    def __repr__(self) -> str:
        """String representation."""
        if self.constraint_type == ConstraintType.FIXED:
            return f"{self.interface}[{self.dimension}] = {self.value}"
        elif self.constraint_type == ConstraintType.PARAMETER:
            return f"{self.interface}[{self.dimension}] = {self.value}"
        elif self.constraint_type == ConstraintType.MULTIPLE:
            return f"{self.interface}[{self.dimension}] % {self.value} = 0"
        elif self.constraint_type == ConstraintType.RANGE:
            return f"{self.interface}[{self.dimension}] ∈ [{self.min_value}, {self.max_value}]"
        else:
            return f"{self.interface}[{self.dimension}] = {self.expression}"


@dataclass
class CompiledSchema:
    """Pre-processed schema optimized for runtime performance.
    
    This class extracts all metadata from a KernelSchema during initialization
    to enable efficient runtime operations:
    - O(1) attribute lookups
    - Targeted cache invalidation
    - Pre-computed dependency relationships
    """
    
    # Original schema
    schema: KernelSchema
    
    # All parameter names referenced in the schema
    all_parameters: FrozenSet[str] = field(init=False)
    
    # Mapping from attribute to affected components
    attr_dependencies: Dict[str, Set[str]] = field(init=False)
    
    # Pre-resolved static parts of templates
    static_resolutions: Dict[str, List[Union[int, str]]] = field(init=False)
    
    # Cache invalidation groups
    invalidation_groups: Dict[str, Set[str]] = field(init=False)
    
    # Performance metadata
    template_metadata: Dict[str, Dict[str, Any]] = field(init=False)
    
    # Shape constraint inference
    shape_constraints: Dict[str, 'ShapeConstraint'] = field(init=False)
    parameter_bounds: Dict[str, Tuple[int, int]] = field(init=False)
    
    def __post_init__(self):
        """Extract all metadata from schema."""
        self.all_parameters = self._extract_all_parameters()
        self.attr_dependencies = self._build_dependency_map()
        self.static_resolutions = self._pre_resolve_static()
        self.invalidation_groups = self._build_invalidation_groups()
        self.template_metadata = self._extract_template_metadata()
        self.shape_constraints = self._infer_shape_constraints()
        self.parameter_bounds = self._infer_parameter_bounds()
    
    def _extract_all_parameters(self) -> FrozenSet[str]:
        """Extract all nodeattr names referenced in schema.
        
        Returns:
            Frozen set of all parameter names for O(1) membership tests
        """
        params = set()
        
        # Extract from input tiling
        for inp in self.schema.inputs:
            if inp.block_tiling:
                params.update(extract_tiling_parameters(inp.block_tiling))
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                params.update(extract_tiling_parameters(inp.stream_tiling))
        
        # Extract from output tiling  
        for out in self.schema.outputs:
            if out.block_tiling:
                params.update(extract_tiling_parameters(out.block_tiling))
        
        # Add common kernel parameters
        common_params = [
            "CHANNELS", "PE", "SIMD", "K", "S",
            "BATCH", "GROUPS", "DEPTHWISE", 
            "clock_freq_mhz"
        ]
        params.update(common_params)
        
        # Add datatype attributes
        for i, inp in enumerate(self.schema.inputs):
            if inp.datatype_attr:
                params.add(inp.datatype_attr)
            else:
                params.add(f"input{i}Datatype")
        
        for i, out in enumerate(self.schema.outputs):
            if out.datatype_attr:
                params.add(out.datatype_attr)
            else:
                params.add(f"output{i}Datatype")
        
        return frozenset(params)
    
    def _build_dependency_map(self) -> Dict[str, Set[str]]:
        """Build mapping from attributes to affected components.
        
        Returns:
            Dict mapping attribute names to sets of affected cache components
        """
        deps = {}
        
        # Parameters in tiling affect resolved config and downstream
        tiling_params = set()
        for inp in self.schema.inputs:
            if inp.block_tiling:
                tiling_params.update(extract_tiling_parameters(inp.block_tiling))
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                tiling_params.update(extract_tiling_parameters(inp.stream_tiling))
        
        for out in self.schema.outputs:
            if out.block_tiling:
                tiling_params.update(extract_tiling_parameters(out.block_tiling))
        
        # Tiling parameters affect all caches
        for param in tiling_params:
            deps[param] = {"resolved_config", "tensor_context", "kernel_model"}
        
        # Common parameters affect resolved config and kernel model
        for param in ["CHANNELS", "PE", "SIMD", "K", "S", "BATCH", "GROUPS", "DEPTHWISE"]:
            deps[param] = {"resolved_config", "tensor_context", "kernel_model"}
        
        # Datatype attributes only affect tensor context and kernel model
        for i in range(len(self.schema.inputs)):
            deps[f"input{i}Datatype"] = {"tensor_context", "kernel_model"}
        for i in range(len(self.schema.outputs)):
            deps[f"output{i}Datatype"] = {"tensor_context", "kernel_model"}
        
        # Performance parameters only affect kernel model
        deps["clock_freq_mhz"] = {"kernel_model"}
        
        return deps
    
    def _pre_resolve_static(self) -> Dict[str, List[Union[int, str]]]:
        """Pre-resolve static parts of templates.
        
        Returns:
            Dict of template keys to partially resolved templates
        """
        resolutions = {}
        
        # Pre-resolve input templates
        for inp in self.schema.inputs:
            if inp.block_tiling:
                key = f"input_{inp.name}_block"
                resolutions[key] = self._resolve_static_parts(inp.block_tiling)
            
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                key = f"input_{inp.name}_stream"
                resolutions[key] = self._resolve_static_parts(inp.stream_tiling)
        
        # Pre-resolve output templates
        for out in self.schema.outputs:
            if out.block_tiling:
                key = f"output_{out.name}_block"
                resolutions[key] = self._resolve_static_parts(out.block_tiling)
        
        return resolutions
    
    def _resolve_static_parts(self, template: List[Union[int, str]]) -> List[Union[int, str]]:
        """Resolve static (non-parameterized) parts of a template.
        
        Args:
            template: Template specification
            
        Returns:
            Template with static parts resolved
        """
        resolved = []
        for item in template:
            if isinstance(item, int):
                # Already resolved
                resolved.append(item)
            elif item == ":":
                # Static full dimension
                resolved.append(":")
            elif isinstance(item, str) and not item.isalpha():
                # Numeric string or expression without parameters
                try:
                    resolved.append(int(item))
                except ValueError:
                    resolved.append(item)
            else:
                # Parameter reference - keep as is
                resolved.append(item)
        
        return resolved
    
    def _build_invalidation_groups(self) -> Dict[str, Set[str]]:
        """Build invalidation groups for efficient cache management.
        
        Returns:
            Dict of cache names to sets of attributes that invalidate them
        """
        groups = {
            "resolved_config": set(),
            "tensor_context": set(), 
            "kernel_model": set()
        }
        
        # Invert the dependency map
        for attr, caches in self.attr_dependencies.items():
            for cache in caches:
                if cache in groups:
                    groups[cache].add(attr)
        
        return groups
    
    def _extract_template_metadata(self) -> Dict[str, Dict[str, Any]]:
        """Extract metadata about templates for optimization.
        
        Returns:
            Dict of template keys to metadata dicts
        """
        metadata = {}
        
        # Analyze input templates
        for inp in self.schema.inputs:
            if inp.block_tiling:
                key = f"input_{inp.name}_block"
                metadata[key] = {
                    "params": extract_tiling_parameters(inp.block_tiling),
                    "has_static": any(isinstance(x, int) for x in inp.block_tiling),
                    "length": len(inp.block_tiling)
                }
            
            if hasattr(inp, 'stream_tiling') and inp.stream_tiling:
                key = f"input_{inp.name}_stream"
                metadata[key] = {
                    "params": extract_tiling_parameters(inp.stream_tiling),
                    "has_static": any(isinstance(x, int) for x in inp.stream_tiling),
                    "length": len(inp.stream_tiling)
                }
        
        # Analyze output templates
        for out in self.schema.outputs:
            if out.block_tiling:
                key = f"output_{out.name}_block"
                metadata[key] = {
                    "params": extract_tiling_parameters(out.block_tiling),
                    "has_static": any(isinstance(x, int) for x in out.block_tiling),
                    "length": len(out.block_tiling)
                }
        
        return metadata
    
    def _infer_shape_constraints(self) -> Dict[str, ShapeConstraint]:
        """Infer shape constraints from tiling and relationships.
        
        Returns:
            Dict mapping constraint keys to ShapeConstraint objects
        """
        constraints = {}
        
        # Infer from block tiling templates
        for inp in self.schema.inputs:
            if inp.block_tiling:
                for dim, tile in enumerate(inp.block_tiling):
                    key = f"{inp.name}_block_{dim}"
                    if isinstance(tile, int):
                        # Fixed value constraint
                        constraints[key] = ShapeConstraint(
                            interface=inp.name,
                            dimension=dim,
                            constraint_type=ConstraintType.FIXED,
                            value=tile
                        )
                    elif isinstance(tile, str) and tile != ":":
                        # Parameter constraint
                        constraints[key] = ShapeConstraint(
                            interface=inp.name,
                            dimension=dim,
                            constraint_type=ConstraintType.PARAMETER,
                            value=tile
                        )
        
        # Infer from relationships
        for rel in self.schema.relationships:
            if rel.relation == RelationType.EQUAL:
                # Equal dimensions constraint
                key = f"{rel.source_interface}_{rel.source_dim}_equals_{rel.target_interface}_{rel.target_dim}"
                constraints[key] = ShapeConstraint(
                    interface=rel.source_interface,
                    dimension=rel.source_dim or 0,
                    constraint_type=ConstraintType.DEPENDENT,
                    expression=f"{rel.target_interface}[{rel.target_dim or 0}]"
                )
            elif rel.relation == RelationType.MULTIPLE:
                # Multiple constraint
                key = f"{rel.source_interface}_{rel.source_dim}_multiple_{rel.factor}"
                constraints[key] = ShapeConstraint(
                    interface=rel.source_interface,
                    dimension=rel.source_dim or 0,
                    constraint_type=ConstraintType.MULTIPLE,
                    value=rel.target_interface,
                    factor=rel.factor
                )
        
        return constraints
    
    def _infer_parameter_bounds(self) -> Dict[str, Tuple[int, int]]:
        """Infer bounds for parameters from usage context.
        
        Returns:
            Dict mapping parameter names to (min, max) bounds
        """
        bounds = {}
        
        # Common parameter bounds based on hardware constraints
        common_bounds = {
            "PE": (1, 1024),  # Processing elements
            "SIMD": (1, 128),  # SIMD width
            "K": (1, 11),  # Kernel size
            "S": (1, 4),  # Stride
            "CHANNELS": (1, 2048),  # Channel count
            "BATCH": (1, 256),  # Batch size
            "GROUPS": (1, 64),  # Group count
        }
        
        # Start with common bounds
        for param in self.all_parameters:
            if param in common_bounds:
                bounds[param] = common_bounds[param]
            else:
                # Default bounds for unknown parameters
                bounds[param] = (1, 65536)
        
        # Refine based on relationships
        for rel in self.schema.relationships:
            # If a parameter must divide another dimension,
            # it can't be larger than typical tensor dimensions
            if rel.relation == RelationType.MULTIPLE:
                param = rel.target_interface
                if param in bounds:
                    bounds[param] = (bounds[param][0], min(bounds[param][1], 4096))
        
        return bounds
    
    # Public API for efficient runtime queries
    
    def is_model_affecting(self, attr_name: str) -> bool:
        """Check if attribute affects model creation (O(1)).
        
        Args:
            attr_name: Attribute name to check
            
        Returns:
            True if attribute affects any model component
        """
        return attr_name in self.all_parameters
    
    def affects_component(self, attr_name: str, component: str) -> bool:
        """Check if attribute affects specific component (O(1)).
        
        Args:
            attr_name: Attribute name to check
            component: Component name (e.g., "resolved_config", "kernel_model")
            
        Returns:
            True if attribute affects the component
        """
        return component in self.attr_dependencies.get(attr_name, set())
    
    def get_affected_caches(self, attr_name: str) -> Set[str]:
        """Get all caches affected by an attribute change.
        
        Args:
            attr_name: Changed attribute name
            
        Returns:
            Set of cache names to invalidate
        """
        return self.attr_dependencies.get(attr_name, set())
    
    def get_invalidating_attrs(self, cache_name: str) -> Set[str]:
        """Get all attributes that invalidate a specific cache.
        
        Args:
            cache_name: Name of cache component
            
        Returns:
            Set of attribute names that invalidate this cache
        """
        return self.invalidation_groups.get(cache_name, set())
    
    def get_template_params(self, template_key: str) -> Set[str]:
        """Get parameter names for a specific template (O(1)).
        
        Args:
            template_key: Template identifier (e.g., "input_X_block")
            
        Returns:
            Set of parameter names in the template
        """
        return self.template_metadata.get(template_key, {}).get("params", set())
    
    @cached_property
    def datatype_attrs(self) -> Set[str]:
        """Get all datatype attribute names."""
        attrs = set()
        for i in range(len(self.schema.inputs)):
            attrs.add(f"input{i}Datatype")
        for i in range(len(self.schema.outputs)):
            attrs.add(f"output{i}Datatype")
        
        # Also include custom datatype attrs
        for inp in self.schema.inputs:
            if inp.datatype_attr:
                attrs.add(inp.datatype_attr)
        for out in self.schema.outputs:
            if out.datatype_attr:
                attrs.add(out.datatype_attr)
        
        return attrs
    
    @cached_property
    def performance_attrs(self) -> Set[str]:
        """Get attributes that only affect performance calculations."""
        return {"clock_freq_mhz"}
    
    def validate_shapes_static(self, params: Dict[str, int]) -> ValidationResult:
        """Validate shapes at compile time when parameters are known.
        
        Args:
            params: Dictionary of parameter values
            
        Returns:
            ValidationResult with any constraint violations
        """
        violations = []
        
        # Check parameter bounds
        for param_name, value in params.items():
            if param_name in self.parameter_bounds:
                min_val, max_val = self.parameter_bounds[param_name]
                if value < min_val or value > max_val:
                    violations.append(ConstraintViolation(
                        constraint_type="parameter_bounds",
                        message=f"Parameter '{param_name}' value {value} outside bounds [{min_val}, {max_val}]",
                        severity="error"
                    ))
        
        # Check shape constraints that can be validated statically
        for key, constraint in self.shape_constraints.items():
            if constraint.constraint_type == ConstraintType.PARAMETER:
                # Can validate if parameter is provided
                if constraint.value in params:
                    param_value = params[constraint.value]
                    # Would need actual shape to validate, skip for now
            elif constraint.constraint_type == ConstraintType.MULTIPLE:
                # Check divisibility constraints when possible
                if constraint.value in params:
                    divisor = params[constraint.value]
                    # Would need actual shape to validate
        
        # Check for required parameters
        required_params = set()
        for template_meta in self.template_metadata.values():
            required_params.update(template_meta["params"])
        
        missing_params = required_params - set(params.keys())
        if missing_params:
            violations.append(ConstraintViolation(
                constraint_type="missing_parameters",
                message=f"Missing required parameters: {', '.join(sorted(missing_params))}",
                severity="error",
                details={"missing": list(missing_params)}
            ))
        
        return ValidationResult(violations=violations)
    
    def get_shape_constraints_for_interface(self, interface_name: str) -> List[ShapeConstraint]:
        """Get all shape constraints for a specific interface.
        
        Args:
            interface_name: Name of the interface
            
        Returns:
            List of ShapeConstraint objects for this interface
        """
        return [
            constraint for constraint in self.shape_constraints.values()
            if constraint.interface == interface_name
        ]
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CompiledSchema(kernel='{self.schema.name}', "
            f"parameters={len(self.all_parameters)}, "
            f"templates={len(self.template_metadata)}, "
            f"constraints={len(self.shape_constraints)})"
        )


class SchemaCompiler:
    """Compiler for KernelSchema objects.
    
    This class provides factory methods for creating CompiledSchema instances
    and manages caching of compiled schemas.
    """
    
    # Class-level cache for compiled schemas
    _cache: Dict[int, CompiledSchema] = {}
    
    @classmethod
    def compile(cls, schema: KernelSchema) -> CompiledSchema:
        """Compile a KernelSchema for efficient runtime use.
        
        Args:
            schema: KernelSchema to compile
            
        Returns:
            CompiledSchema instance
        """
        # Use schema object id for caching
        schema_id = id(schema)
        
        if schema_id not in cls._cache:
            cls._cache[schema_id] = CompiledSchema(schema)
        
        return cls._cache[schema_id]
    
    @classmethod
    def clear_cache(cls) -> None:
        """Clear the compilation cache."""
        cls._cache.clear()