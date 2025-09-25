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

from .schemas import KernelSchema, InputSchema, OutputSchema
from .template_utils import extract_tiling_parameters


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
    
    def __post_init__(self):
        """Extract all metadata from schema."""
        self.all_parameters = self._extract_all_parameters()
        self.attr_dependencies = self._build_dependency_map()
        self.static_resolutions = self._pre_resolve_static()
        self.invalidation_groups = self._build_invalidation_groups()
        self.template_metadata = self._extract_template_metadata()
    
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
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CompiledSchema(kernel='{self.schema.name}', "
            f"parameters={len(self.all_parameters)}, "
            f"templates={len(self.template_metadata)})"
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