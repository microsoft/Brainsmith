"""
Plugin Dependency Management

BREAKING CHANGE: Explicit dependency declarations are now required.
"""

import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

from .contracts import PluginDependency, SystemDependency, ValidationResult

logger = logging.getLogger(__name__)


class DependencyResolutionError(Exception):
    """Raised when dependency resolution fails"""
    pass


@dataclass
class ResolvedDependency:
    """A resolved dependency with its plugin spec"""
    dependency: PluginDependency
    plugin_spec: 'PluginSpec'
    satisfied: bool
    error: Optional[str] = None


class DependencyResolver:
    """
    Resolves plugin dependencies and detects circular dependencies.
    
    BREAKING CHANGE: Dependency resolution is now mandatory.
    """
    
    def __init__(self, registry: 'PluginRegistry'):
        self.registry = registry
    
    def resolve_dependencies(self, plugin_spec: 'PluginSpec') -> List[ResolvedDependency]:
        """
        Resolve all dependencies for a plugin.
        
        Args:
            plugin_spec: Plugin to resolve dependencies for
            
        Returns:
            List of resolved dependencies
            
        Raises:
            DependencyResolutionError: If circular dependencies detected
        """
        resolved = []
        
        for dependency in plugin_spec.dependencies:
            if isinstance(dependency, PluginDependency):
                resolved_dep = self._resolve_plugin_dependency(dependency)
                resolved.append(resolved_dep)
            elif isinstance(dependency, SystemDependency):
                # System dependencies are handled during validation
                continue
        
        # Check for circular dependencies
        self._check_circular_dependencies(plugin_spec, resolved)
        
        return resolved
    
    def _resolve_plugin_dependency(self, dependency: PluginDependency) -> ResolvedDependency:
        """Resolve a single plugin dependency"""
        # Find matching plugins
        from .query import PluginQuery
        
        query = PluginQuery(name=dependency.name)
        if dependency.plugin_type:
            query.type = dependency.plugin_type
        
        matches = self.registry.query(query)
        
        if not matches:
            return ResolvedDependency(
                dependency=dependency,
                plugin_spec=None,
                satisfied=False,
                error=f"Dependency '{dependency.name}' not found"
            )
        
        # If multiple matches, pick the best one
        # TODO: Implement proper version constraint checking
        best_match = matches[0]
        
        return ResolvedDependency(
            dependency=dependency,
            plugin_spec=best_match,
            satisfied=True
        )
    
    def _check_circular_dependencies(self, plugin_spec: 'PluginSpec', 
                                   resolved: List[ResolvedDependency]):
        """Check for circular dependencies"""
        visited = set()
        path = []
        
        def visit(spec: 'PluginSpec'):
            if spec.plugin_id in path:
                cycle = " -> ".join(path[path.index(spec.plugin_id):] + [spec.plugin_id])
                raise DependencyResolutionError(f"Circular dependency detected: {cycle}")
            
            if spec.plugin_id in visited:
                return
            
            visited.add(spec.plugin_id)
            path.append(spec.plugin_id)
            
            # Recursively check dependencies
            for dep in resolved:
                if dep.plugin_spec and dep.satisfied:
                    dep_resolved = self.resolve_dependencies(dep.plugin_spec)
                    self._check_circular_dependencies(dep.plugin_spec, dep_resolved)
            
            path.pop()
        
        visit(plugin_spec)
    
    def get_dependency_graph(self) -> Dict[str, List[str]]:
        """
        Get the complete dependency graph.
        
        Returns:
            Dictionary mapping plugin_id -> list of dependency plugin_ids
        """
        graph = {}
        
        for plugin_spec in self.registry.get_active_plugins():
            dependencies = []
            
            for dep in plugin_spec.dependencies:
                if isinstance(dep, PluginDependency):
                    resolved = self._resolve_plugin_dependency(dep)
                    if resolved.satisfied and resolved.plugin_spec:
                        dependencies.append(resolved.plugin_spec.plugin_id)
            
            graph[plugin_spec.plugin_id] = dependencies
        
        return graph
    
    def topological_sort(self) -> List['PluginSpec']:
        """
        Return plugins in dependency order (topological sort).
        
        Returns:
            List of plugin specs in dependency order
            
        Raises:
            DependencyResolutionError: If circular dependencies exist
        """
        graph = self.get_dependency_graph()
        visited = set()
        visiting = set()
        result = []
        
        def visit(plugin_id: str):
            if plugin_id in visiting:
                raise DependencyResolutionError(f"Circular dependency involving {plugin_id}")
            
            if plugin_id in visited:
                return
            
            visiting.add(plugin_id)
            
            # Visit dependencies first
            for dep_id in graph.get(plugin_id, []):
                visit(dep_id)
            
            visiting.remove(plugin_id)
            visited.add(plugin_id)
            
            # Add to result
            plugin_spec = self.registry.get_plugin_spec(plugin_id)
            if plugin_spec:
                result.append(plugin_spec)
        
        # Visit all plugins
        for plugin_id in graph:
            visit(plugin_id)
        
        return result