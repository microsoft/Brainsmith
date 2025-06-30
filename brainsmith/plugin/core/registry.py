"""
Plugin Registry

Central registry for all discovered plugins with advanced query capabilities.
Separates the concerns of plugin storage/querying from discovery and loading.
"""

import logging
from threading import RLock
from typing import Dict, List, Optional, Set, Any, Callable
from collections import defaultdict

from .data_models import PluginInfo, PluginCatalog

logger = logging.getLogger(__name__)


class PluginRegistry:
    """
    Central registry for all plugins with advanced query and filter capabilities.
    
    This class is responsible for:
    - Storing discovered plugins
    - Providing query interfaces
    - Managing plugin metadata
    - Handling conflict resolution
    
    It does NOT handle discovery or loading - those are separate concerns.
    """
    
    def __init__(self):
        self._lock = RLock()
        self._catalog = PluginCatalog()
        self._indexes: Dict[str, Dict[str, Set[PluginInfo]]] = {}
        self._custom_filters: Dict[str, Callable] = {}
    
    def register_plugin(self, plugin: PluginInfo) -> None:
        """
        Register a plugin in the registry.
        
        Thread-safe registration with automatic indexing.
        """
        with self._lock:
            self._catalog.add_plugin(plugin)
            self._update_indexes(plugin)
            logger.debug(f"Registered plugin: {plugin}")
    
    def register_plugins(self, plugins: List[PluginInfo]) -> None:
        """Register multiple plugins at once."""
        with self._lock:
            for plugin in plugins:
                self.register_plugin(plugin)
    
    def get_plugin(self, name: str, framework: str = None) -> Optional[PluginInfo]:
        """
        Get a specific plugin by name and optionally framework.
        
        Raises:
            ValueError: If plugin name is ambiguous
        """
        with self._lock:
            return self._catalog.get_plugin(name, framework)
    
    def find_plugins(self, **criteria) -> List[PluginInfo]:
        """
        Find plugins matching given criteria.
        
        Supports filtering by any plugin attribute or metadata field.
        """
        with self._lock:
            return self._catalog.find_plugins(**criteria)
    
    def query(self, query_func: Callable[[PluginInfo], bool]) -> List[PluginInfo]:
        """
        Query plugins with a custom function.
        
        Args:
            query_func: Function that takes PluginInfo and returns bool
            
        Returns:
            List of plugins where query_func returns True
        """
        with self._lock:
            results = []
            for plugin_list in self._catalog.plugins_by_name.values():
                for plugin in plugin_list:
                    if query_func(plugin):
                        results.append(plugin)
            return results
    
    def list_by_type(self, plugin_type: str) -> List[PluginInfo]:
        """List all plugins of a specific type."""
        with self._lock:
            return self._catalog.plugins_by_type.get(plugin_type, []).copy()
    
    def list_by_framework(self, framework: str) -> List[PluginInfo]:
        """List all plugins from a specific framework."""
        with self._lock:
            return self._catalog.plugins_by_framework.get(framework, []).copy()
    
    def list_transforms(self, framework: str = None, stage: str = None) -> List[PluginInfo]:
        """
        List transform plugins with optional filtering.
        
        Args:
            framework: Filter by framework
            stage: Filter by compilation stage
        """
        with self._lock:
            transforms = self._catalog.list_transforms(framework)
            
            if stage:
                transforms = [t for t in transforms if t.stage == stage]
            
            return transforms
    
    def list_kernels(self, framework: str = None) -> List[PluginInfo]:
        """List kernel plugins with optional framework filter."""
        with self._lock:
            return self._catalog.list_kernels(framework)
    
    def list_backends(self, kernel: str = None, backend_type: str = None) -> List[PluginInfo]:
        """
        List backend plugins with optional filtering.
        
        Args:
            kernel: Filter by kernel name
            backend_type: Filter by backend type (hls/rtl)
        """
        with self._lock:
            backends = self._catalog.list_backends(kernel)
            
            if backend_type:
                backends = [b for b in backends if b.backend_type == backend_type]
            
            return backends
    
    def list_steps(self, category: str = None) -> List[PluginInfo]:
        """List step plugins with optional category filter."""
        with self._lock:
            return self._catalog.list_steps(category)
    
    def list_kernel_inference_transforms(self, kernel: str = None) -> List[PluginInfo]:
        """List kernel inference transforms with optional kernel filter."""
        with self._lock:
            transforms = self._catalog.plugins_by_type.get("kernel_inference", [])
            
            if kernel:
                transforms = [t for t in transforms if t.kernel == kernel]
            
            return transforms
    
    def get_conflicts(self) -> Dict[str, List[PluginInfo]]:
        """Get all naming conflicts."""
        with self._lock:
            return self._catalog.conflicts.copy()
    
    def get_unique_plugins(self) -> Dict[str, PluginInfo]:
        """Get all plugins with unique names."""
        with self._lock:
            return self._catalog.unique_plugins.copy()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of registry contents."""
        with self._lock:
            summary = self._catalog.get_summary()
            
            # Add additional statistics
            summary['stages'] = self._get_stage_distribution()
            summary['kernel_backends'] = self._get_kernel_backend_summary()
            
            return summary
    
    def _get_stage_distribution(self) -> Dict[str, int]:
        """Get distribution of transforms by stage."""
        stages = defaultdict(int)
        
        for transform in self._catalog.list_transforms():
            stage = transform.stage or "unspecified"
            stages[stage] += 1
        
        return dict(stages)
    
    def _get_kernel_backend_summary(self) -> Dict[str, Dict[str, int]]:
        """Get summary of backends available for each kernel."""
        kernel_backends = defaultdict(lambda: defaultdict(int))
        
        for backend in self._catalog.list_backends():
            kernel = backend.kernel
            backend_type = backend.backend_type
            if kernel and backend_type:
                kernel_backends[kernel][backend_type] += 1
        
        return {k: dict(v) for k, v in kernel_backends.items()}
    
    def _update_indexes(self, plugin: PluginInfo) -> None:
        """Update custom indexes for efficient querying."""
        # Index by stage (for transforms)
        if plugin.stage:
            if 'stage' not in self._indexes:
                self._indexes['stage'] = defaultdict(set)
            self._indexes['stage'][plugin.stage].add(plugin)
        
        # Index by kernel (for kernel inference and backends)
        if plugin.kernel:
            if 'kernel' not in self._indexes:
                self._indexes['kernel'] = defaultdict(set)
            self._indexes['kernel'][plugin.kernel].add(plugin)
        
        # Index by category (for steps)
        category = plugin.metadata.get('category')
        if category:
            if 'category' not in self._indexes:
                self._indexes['category'] = defaultdict(set)
            self._indexes['category'][category].add(plugin)
    
    def get_by_index(self, index_name: str, key: str) -> Set[PluginInfo]:
        """Get plugins from a custom index."""
        with self._lock:
            if index_name in self._indexes and key in self._indexes[index_name]:
                return self._indexes[index_name][key].copy()
            return set()
    
    def clear(self) -> None:
        """Clear all registered plugins."""
        with self._lock:
            self._catalog = PluginCatalog()
            self._indexes.clear()
            logger.info("Cleared plugin registry")
    
    def __repr__(self) -> str:
        with self._lock:
            return f"PluginRegistry({self._catalog!r})"


# Global registry instance
_global_registry: Optional[PluginRegistry] = None
_registry_lock = RLock()


def get_plugin_registry() -> PluginRegistry:
    """Get the global plugin registry instance."""
    global _global_registry
    
    if _global_registry is None:
        with _registry_lock:
            if _global_registry is None:
                _global_registry = PluginRegistry()
    
    return _global_registry


def set_plugin_registry(registry: PluginRegistry) -> None:
    """Set the global plugin registry (useful for testing)."""
    global _global_registry
    with _registry_lock:
        _global_registry = registry