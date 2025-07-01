"""
Enhanced Plugin Collections with Weak Reference Caching

Provides natural access patterns with memory-efficient caching using weak references.
This ensures plugin instances can be garbage collected when no longer in use.
"""

import logging
from typing import Any, Dict, Optional, Type, Union
from threading import Lock
from weakref import WeakValueDictionary

logger = logging.getLogger(__name__)


class PluginWrapper:
    """
    Enhanced plugin wrapper with weak reference caching.
    
    Uses WeakValueDictionary to allow garbage collection of unused instances
    while maintaining performance through caching.
    """
    
    def __init__(self, plugin_info, manager):
        self.plugin_info = plugin_info
        self.manager = manager
        self._instance_cache = WeakValueDictionary()  # Weak reference caching
        self._lock = Lock()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __call__(self, *args, **kwargs):
        """Create and return plugin instance with weak reference caching."""
        # Use simple caching for performance
        cache_key = (tuple(args), tuple(sorted(kwargs.items())))
        
        # Check cache first (no lock needed for read)
        instance = self._instance_cache.get(cache_key)
        if instance is not None:
            self._cache_hits += 1
            return instance
        
        # Cache miss - need to create instance
        with self._lock:
            # Double-check pattern
            instance = self._instance_cache.get(cache_key)
            if instance is not None:
                self._cache_hits += 1
                return instance
            
            try:
                # Create new instance
                instance = self.plugin_info.plugin_class(*args, **kwargs)
                
                # Only cache if instance is not explicitly marked as stateless
                if not getattr(instance, '_stateless', False):
                    self._instance_cache[cache_key] = instance
                
                self._cache_misses += 1
                return instance
                
            except Exception as e:
                logger.warning(f"Failed to instantiate {self.plugin_info.name}: {e}")
                raise
    
    def clear_cache(self):
        """Clear the instance cache to free memory."""
        with self._lock:
            self._instance_cache.clear()
            logger.debug(f"Cleared cache for {self.plugin_info.name}")
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring."""
        return {
            'plugin_name': self.plugin_info.name,
            'cache_size': len(self._instance_cache),
            'cache_hits': self._cache_hits,
            'cache_misses': self._cache_misses,
            'hit_rate': self._cache_hits / (self._cache_hits + self._cache_misses) 
                       if (self._cache_hits + self._cache_misses) > 0 else 0
        }
    
    def __repr__(self):
        return f"<PluginWrapper: {self.plugin_info.name} ({self.plugin_info.framework})>"
    
    @property
    def name(self):
        return self.plugin_info.name
    
    @property
    def framework(self):
        return self.plugin_info.framework
    
    @property
    def plugin_type(self):
        return self.plugin_info.plugin_type
    
    @property
    def metadata(self):
        return self.plugin_info.metadata


class FrameworkAccessor:
    """
    Provides framework-specific plugin access with memory management.
    """
    
    def __init__(self, collection, framework_name):
        self.collection = collection
        self.framework_name = framework_name
        self._wrapper_cache = {}  # Regular cache for wrappers (lightweight)
    
    def __getattr__(self, name):
        """Get plugin by name within this framework context."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check wrapper cache first
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]
        
        # Look for plugin in this framework
        plugin_info = self.collection.manager.get_plugin(name, framework=self.framework_name)
        if plugin_info is None:
            # Try with framework prefix
            plugin_info = self.collection.manager.get_plugin(f"{self.framework_name}:{name}")
        
        if plugin_info is None:
            available = [p.name for p in self.collection.manager.list_plugins(
                plugin_type=self.collection.plugin_type,
                framework=self.framework_name
            )]
            raise AttributeError(
                f"No {self.collection.plugin_type} plugin '{name}' found in {self.framework_name} framework. "
                f"Available: {available}"
            )
        
        # Create wrapper and cache it
        wrapper = PluginWrapper(plugin_info, self.collection.manager)
        self._wrapper_cache[name] = wrapper
        return wrapper
    
    def clear_caches(self):
        """Clear all caches in this framework accessor."""
        for wrapper in self._wrapper_cache.values():
            wrapper.clear_cache()
        logger.debug(f"Cleared all caches for {self.framework_name} framework")
    
    def get_cache_stats(self):
        """Get cache statistics for all plugins in this framework."""
        stats = []
        for wrapper in self._wrapper_cache.values():
            stats.append(wrapper.get_cache_stats())
        return stats
    
    def __dir__(self):
        """Support for tab completion."""
        plugins = self.collection.manager.list_plugins(
            plugin_type=self.collection.plugin_type,
            framework=self.framework_name
        )
        return [p.name.split(':')[-1] for p in plugins]  # Remove framework prefix if present


class BaseCollection:
    """
    Enhanced base collection with memory management capabilities.
    """
    
    def __init__(self, manager, plugin_type):
        self.manager = manager
        self.plugin_type = plugin_type
        self._wrapper_cache = {}  # Regular cache for wrappers
        self._framework_accessors = {}
    
    def __getattr__(self, name):
        """Get plugin or framework accessor by name."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if it's a framework name
        frameworks = {'brainsmith', 'qonnx', 'finn', 'external'}
        if name in frameworks:
            if name not in self._framework_accessors:
                self._framework_accessors[name] = FrameworkAccessor(self, name)
            return self._framework_accessors[name]
        
        # Check wrapper cache first
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]
        
        # Look for plugin by name (any framework)
        plugin_info = self.manager.get_plugin(name)
        if plugin_info is None or plugin_info.plugin_type != self.plugin_type:
            # Try to find in any framework
            all_plugins = self.manager.list_plugins(plugin_type=self.plugin_type)
            matches = [p for p in all_plugins if p.name == name or p.name.endswith(f':{name}')]
            
            if not matches:
                available = [p.name for p in all_plugins[:10]]  # Show first 10
                if len(all_plugins) > 10:
                    available.append(f"... and {len(all_plugins) - 10} more")
                raise AttributeError(
                    f"No {self.plugin_type} plugin '{name}' found. "
                    f"Available: {available}"
                )
            
            plugin_info = matches[0]  # Use first match
        
        # Create wrapper and cache it
        wrapper = PluginWrapper(plugin_info, self.manager)
        self._wrapper_cache[name] = wrapper
        return wrapper
    
    def clear_all_caches(self):
        """Clear all caches in this collection."""
        # Clear wrapper instance caches
        for wrapper in self._wrapper_cache.values():
            wrapper.clear_cache()
        
        # Clear framework accessor caches
        for accessor in self._framework_accessors.values():
            accessor.clear_caches()
        
        logger.info(f"Cleared all caches for {self.plugin_type} collection")
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics."""
        stats = {
            'collection_type': self.plugin_type,
            'wrapper_cache_size': len(self._wrapper_cache),
            'framework_accessors': len(self._framework_accessors),
            'plugin_stats': []
        }
        
        # Get stats from all wrappers
        for wrapper in self._wrapper_cache.values():
            stats['plugin_stats'].append(wrapper.get_cache_stats())
        
        # Get stats from framework accessors
        for framework, accessor in self._framework_accessors.items():
            framework_stats = accessor.get_cache_stats()
            for stat in framework_stats:
                stat['framework'] = framework
                stats['plugin_stats'].append(stat)
        
        return stats
    
    def __dir__(self):
        """Support for tab completion."""
        plugins = self.manager.list_plugins(plugin_type=self.plugin_type)
        names = ['brainsmith', 'qonnx', 'finn', 'external']  # Framework accessors
        names.extend([p.name.split(':')[-1] for p in plugins])  # Plugin names
        return sorted(set(names))
    
    def list_plugins(self, framework: Optional[str] = None):
        """List available plugins, optionally filtered by framework."""
        return self.manager.list_plugins(plugin_type=self.plugin_type, framework=framework)
    
    def get_plugin(self, name: str, framework: Optional[str] = None):
        """Get plugin by name, optionally filtered by framework."""
        plugin_info = self.manager.get_plugin(name, framework=framework)
        if plugin_info and plugin_info.plugin_type == self.plugin_type:
            return PluginWrapper(plugin_info, self.manager)
        return None


class TransformCollection(BaseCollection):
    """Collection for transform plugins with memory management."""
    
    def __init__(self, manager):
        super().__init__(manager, 'transform')


class KernelCollection(BaseCollection):
    """Collection for kernel plugins with memory management."""
    
    def __init__(self, manager):
        super().__init__(manager, 'kernel')


class BackendCollection(BaseCollection):
    """Collection for backend plugins with memory management."""
    
    def __init__(self, manager):
        super().__init__(manager, 'backend')


class StepsCollection:
    """
    Enhanced steps collection with category organization and memory management.
    """
    
    def __init__(self, manager):
        self.manager = manager
        self._wrapper_cache = {}
        self._category_accessors = {}
    
    def __getattr__(self, name):
        """Get step plugin or category accessor by name."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check if it's a category name
        steps = self.manager.list_plugins(plugin_type='step')
        categories = set()
        for step in steps:
            category = step.metadata.get('category', 'general')
            categories.add(category)
        
        if name in categories:
            if name not in self._category_accessors:
                self._category_accessors[name] = CategoryAccessor(self, name)
            return self._category_accessors[name]
        
        # Check wrapper cache first
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]
        
        # Look for step plugin by name
        plugin_info = self.manager.get_plugin(name)
        if plugin_info is None or plugin_info.plugin_type != 'step':
            # Try to find in any category
            all_steps = self.manager.list_plugins(plugin_type='step')
            matches = [p for p in all_steps if p.name == name or p.name.endswith(f':{name}')]
            
            if not matches:
                available = [p.name for p in all_steps[:10]]
                if len(all_steps) > 10:
                    available.append(f"... and {len(all_steps) - 10} more")
                raise AttributeError(
                    f"No step plugin '{name}' found. "
                    f"Available: {available}"
                )
            
            plugin_info = matches[0]
        
        # Create wrapper and cache it
        wrapper = PluginWrapper(plugin_info, self.manager)
        self._wrapper_cache[name] = wrapper
        return wrapper
    
    def clear_all_caches(self):
        """Clear all caches in the steps collection."""
        # Clear wrapper caches
        for wrapper in self._wrapper_cache.values():
            wrapper.clear_cache()
        
        # Clear category accessor caches
        for accessor in self._category_accessors.values():
            accessor.clear_caches()
        
        logger.info("Cleared all caches for steps collection")
    
    def get_cache_stats(self):
        """Get comprehensive cache statistics for steps."""
        stats = {
            'collection_type': 'steps',
            'wrapper_cache_size': len(self._wrapper_cache),
            'category_accessors': len(self._category_accessors),
            'plugin_stats': []
        }
        
        # Get stats from all wrappers
        for wrapper in self._wrapper_cache.values():
            stats['plugin_stats'].append(wrapper.get_cache_stats())
        
        # Get stats from category accessors
        for category, accessor in self._category_accessors.items():
            category_stats = accessor.get_cache_stats()
            for stat in category_stats:
                stat['category'] = category
                stats['plugin_stats'].append(stat)
        
        return stats
    
    def __dir__(self):
        """Support for tab completion."""
        steps = self.manager.list_plugins(plugin_type='step')
        categories = set()
        step_names = []
        
        for step in steps:
            category = step.metadata.get('category', 'general')
            categories.add(category)
            step_names.append(step.name.split(':')[-1])
        
        return sorted(list(categories) + step_names)
    
    def list_plugins(self, category: Optional[str] = None):
        """List available step plugins, optionally filtered by category."""
        steps = self.manager.list_plugins(plugin_type='step')
        if category:
            steps = [s for s in steps if s.metadata.get('category', 'general') == category]
        return steps


class CategoryAccessor:
    """
    Provides category-specific step access with memory management.
    """
    
    def __init__(self, collection, category_name):
        self.collection = collection
        self.category_name = category_name
        self._wrapper_cache = {}
    
    def __getattr__(self, name):
        """Get step plugin by name within this category."""
        if name.startswith('_'):
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
        
        # Check wrapper cache first
        if name in self._wrapper_cache:
            return self._wrapper_cache[name]
        
        # Look for step in this category
        steps = self.collection.manager.list_plugins(plugin_type='step')
        matches = []
        for step in steps:
            step_category = step.metadata.get('category', 'general')
            if step_category == self.category_name and (step.name == name or step.name.endswith(f':{name}')):
                matches.append(step)
        
        if not matches:
            available = []
            for step in steps:
                step_category = step.metadata.get('category', 'general')
                if step_category == self.category_name:
                    available.append(step.name.split(':')[-1])
            
            raise AttributeError(
                f"No step plugin '{name}' found in category '{self.category_name}'. "
                f"Available: {available}"
            )
        
        # Create wrapper and cache it
        plugin_info = matches[0]
        wrapper = PluginWrapper(plugin_info, self.collection.manager)
        self._wrapper_cache[name] = wrapper
        return wrapper
    
    def clear_caches(self):
        """Clear all caches in this category accessor."""
        for wrapper in self._wrapper_cache.values():
            wrapper.clear_cache()
        logger.debug(f"Cleared all caches for {self.category_name} category")
    
    def get_cache_stats(self):
        """Get cache statistics for all plugins in this category."""
        stats = []
        for wrapper in self._wrapper_cache.values():
            stats.append(wrapper.get_cache_stats())
        return stats
    
    def __dir__(self):
        """Support for tab completion."""
        steps = self.collection.manager.list_plugins(plugin_type='step')
        names = []
        for step in steps:
            step_category = step.metadata.get('category', 'general')
            if step_category == self.category_name:
                names.append(step.name.split(':')[-1])
        return sorted(names)


def create_collections(manager):
    """
    Factory function to create all collection instances with memory management.
    
    Returns collections with weak reference caching and memory management capabilities.
    """
    return {
        'transforms': TransformCollection(manager),
        'kernels': KernelCollection(manager),
        'backends': BackendCollection(manager),
        'steps': StepsCollection(manager)
    }