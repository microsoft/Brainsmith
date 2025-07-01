"""
Enhanced Plugin Manager with Memory Management

Adds memory management capabilities to the simplified plugin manager
while maintaining the pragmatic hybrid discovery approach.
"""

import logging
import importlib
import pkgutil
import gc
from typing import Dict, List, Optional, Type, Any, Set
from dataclasses import dataclass, field
from threading import Lock
from datetime import datetime

try:
    from stevedore import extension
except ImportError:
    extension = None

from .data_models import PluginInfo
from .framework_adapters import UnifiedFrameworkDiscovery

logger = logging.getLogger(__name__)


class PluginManager:
    """
    Enhanced plugin manager with memory management capabilities.
    
    Maintains the simplified design while adding:
    - Memory usage tracking
    - Cache cleanup methods
    - Discovery prioritization
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginInfo] = {}  # name -> PluginInfo
        self._discovery_done = False
        self._lock = Lock()  # Thread safety
        
        # Framework discovery system
        self._framework_discovery = UnifiedFrameworkDiscovery()
        
        # Performance and caching
        self._discovery_cache: Optional[Dict[str, List[PluginInfo]]] = None
        self._discovery_cache_timestamp: Optional[float] = None
        self._cache_ttl = 300  # 5 minutes cache TTL
        
        # Memory management tracking
        self._discovery_stats = {
            'stevedore': 0,
            'internal': 0,
            'framework': 0,
            'discovery_time': None,
            'cache_hits': 0,
            'cache_misses': 0
        }
        self._cache_references: Set[Any] = set()  # Track collection instances
    
    def discover_plugins(self, modes=None, frameworks=None, types=None) -> None:
        """
        Discover plugins using conditional discovery based on requirements.
        
        Args:
            modes: Discovery modes ('full', 'blueprint', 'selective'). Default: 'full'
            frameworks: List of frameworks to discover ('brainsmith', 'qonnx', 'finn'). Default: all
            types: List of plugin types to discover. Default: all
        
        Discovery Priority:
        1. Stevedore entry points (external plugins)
        2. Internal module scanning (brainsmith plugins)
        3. Framework registries (qonnx/finn plugins)
        """
        with self._lock:
            # Check if discovery can be served from cache
            cache_key = self._generate_cache_key(modes, frameworks, types)
            if self._can_use_cache(cache_key):
                self._load_from_cache(cache_key)
                self._discovery_stats['cache_hits'] += 1
                return
            
            self._discovery_stats['cache_misses'] += 1
            
            if self._discovery_done:
                return
            
            # Set default modes
            if modes is None:
                modes = ['full']
            elif isinstance(modes, str):
                modes = [modes]
                
            start_time = datetime.now()
            
            # Conditional discovery based on modes and requirements
            stevedore_count = 0
            internal_count = 0
            framework_count = 0
            
            # Always discover Stevedore (lightweight, external plugins)
            if self._should_discover_stevedore(modes, frameworks):
                stevedore_count = self._discover_stevedore()
                self._discovery_stats['stevedore'] = stevedore_count
            
            # Always discover internal plugins (zero-friction development)
            if self._should_discover_internal(modes, frameworks):
                internal_count = self._discover_internal_plugins()
                self._discovery_stats['internal'] = internal_count
            
            # Conditionally discover framework plugins
            if self._should_discover_frameworks(modes, frameworks):
                framework_count = self._discover_framework_plugins(frameworks)
                self._discovery_stats['framework'] = framework_count
            
            self._discovery_done = True
            self._discovery_stats['discovery_time'] = (datetime.now() - start_time).total_seconds()
            
            # Cache the results
            self._cache_discovery_results(cache_key)
            
            logger.info(f"Discovered {len(self._plugins)} plugins in {self._discovery_stats['discovery_time']:.2f}s")
            logger.debug(f"Discovery breakdown: {self._discovery_stats}")
    
    def _discover_stevedore(self) -> int:
        """Discover plugins via Stevedore entry points."""
        if not extension:
            logger.debug("Stevedore not available")
            return 0
        
        count = 0
        try:
            # Standard plugin namespace
            mgr = extension.ExtensionManager(
                namespace='brainsmith.plugins',
                invoke_on_load=False
            )
            
            for ext in mgr.extensions:
                try:
                    plugin_class = ext.obj
                    if self._register_from_class(plugin_class, 'brainsmith', 'stevedore'):
                        count += 1
                except Exception as e:
                    logger.warning(f"Failed to load Stevedore plugin {ext.name}: {e}")
                    
        except Exception as e:
            logger.debug(f"Stevedore discovery failed: {e}")
        
        return count
    
    def _discover_internal_plugins(self) -> int:
        """Discover BrainSmith internal plugins via framework adapter."""
        count = 0
        try:
            # Use BrainSmith adapter for internal plugin discovery
            brainsmith_adapter = self._framework_discovery.adapters['brainsmith']
            plugins = brainsmith_adapter.discover_plugins()
            
            # Register discovered plugins
            for plugin_info in plugins:
                self.register_plugin(plugin_info)
                count += 1
                
        except Exception as e:
            logger.debug(f"Internal discovery failed: {e}")
        
        return count
    
    def _discover_framework_plugins(self, frameworks: Optional[List[str]] = None) -> int:
        """Discover plugins from framework registries using adapters."""
        count = 0
        
        try:
            if frameworks:
                # Discover only specified frameworks
                plugins = self._framework_discovery.discover_framework_plugins(frameworks)
            else:
                # Discover all available frameworks
                plugins = self._framework_discovery.discover_all_plugins()
            
            # Register discovered plugins
            for plugin_info in plugins:
                self.register_plugin(plugin_info)
                count += 1
                
        except Exception as e:
            logger.warning(f"Framework discovery failed: {e}")
        
        return count
    
    def _should_discover_stevedore(self, modes: List[str], frameworks: Optional[List[str]]) -> bool:
        """Determine if Stevedore discovery should run."""
        # Always discover external plugins (lightweight discovery)
        return True
    
    def _should_discover_internal(self, modes: List[str], frameworks: Optional[List[str]]) -> bool:
        """Determine if internal plugin discovery should run."""
        # Always discover internal plugins for zero-friction development
        return True
    
    def _should_discover_frameworks(self, modes: List[str], frameworks: Optional[List[str]]) -> bool:
        """Determine if framework plugin discovery should run."""
        # Only discover frameworks if:
        # 1. Full discovery mode is requested
        # 2. Specific frameworks are requested
        # 3. Blueprint mode with framework plugins specified
        
        if 'full' in modes:
            return True
            
        if frameworks:
            # Check if any framework plugins are specifically requested
            return bool(set(frameworks) & {'qonnx', 'finn'})
            
        # For blueprint and selective modes, only discover if explicitly needed
        return False
    
    def _generate_cache_key(self, modes: Optional[List[str]], frameworks: Optional[List[str]], types: Optional[List[str]]) -> str:
        """Generate a cache key for discovery parameters."""
        import hashlib
        
        # Normalize parameters
        modes_str = ','.join(sorted(modes or ['full']))
        frameworks_str = ','.join(sorted(frameworks or ['all']))
        types_str = ','.join(sorted(types or ['all']))
        
        # Create cache key
        key_data = f"{modes_str}|{frameworks_str}|{types_str}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _can_use_cache(self, cache_key: str) -> bool:
        """Check if cache can be used for the given key."""
        import time
        
        if self._discovery_cache is None:
            return False
        
        if cache_key not in self._discovery_cache:
            return False
        
        # Check TTL
        if self._discovery_cache_timestamp is None:
            return False
        
        age = time.time() - self._discovery_cache_timestamp
        if age > self._cache_ttl:
            logger.debug(f"Cache expired (age: {age:.1f}s, TTL: {self._cache_ttl}s)")
            self._discovery_cache = None
            self._discovery_cache_timestamp = None
            return False
        
        return True
    
    def _load_from_cache(self, cache_key: str) -> None:
        """Load plugins from cache."""
        if self._discovery_cache and cache_key in self._discovery_cache:
            cached_plugins = self._discovery_cache[cache_key]
            
            # Load cached plugins into manager
            for plugin_info in cached_plugins:
                self._plugins[plugin_info.name] = plugin_info
            
            self._discovery_done = True
            logger.debug(f"Loaded {len(cached_plugins)} plugins from cache")
    
    def _cache_discovery_results(self, cache_key: str) -> None:
        """Cache current discovery results."""
        import time
        
        if self._discovery_cache is None:
            self._discovery_cache = {}
        
        # Cache all discovered plugins
        self._discovery_cache[cache_key] = list(self._plugins.values())
        self._discovery_cache_timestamp = time.time()
        
        logger.debug(f"Cached {len(self._plugins)} plugins with key: {cache_key}")
    
    def clear_discovery_cache(self) -> None:
        """Clear the discovery cache."""
        with self._lock:
            self._discovery_cache = None
            self._discovery_cache_timestamp = None
            logger.debug("Discovery cache cleared")
    
    
    def register_plugin(self, plugin_info: PluginInfo) -> None:
        """Register a plugin with conflict resolution."""
        key = plugin_info.name
        
        # Check for conflicts
        if key in self._plugins:
            existing = self._plugins[key]
            
            # Prioritize by discovery source
            existing_source = existing.metadata.get('discovery_source', 'unknown')
            new_source = plugin_info.metadata.get('discovery_source', 'unknown')
            
            # Priority: stevedore > qonnx_manual_registry > module_scan > qonnx_registry > unknown
            source_priority = {
                'stevedore': 0,
                'qonnx_manual_registry': 1,
                'module_scan': 2,
                'qonnx_registry': 3,
                'unknown': 4
            }
            
            if source_priority.get(new_source, 4) < source_priority.get(existing_source, 4):
                # New plugin has higher priority
                logger.debug(f"Replacing {key} from {existing_source} with {new_source}")
                self._plugins[key] = plugin_info
            else:
                # Keep existing plugin
                logger.debug(f"Keeping existing {key} from {existing_source} over {new_source}")
        else:
            self._plugins[key] = plugin_info
    
    def get_plugin(self, name: str, framework: Optional[str] = None) -> Optional[PluginInfo]:
        """Get a plugin by name, optionally filtered by framework."""
        # Full discovery for manual plugin access
        self._ensure_discovered(modes=['full'])
        
        # Direct lookup
        if name in self._plugins:
            plugin = self._plugins[name]
            if framework is None or plugin.framework == framework:
                return plugin
        
        # Try with framework prefix
        if framework:
            prefixed = f"{framework}:{name}"
            if prefixed in self._plugins:
                return self._plugins[prefixed]
        
        return None
    
    def list_plugins(self, plugin_type: Optional[str] = None, 
                    framework: Optional[str] = None) -> List[PluginInfo]:
        """List plugins with optional filtering."""
        # Full discovery for listing plugins
        self._ensure_discovered(modes=['full'])
        
        plugins = list(self._plugins.values())
        
        if plugin_type:
            plugins = [p for p in plugins if p.plugin_type == plugin_type]
        
        if framework:
            plugins = [p for p in plugins if p.framework == framework]
        
        return plugins
    
    def get_summary(self) -> Dict[str, Any]:
        """Get comprehensive plugin system summary."""
        self._ensure_discovered()
        
        plugins = list(self._plugins.values())
        
        # Count by type
        by_type = {}
        for plugin in plugins:
            plugin_type = plugin.plugin_type
            by_type[plugin_type] = by_type.get(plugin_type, 0) + 1
        
        # Count by framework
        by_framework = {}
        for plugin in plugins:
            framework = plugin.framework
            by_framework[framework] = by_framework.get(framework, 0) + 1
        
        # Count by discovery source
        by_source = {}
        for plugin in plugins:
            source = plugin.metadata.get('discovery_source', 'unknown')
            by_source[source] = by_source.get(source, 0) + 1
        
        # Add performance metrics
        cache_hit_rate = 0.0
        total_cache_ops = self._discovery_stats['cache_hits'] + self._discovery_stats['cache_misses']
        if total_cache_ops > 0:
            cache_hit_rate = self._discovery_stats['cache_hits'] / total_cache_ops
        
        return {
            'total_plugins': len(plugins),
            'by_type': by_type,
            'by_framework': by_framework,
            'by_source': by_source,
            'discovery_stats': self._discovery_stats.copy(),
            'performance_stats': {
                'cache_hit_rate': cache_hit_rate,
                'cache_enabled': self._discovery_cache is not None,
                'cache_ttl': self._cache_ttl,
                'available_frameworks': self._framework_discovery.get_available_frameworks()
            }
        }
    
    def _ensure_discovered(self, modes=None, frameworks=None, types=None) -> None:
        """Ensure plugins have been discovered with specified parameters."""
        if not self._discovery_done:
            self.discover_plugins(modes=modes, frameworks=frameworks, types=types)
    
    def reset(self) -> None:
        """Reset the plugin manager state."""
        with self._lock:
            self._plugins.clear()
            self._discovery_done = False
            self._discovery_stats = {
                'stevedore': 0,
                'internal': 0,
                'framework': 0,
                'discovery_time': None,
                'cache_hits': 0,
                'cache_misses': 0
            }
            logger.debug("Reset plugin manager state")
    
    # Memory management methods
    
    def register_collection(self, collection: Any) -> None:
        """Register a collection instance for memory tracking."""
        self._cache_references.add(collection)
    
    def clear_all_caches(self) -> None:
        """Clear all caches across all registered collections."""
        logger.info("Clearing all plugin caches...")
        
        # Clear caches in all registered collections
        for collection in self._cache_references:
            if hasattr(collection, 'clear_all_caches'):
                try:
                    collection.clear_all_caches()
                except Exception as e:
                    logger.warning(f"Failed to clear cache for {collection}: {e}")
        
        # Force garbage collection
        gc.collect()
        logger.info("Cache clearing complete")
    
    def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        stats = {
            'plugin_count': len(self._plugins),
            'registered_collections': len(self._cache_references),
            'collection_stats': []
        }
        
        # Get stats from each collection
        for collection in self._cache_references:
            if hasattr(collection, 'get_cache_stats'):
                try:
                    collection_stats = collection.get_cache_stats()
                    stats['collection_stats'].append(collection_stats)
                except Exception as e:
                    logger.warning(f"Failed to get stats for {collection}: {e}")
        
        return stats
    
    def load_for_blueprint(self, blueprint_requirements: Dict[str, List[str]]) -> Dict[str, List[PluginInfo]]:
        """
        Load only plugins specified in blueprint requirements.
        
        Args:
            blueprint_requirements: Dict with keys like 'transforms', 'kernels', 'backends'
                                   and values as lists of plugin names
        
        Returns:
            Dict mapping plugin types to lists of loaded PluginInfo objects
        """
        with self._lock:
            # Force discovery in blueprint mode with selective loading
            self._discovery_done = False  # Reset to allow selective discovery
            
            # Determine required frameworks from plugin names
            required_frameworks = self._analyze_blueprint_frameworks(blueprint_requirements)
            
            # Selective discovery based on blueprint needs
            self.discover_plugins(modes=['blueprint'], frameworks=required_frameworks)
            
            # Filter to only requested plugins from discovered plugins
            result = {}
            for plugin_type, requested_names in blueprint_requirements.items():
                result[plugin_type] = []
                for name in requested_names:
                    # Direct lookup in discovered plugins
                    if name in self._plugins:
                        plugin = self._plugins[name]
                        if plugin.plugin_type == plugin_type:
                            result[plugin_type].append(plugin)
                            continue
                    
                    # Try framework-prefixed lookup
                    found = False
                    for framework in ['brainsmith', 'qonnx', 'finn']:
                        prefixed = f"{framework}:{name}"
                        if prefixed in self._plugins:
                            plugin = self._plugins[prefixed]
                            if plugin.plugin_type == plugin_type:
                                result[plugin_type].append(plugin)
                                found = True
                                break
                    
                    if not found:
                        logger.warning(f"Blueprint plugin not found: {plugin_type}.{name}")
            
            return result
    
    def _analyze_blueprint_frameworks(self, blueprint_requirements: Dict[str, List[str]]) -> List[str]:
        """Analyze blueprint to determine which frameworks are needed."""
        # For now, we'll do a simple analysis
        # In the future, this could use plugin name patterns or explicit framework specs
        frameworks = ['brainsmith']  # Always include brainsmith
        
        # Add framework-specific plugins if they exist
        all_names = []
        for names in blueprint_requirements.values():
            all_names.extend(names)
        
        # Simple heuristics for framework detection
        for name in all_names:
            if name.startswith('qonnx_') or 'qonnx' in name.lower():
                frameworks.append('qonnx')
            elif name.startswith('finn_') or 'finn' in name.lower():
                frameworks.append('finn')
        
        return list(set(frameworks))
    
    def optimize_memory(self) -> None:
        """
        Optimize memory usage by clearing unused caches.
        
        This is a more selective approach than clear_all_caches,
        only clearing caches with low hit rates.
        """
        logger.info("Optimizing plugin memory usage...")
        
        cleared = 0
        for collection in self._cache_references:
            if hasattr(collection, 'get_cache_stats') and hasattr(collection, 'clear_all_caches'):
                try:
                    stats = collection.get_cache_stats()
                    
                    # Clear collections with low hit rate
                    if isinstance(stats, dict) and 'plugin_stats' in stats:
                        for plugin_stat in stats['plugin_stats']:
                            if plugin_stat.get('hit_rate', 1.0) < 0.2:  # Less than 20% hit rate
                                # This is inefficient, but we'd need per-plugin clearing
                                logger.debug(f"Low hit rate for {plugin_stat.get('plugin_name')}")
                                
                except Exception as e:
                    logger.warning(f"Failed to optimize {collection}: {e}")
        
        gc.collect()
        logger.info("Memory optimization complete")


# Global plugin manager instance
_global_manager: Optional[PluginManager] = None
_manager_lock = Lock()


def get_plugin_manager() -> PluginManager:
    """Get the global plugin manager instance."""
    global _global_manager
    
    if _global_manager is None:
        with _manager_lock:
            if _global_manager is None:
                _global_manager = PluginManager()
    
    return _global_manager


def set_plugin_manager(manager: PluginManager) -> None:
    """Set the global plugin manager (useful for testing)."""
    global _global_manager
    with _manager_lock:
        _global_manager = manager


def clear_global_caches() -> None:
    """Convenience function to clear all global caches."""
    manager = get_plugin_manager()
    manager.clear_all_caches()


def get_global_memory_stats() -> Dict[str, Any]:
    """Convenience function to get global memory statistics."""
    manager = get_plugin_manager()
    return manager.get_memory_stats()