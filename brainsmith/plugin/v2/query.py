"""
High-Performance Indexed Query Engine

Provides O(log n) query performance vs the current O(n) linear scans.
This is a BREAKING CHANGE in query API and semantics.
"""

import logging
from typing import Dict, List, Any, Optional, Set, Union
from collections import defaultdict
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


@dataclass
class PluginQuery:
    """
    Structured query for plugins.
    
    BREAKING CHANGE: Replaces the old **kwargs filter approach.
    """
    type: Optional[str] = None
    name: Optional[str] = None
    stage: Optional[str] = None
    target_kernel: Optional[str] = None
    framework: Optional[str] = None
    version: Optional[str] = None
    author: Optional[str] = None
    tags: Optional[List[str]] = None
    custom_filters: Optional[Dict[str, Any]] = None
    
    def is_simple(self) -> bool:
        """Check if this is a simple single-field query"""
        field_count = sum(1 for field in [self.type, self.name, self.stage, 
                                         self.target_kernel, self.framework] 
                         if field is not None)
        return field_count == 1 and not self.tags and not self.custom_filters
    
    def get_primary_field(self) -> tuple:
        """Get the primary field for simple queries"""
        if self.type:
            return ('type', self.type)
        elif self.name:
            return ('name', self.name)
        elif self.stage:
            return ('stage', self.stage)
        elif self.target_kernel:
            return ('target_kernel', self.target_kernel)
        elif self.framework:
            return ('framework', self.framework)
        else:
            return None


class BTreeIndex:
    """
    Simple B-tree style index for sorted data.
    Provides O(log n) lookups for ordered fields.
    """
    
    def __init__(self):
        self._data = {}  # value -> set of plugin_ids
        self._sorted_keys = []
        self._lock = threading.RWLock()
    
    def add(self, value: str, plugin_id: str):
        """Add a value->plugin_id mapping"""
        with self._lock.writer():
            if value not in self._data:
                self._data[value] = set()
                self._sorted_keys.append(value)
                self._sorted_keys.sort()
            self._data[value].add(plugin_id)
    
    def remove(self, value: str, plugin_id: str):
        """Remove a value->plugin_id mapping"""
        with self._lock.writer():
            if value in self._data:
                self._data[value].discard(plugin_id)
                if not self._data[value]:
                    del self._data[value]
                    self._sorted_keys.remove(value)
    
    def lookup(self, value: str) -> Set[str]:
        """O(log n) lookup of plugin_ids for value"""
        with self._lock.reader():
            return self._data.get(value, set()).copy()
    
    def range_lookup(self, start: str = None, end: str = None) -> Set[str]:
        """Range lookup between start and end values"""
        with self._lock.reader():
            result = set()
            for key in self._sorted_keys:
                if start and key < start:
                    continue
                if end and key > end:
                    break
                result.update(self._data[key])
            return result


class HashIndex:
    """
    Hash-based index for unordered fields.
    Provides O(1) lookups.
    """
    
    def __init__(self):
        self._data = defaultdict(set)  # value -> set of plugin_ids
        self._lock = threading.RWLock()
    
    def add(self, value: str, plugin_id: str):
        """Add a value->plugin_id mapping"""
        with self._lock.writer():
            self._data[value].add(plugin_id)
    
    def remove(self, value: str, plugin_id: str):
        """Remove a value->plugin_id mapping"""
        with self._lock.writer():
            self._data[value].discard(plugin_id)
            if not self._data[value]:
                del self._data[value]
    
    def lookup(self, value: str) -> Set[str]:
        """O(1) lookup of plugin_ids for value"""
        with self._lock.reader():
            return self._data[value].copy()
    
    def get_all_values(self) -> List[str]:
        """Get all indexed values"""
        with self._lock.reader():
            return list(self._data.keys())


class MultiValueIndex:
    """
    Index for fields that can have multiple values (like tags).
    """
    
    def __init__(self):
        self._data = defaultdict(set)  # value -> set of plugin_ids
        self._lock = threading.RWLock()
    
    def add(self, values: List[str], plugin_id: str):
        """Add multiple values for a plugin"""
        with self._lock.writer():
            for value in values:
                self._data[value].add(plugin_id)
    
    def remove(self, values: List[str], plugin_id: str):
        """Remove multiple values for a plugin"""
        with self._lock.writer():
            for value in values:
                self._data[value].discard(plugin_id)
                if not self._data[value]:
                    del self._data[value]
    
    def lookup_any(self, values: List[str]) -> Set[str]:
        """Find plugins that have any of the specified values"""
        with self._lock.reader():
            result = set()
            for value in values:
                result.update(self._data[value])
            return result
    
    def lookup_all(self, values: List[str]) -> Set[str]:
        """Find plugins that have all of the specified values"""
        with self._lock.reader():
            if not values:
                return set()
            
            result = self._data[values[0]].copy()
            for value in values[1:]:
                result.intersection_update(self._data[value])
            return result


class QueryEngine:
    """
    High-performance query engine with indexed lookups.
    
    BREAKING CHANGE: Completely new query API replacing old registry.query()
    """
    
    def __init__(self):
        # Create indices for common query fields
        self._type_index = HashIndex()
        self._name_index = HashIndex()
        self._stage_index = HashIndex()
        self._target_kernel_index = HashIndex()
        self._framework_index = HashIndex()
        self._author_index = HashIndex()
        self._version_index = BTreeIndex()  # Sorted for version comparisons
        self._tags_index = MultiValueIndex()
        
        # Keep reference to plugin storage
        self._plugin_storage = {}
        self._lock = threading.RWLock()
    
    def index_plugin(self, plugin_id: str, plugin_spec: 'PluginSpec'):
        """Add plugin to all relevant indices"""
        with self._lock.writer():
            # Index basic fields
            if plugin_spec.type:
                self._type_index.add(plugin_spec.type, plugin_id)
            if plugin_spec.name:
                self._name_index.add(plugin_spec.name, plugin_id)
            if plugin_spec.stage:
                self._stage_index.add(plugin_spec.stage, plugin_id)
            if plugin_spec.target_kernel:
                self._target_kernel_index.add(plugin_spec.target_kernel, plugin_id)
            if plugin_spec.framework:
                self._framework_index.add(plugin_spec.framework, plugin_id)
            if plugin_spec.author:
                self._author_index.add(plugin_spec.author, plugin_id)
            if plugin_spec.version:
                self._version_index.add(plugin_spec.version, plugin_id)
            if plugin_spec.tags:
                self._tags_index.add(plugin_spec.tags, plugin_id)
            
            # Store plugin data
            self._plugin_storage[plugin_id] = plugin_spec
    
    def unindex_plugin(self, plugin_id: str):
        """Remove plugin from all indices"""
        with self._lock.writer():
            if plugin_id not in self._plugin_storage:
                return
            
            plugin_spec = self._plugin_storage[plugin_id]
            
            # Remove from indices
            if plugin_spec.type:
                self._type_index.remove(plugin_spec.type, plugin_id)
            if plugin_spec.name:
                self._name_index.remove(plugin_spec.name, plugin_id)
            if plugin_spec.stage:
                self._stage_index.remove(plugin_spec.stage, plugin_id)
            if plugin_spec.target_kernel:
                self._target_kernel_index.remove(plugin_spec.target_kernel, plugin_id)
            if plugin_spec.framework:
                self._framework_index.remove(plugin_spec.framework, plugin_id)
            if plugin_spec.author:
                self._author_index.remove(plugin_spec.author, plugin_id)
            if plugin_spec.version:
                self._version_index.remove(plugin_spec.version, plugin_id)
            if plugin_spec.tags:
                self._tags_index.remove(plugin_spec.tags, plugin_id)
            
            # Remove from storage
            del self._plugin_storage[plugin_id]
    
    def query(self, query: PluginQuery) -> List['PluginSpec']:
        """
        Execute a query and return matching plugins.
        
        BREAKING CHANGE: New query API with structured PluginQuery objects.
        """
        with self._lock.reader():
            # Handle simple queries with direct index lookup
            if query.is_simple():
                field, value = query.get_primary_field()
                plugin_ids = self._get_index_for_field(field).lookup(value)
                return [self._plugin_storage[pid] for pid in plugin_ids 
                       if pid in self._plugin_storage]
            
            # Handle compound queries
            return self._execute_compound_query(query)
    
    def _get_index_for_field(self, field: str):
        """Get the appropriate index for a field"""
        index_map = {
            'type': self._type_index,
            'name': self._name_index,
            'stage': self._stage_index,
            'target_kernel': self._target_kernel_index,
            'framework': self._framework_index,
            'author': self._author_index,
            'version': self._version_index,
        }
        return index_map.get(field, self._name_index)  # Default fallback
    
    def _execute_compound_query(self, query: PluginQuery) -> List['PluginSpec']:
        """Execute a compound query using index intersection"""
        candidate_sets = []
        
        # Get candidate sets from each filter
        if query.type:
            candidate_sets.append(self._type_index.lookup(query.type))
        if query.name:
            candidate_sets.append(self._name_index.lookup(query.name))
        if query.stage:
            candidate_sets.append(self._stage_index.lookup(query.stage))
        if query.target_kernel:
            candidate_sets.append(self._target_kernel_index.lookup(query.target_kernel))
        if query.framework:
            candidate_sets.append(self._framework_index.lookup(query.framework))
        if query.author:
            candidate_sets.append(self._author_index.lookup(query.author))
        if query.version:
            candidate_sets.append(self._version_index.lookup(query.version))
        if query.tags:
            candidate_sets.append(self._tags_index.lookup_all(query.tags))
        
        # If no filters, return all plugins
        if not candidate_sets:
            return list(self._plugin_storage.values())
        
        # Intersect all candidate sets
        result_ids = candidate_sets[0]
        for candidate_set in candidate_sets[1:]:
            result_ids = result_ids.intersection(candidate_set)
        
        # Apply custom filters if any
        if query.custom_filters:
            result_ids = self._apply_custom_filters(result_ids, query.custom_filters)
        
        return [self._plugin_storage[pid] for pid in result_ids 
               if pid in self._plugin_storage]
    
    def _apply_custom_filters(self, plugin_ids: Set[str], 
                            custom_filters: Dict[str, Any]) -> Set[str]:
        """Apply custom filters that can't use indices"""
        filtered_ids = set()
        
        for plugin_id in plugin_ids:
            plugin_spec = self._plugin_storage.get(plugin_id)
            if not plugin_spec:
                continue
            
            # Check custom filters
            matches = True
            for field, value in custom_filters.items():
                plugin_value = getattr(plugin_spec, field, None)
                if plugin_value != value:
                    matches = False
                    break
            
            if matches:
                filtered_ids.add(plugin_id)
        
        return filtered_ids
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get query engine statistics"""
        with self._lock.reader():
            return {
                'total_plugins': len(self._plugin_storage),
                'indices': {
                    'type_values': len(self._type_index.get_all_values()),
                    'name_values': len(self._name_index.get_all_values()),
                    'stage_values': len(self._stage_index.get_all_values()),
                    'framework_values': len(self._framework_index.get_all_values()),
                }
            }


# Add threading RWLock if not available
if not hasattr(threading, 'RWLock'):
    class RWLock:
        """Simple Read-Write lock implementation"""
        
        def __init__(self):
            self._readers = 0
            self._writers = 0
            self._read_ready = threading.Condition(threading.RLock())
            self._write_ready = threading.Condition(threading.RLock())
        
        def reader(self):
            return _ReaderContext(self)
        
        def writer(self):
            return _WriterContext(self)
    
    class _ReaderContext:
        def __init__(self, lock):
            self.lock = lock
        
        def __enter__(self):
            with self.lock._read_ready:
                while self.lock._writers > 0:
                    self.lock._read_ready.wait()
                self.lock._readers += 1
        
        def __exit__(self, *args):
            with self.lock._read_ready:
                self.lock._readers -= 1
                if self.lock._readers == 0:
                    self.lock._read_ready.notify_all()
    
    class _WriterContext:
        def __init__(self, lock):
            self.lock = lock
        
        def __enter__(self):
            with self.lock._write_ready:
                while self.lock._readers > 0 or self.lock._writers > 0:
                    self.lock._write_ready.wait()
                self.lock._writers += 1
        
        def __exit__(self, *args):
            with self.lock._write_ready:
                self.lock._writers -= 1
                self.lock._write_ready.notify_all()
    
    threading.RWLock = RWLock