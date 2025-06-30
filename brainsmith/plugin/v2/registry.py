"""
Thread-Safe Plugin Registry V2

Complete redesign with proper thread safety, plugin lifecycle management,
and O(log n) query performance. This is a BREAKING CHANGE.
"""

import logging
import threading
import uuid
from typing import Dict, List, Any, Optional, Type, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime

from .query import QueryEngine, PluginQuery
from .contracts import PluginContract, ValidationResult, validate_contract_compliance

logger = logging.getLogger(__name__)


class PluginState(Enum):
    """Plugin lifecycle states"""
    DISCOVERED = "discovered"      # Found but not loaded
    LOADING = "loading"           # Currently being loaded
    LOADED = "loaded"             # Loaded but not validated
    VALIDATING = "validating"     # Running validation checks
    ACTIVE = "active"             # Ready for use
    FAILED = "failed"             # Failed validation or execution
    DISABLED = "disabled"         # Manually disabled
    UNLOADING = "unloading"       # Being removed


@dataclass
class PluginSpec:
    """
    Complete plugin specification.
    
    BREAKING CHANGE: Replaces old metadata dictionary approach.
    """
    plugin_id: str
    name: str
    type: str  # "transform", "kernel", "backend" (no more kernel_inference confusion)
    plugin_class: Type[PluginContract]
    
    # Classification metadata
    stage: Optional[str] = None           # For transforms: cleanup, topology_opt, etc.
    target_kernel: Optional[str] = None   # For kernel inference transforms
    backend_type: Optional[str] = None    # For backends: hls, rtl
    
    # Descriptive metadata
    version: str = "0.0.0"
    author: Optional[str] = None
    description: Optional[str] = None
    framework: str = "brainsmith"
    tags: List[str] = field(default_factory=list)
    
    # Technical metadata
    dependencies: List[Any] = field(default_factory=list)
    config_schema: Optional[Any] = None
    
    # Lifecycle metadata
    state: PluginState = PluginState.DISCOVERED
    created_at: datetime = field(default_factory=datetime.now)
    last_error: Optional[str] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def key(self) -> str:
        """Get unique registry key for this plugin"""
        return f"{self.type}:{self.name}"
    
    def is_kernel_inference(self) -> bool:
        """Check if this is a kernel inference transform"""
        return self.type == "transform" and self.target_kernel is not None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for backward compatibility queries)"""
        result = {
            'plugin_id': self.plugin_id,
            'name': self.name,
            'type': self.type,
            'class': self.plugin_class,
            'stage': self.stage,
            'target_kernel': self.target_kernel,
            'backend_type': self.backend_type,
            'version': self.version,
            'author': self.author,
            'description': self.description,
            'framework': self.framework,
            'tags': self.tags.copy(),
            'dependencies': self.dependencies.copy(),
            'state': self.state.value,
            'created_at': self.created_at,
            'last_error': self.last_error,
        }
        result.update(self.custom_metadata)
        return result


class RegistrationResult:
    """Result of plugin registration"""
    
    def __init__(self, success: bool, plugin_id: str = None, 
                 errors: List[str] = None, warnings: List[str] = None):
        self.success = success
        self.plugin_id = plugin_id
        self.errors = errors or []
        self.warnings = warnings or []
    
    @classmethod
    def succeeded(cls, plugin_id: str, warnings: List[str] = None) -> 'RegistrationResult':
        return cls(True, plugin_id, warnings=warnings)
    
    @classmethod
    def failed(cls, errors: List[str], warnings: List[str] = None) -> 'RegistrationResult':
        return cls(False, errors=errors, warnings=warnings)


class StateTransitionResult:
    """Result of plugin state transition"""
    
    def __init__(self, success: bool, from_state: PluginState, 
                 to_state: PluginState, error: str = None):
        self.success = success
        self.from_state = from_state
        self.to_state = to_state
        self.error = error


class PluginRegistry:
    """
    Thread-safe plugin registry with lifecycle management.
    
    BREAKING CHANGE: Completely new API replacing UnifiedRegistry.
    """
    
    def __init__(self):
        self._plugins: Dict[str, PluginSpec] = {}  # plugin_id -> PluginSpec
        self._name_to_id: Dict[str, str] = {}      # name -> plugin_id (for lookups)
        self._query_engine = QueryEngine()
        self._lock = threading.RWLock()
        self._state_lock = threading.Lock()
        
        logger.info("PluginRegistry V2 initialized")
    
    def register(self, plugin_spec: PluginSpec) -> RegistrationResult:
        """
        Register a plugin with validation.
        
        BREAKING CHANGE: Takes PluginSpec instead of individual parameters.
        """
        with self._lock.writer():
            try:
                # 1. Validate plugin specification
                validation_result = self._validate_plugin_spec(plugin_spec)
                if not validation_result.is_valid:
                    return RegistrationResult.failed(validation_result.errors, 
                                                   validation_result.warnings)
                
                # 2. Check for conflicts
                conflict_result = self._check_conflicts(plugin_spec)
                if not conflict_result.is_valid:
                    logger.warning(f"Plugin conflicts detected for {plugin_spec.name}: "
                                 f"{conflict_result.warnings}")
                
                # 3. Generate unique plugin ID
                plugin_spec.plugin_id = str(uuid.uuid4())
                
                # 4. Store plugin
                self._plugins[plugin_spec.plugin_id] = plugin_spec
                self._name_to_id[plugin_spec.key] = plugin_spec.plugin_id
                
                # 5. Index for queries
                self._query_engine.index_plugin(plugin_spec.plugin_id, plugin_spec)
                
                # 6. Transition to loaded state
                self._transition_state(plugin_spec.plugin_id, 
                                     PluginState.DISCOVERED, PluginState.LOADED)
                
                logger.info(f"Registered plugin: {plugin_spec.key} (id: {plugin_spec.plugin_id})")
                
                warnings = validation_result.warnings + conflict_result.warnings
                return RegistrationResult.succeeded(plugin_spec.plugin_id, warnings)
                
            except Exception as e:
                logger.error(f"Failed to register plugin {plugin_spec.name}: {e}")
                return RegistrationResult.failed([str(e)])
    
    def unregister(self, plugin_id: str) -> bool:
        """
        Unregister a plugin.
        
        BREAKING CHANGE: Uses plugin_id instead of name.
        """
        with self._lock.writer():
            if plugin_id not in self._plugins:
                return False
            
            plugin_spec = self._plugins[plugin_id]
            
            # Transition to unloading state
            self._transition_state(plugin_id, plugin_spec.state, PluginState.UNLOADING)
            
            # Remove from indices
            self._query_engine.unindex_plugin(plugin_id)
            
            # Remove from storage
            del self._name_to_id[plugin_spec.key]
            del self._plugins[plugin_id]
            
            logger.info(f"Unregistered plugin: {plugin_spec.key}")
            return True
    
    def get(self, plugin_type: str, name: str) -> Optional[Type[PluginContract]]:
        """
        Get plugin class by type and name.
        
        Maintains compatibility with old API for easier migration.
        """
        key = f"{plugin_type}:{name}"
        
        with self._lock.reader():
            plugin_id = self._name_to_id.get(key)
            if not plugin_id:
                return None
            
            plugin_spec = self._plugins.get(plugin_id)
            if not plugin_spec or plugin_spec.state != PluginState.ACTIVE:
                return None
            
            return plugin_spec.plugin_class
    
    def get_plugin_spec(self, plugin_id: str) -> Optional[PluginSpec]:
        """Get complete plugin specification by ID"""
        with self._lock.reader():
            return self._plugins.get(plugin_id)
    
    def query(self, query: PluginQuery = None, **legacy_filters) -> List[PluginSpec]:
        """
        Query plugins with new structured query or legacy filters.
        
        BREAKING CHANGE: Returns PluginSpec objects instead of dictionaries.
        """
        if query is None:
            # Handle legacy **kwargs queries for backward compatibility
            query = PluginQuery(
                type=legacy_filters.get('type'),
                name=legacy_filters.get('name'),
                stage=legacy_filters.get('stage'),
                target_kernel=legacy_filters.get('target_kernel'),
                framework=legacy_filters.get('framework'),
                version=legacy_filters.get('version'),
                author=legacy_filters.get('author'),
                custom_filters={k: v for k, v in legacy_filters.items()
                              if k not in ['type', 'name', 'stage', 'target_kernel', 
                                         'framework', 'version', 'author']}
            )
        
        with self._lock.reader():
            return self._query_engine.query(query)
    
    def query_legacy(self, **filters) -> List[Dict[str, Any]]:
        """
        Legacy query method that returns dictionaries.
        
        Provided for backward compatibility during migration.
        """
        plugin_specs = self.query(**filters)
        return [spec.to_dict() for spec in plugin_specs]
    
    def activate_plugin(self, plugin_id: str) -> StateTransitionResult:
        """Activate a loaded plugin"""
        with self._state_lock:
            plugin_spec = self._plugins.get(plugin_id)
            if not plugin_spec:
                return StateTransitionResult(False, None, None, "Plugin not found")
            
            if plugin_spec.state != PluginState.LOADED:
                return StateTransitionResult(False, plugin_spec.state, PluginState.ACTIVE,
                                           f"Cannot activate from state {plugin_spec.state}")
            
            # Validate plugin before activation
            try:
                validation_result = self._validate_plugin_instance(plugin_spec)
                if not validation_result.is_valid:
                    self._transition_state(plugin_id, plugin_spec.state, PluginState.FAILED)
                    return StateTransitionResult(False, plugin_spec.state, PluginState.FAILED,
                                               f"Validation failed: {validation_result.errors}")
                
                return self._transition_state(plugin_id, plugin_spec.state, PluginState.ACTIVE)
                
            except Exception as e:
                self._transition_state(plugin_id, plugin_spec.state, PluginState.FAILED)
                return StateTransitionResult(False, plugin_spec.state, PluginState.FAILED, str(e))
    
    def disable_plugin(self, plugin_id: str) -> StateTransitionResult:
        """Disable an active plugin"""
        with self._state_lock:
            plugin_spec = self._plugins.get(plugin_id)
            if not plugin_spec:
                return StateTransitionResult(False, None, None, "Plugin not found")
            
            if plugin_spec.state not in [PluginState.ACTIVE, PluginState.FAILED]:
                return StateTransitionResult(False, plugin_spec.state, PluginState.DISABLED,
                                           f"Cannot disable from state {plugin_spec.state}")
            
            return self._transition_state(plugin_id, plugin_spec.state, PluginState.DISABLED)
    
    def get_plugins_by_state(self, state: PluginState) -> List[PluginSpec]:
        """Get all plugins in a specific state"""
        with self._lock.reader():
            return [spec for spec in self._plugins.values() if spec.state == state]
    
    def get_active_plugins(self) -> List[PluginSpec]:
        """Get all active plugins"""
        return self.get_plugins_by_state(PluginState.ACTIVE)
    
    def get_failed_plugins(self) -> List[PluginSpec]:
        """Get all failed plugins"""
        return self.get_plugins_by_state(PluginState.FAILED)
    
    def clear(self):
        """Clear all plugins (for testing)"""
        with self._lock.writer():
            self._plugins.clear()
            self._name_to_id.clear()
            # Note: We can't easily clear the query engine without rebuilding it
            self._query_engine = QueryEngine()
            logger.info("Registry cleared")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get registry statistics"""
        with self._lock.reader():
            state_counts = {}
            for state in PluginState:
                state_counts[state.value] = len(self.get_plugins_by_state(state))
            
            return {
                'total_plugins': len(self._plugins),
                'plugins_by_state': state_counts,
                'query_engine_stats': self._query_engine.get_statistics()
            }
    
    def _validate_plugin_spec(self, plugin_spec: PluginSpec) -> ValidationResult:
        """Validate plugin specification"""
        errors = []
        warnings = []
        
        # Validate required fields
        if not plugin_spec.name:
            errors.append("Plugin name is required")
        if not plugin_spec.type:
            errors.append("Plugin type is required")
        if not plugin_spec.plugin_class:
            errors.append("Plugin class is required")
        
        # Validate plugin type
        valid_types = ["transform", "kernel", "backend"]
        if plugin_spec.type not in valid_types:
            errors.append(f"Invalid plugin type '{plugin_spec.type}'. Must be one of: {valid_types}")
        
        # Validate stage/target_kernel for transforms
        if plugin_spec.type == "transform":
            if not plugin_spec.stage and not plugin_spec.target_kernel:
                errors.append("Transform must specify either 'stage' or 'target_kernel'")
            elif plugin_spec.stage and plugin_spec.target_kernel:
                errors.append("Transform cannot specify both 'stage' and 'target_kernel'")
        
        # Validate backend requirements
        if plugin_spec.type == "backend":
            if not plugin_spec.target_kernel:
                errors.append("Backend must specify 'target_kernel'")
            if not plugin_spec.backend_type:
                errors.append("Backend must specify 'backend_type'")
            elif plugin_spec.backend_type not in ["hls", "rtl"]:
                errors.append(f"Invalid backend_type '{plugin_spec.backend_type}'. Must be 'hls' or 'rtl'")
        
        # Validate contract compliance
        try:
            contract_validation = validate_contract_compliance(plugin_spec.plugin_class, PluginContract)
            errors.extend(contract_validation.errors)
            warnings.extend(contract_validation.warnings)
        except Exception as e:
            errors.append(f"Contract validation failed: {e}")
        
        return ValidationResult(len(errors) == 0, errors, warnings)
    
    def _check_conflicts(self, plugin_spec: PluginSpec) -> ValidationResult:
        """Check for naming conflicts"""
        warnings = []
        
        # Check if name already exists
        if plugin_spec.key in self._name_to_id:
            existing_id = self._name_to_id[plugin_spec.key]
            existing_spec = self._plugins[existing_id]
            warnings.append(f"Plugin {plugin_spec.key} already exists (id: {existing_id}). "
                          f"Will overwrite existing registration from {existing_spec.framework}")
        
        return ValidationResult(True, [], warnings)
    
    def _validate_plugin_instance(self, plugin_spec: PluginSpec) -> ValidationResult:
        """Validate that plugin instance works correctly"""
        try:
            # Try to create an instance
            instance = plugin_spec.plugin_class()
            
            # Call contract methods to ensure they work
            instance.validate_environment()
            instance.get_dependencies()
            instance.get_metadata()
            
            return ValidationResult.success()
            
        except Exception as e:
            return ValidationResult.failure([f"Plugin instance validation failed: {e}"])
    
    def _transition_state(self, plugin_id: str, from_state: PluginState, 
                         to_state: PluginState) -> StateTransitionResult:
        """Transition plugin state"""
        plugin_spec = self._plugins.get(plugin_id)
        if not plugin_spec:
            return StateTransitionResult(False, from_state, to_state, "Plugin not found")
        
        if plugin_spec.state != from_state:
            return StateTransitionResult(False, plugin_spec.state, to_state,
                                       f"Expected state {from_state}, got {plugin_spec.state}")
        
        plugin_spec.state = to_state
        logger.debug(f"Plugin {plugin_spec.name} transitioned: {from_state} -> {to_state}")
        
        return StateTransitionResult(True, from_state, to_state)


# Global registry instance
_registry_instance = None
_registry_lock = threading.Lock()


def get_registry() -> PluginRegistry:
    """
    Get the global plugin registry instance.
    
    BREAKING CHANGE: Returns PluginRegistry instead of UnifiedRegistry.
    """
    global _registry_instance
    
    if _registry_instance is None:
        with _registry_lock:
            if _registry_instance is None:
                _registry_instance = PluginRegistry()
    
    return _registry_instance


def reset_registry():
    """Reset global registry (for testing)"""
    global _registry_instance
    with _registry_lock:
        _registry_instance = None