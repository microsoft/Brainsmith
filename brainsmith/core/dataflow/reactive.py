############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################
"""
Reactive cache management with automatic dependency tracking.

Provides infrastructure for automatic cache invalidation when dependencies
change, eliminating manual cache management and reducing bugs.
"""

from typing import Dict, Any, Set, Callable, Optional, TypeVar, Generic, List
from dataclasses import dataclass, field
from weakref import WeakSet
import functools


T = TypeVar('T')


class Dependency:
    """Base class for trackable dependencies."""
    
    def __init__(self):
        """Initialize with empty dependents set."""
        self._dependents: WeakSet['ComputedProperty'] = WeakSet()
    
    def add_dependent(self, dependent: 'ComputedProperty') -> None:
        """Register a computed property that depends on this."""
        self._dependents.add(dependent)
    
    def notify_change(self) -> None:
        """Notify all dependents that this dependency has changed."""
        # Copy to avoid mutation during iteration
        dependents = list(self._dependents)
        for dependent in dependents:
            dependent.invalidate()


class ReactiveDict(Dependency):
    """Dictionary that tracks changes and notifies dependents.
    
    Usage:
        ```python
        attrs = ReactiveDict()
        computed = ComputedProperty(
            lambda: attrs.get("x", 0) * 2,
            dependencies=[attrs]
        )
        
        attrs["x"] = 5  # Automatically invalidates computed
        ```
    """
    
    def __init__(self, initial: Optional[Dict[str, Any]] = None):
        """Initialize with optional initial data."""
        super().__init__()
        self._data: Dict[str, Any] = initial.copy() if initial else {}
        self._change_listeners: List[Callable[[str, Any, Any], None]] = []
    
    def __getitem__(self, key: str) -> Any:
        """Get item value."""
        return self._data[key]
    
    def __setitem__(self, key: str, value: Any) -> None:
        """Set item and notify dependents if changed."""
        old_value = self._data.get(key)
        if old_value != value:
            self._data[key] = value
            # Notify listeners
            for listener in self._change_listeners:
                listener(key, old_value, value)
            # Notify dependents
            self.notify_change()
    
    def __delitem__(self, key: str) -> None:
        """Delete item and notify dependents."""
        if key in self._data:
            old_value = self._data[key]
            del self._data[key]
            # Notify listeners
            for listener in self._change_listeners:
                listener(key, old_value, None)
            # Notify dependents
            self.notify_change()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get with default value."""
        return self._data.get(key, default)
    
    def update(self, other: Dict[str, Any]) -> None:
        """Update multiple values and notify once."""
        changed = False
        for key, value in other.items():
            old_value = self._data.get(key)
            if old_value != value:
                self._data[key] = value
                changed = True
                # Notify listeners for each change
                for listener in self._change_listeners:
                    listener(key, old_value, value)
        
        if changed:
            self.notify_change()
    
    def add_change_listener(self, listener: Callable[[str, Any, Any], None]) -> None:
        """Add a change listener.
        
        Args:
            listener: Callable that receives (key, old_value, new_value)
        """
        self._change_listeners.append(listener)
    
    def keys(self):
        """Return dictionary keys."""
        return self._data.keys()
    
    def values(self):
        """Return dictionary values."""
        return self._data.values()
    
    def items(self):
        """Return dictionary items."""
        return self._data.items()
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists."""
        return key in self._data
    
    def __len__(self) -> int:
        """Return number of items."""
        return len(self._data)
    
    def __repr__(self) -> str:
        """String representation."""
        return f"ReactiveDict({self._data})"


@dataclass
class ComputedProperty(Generic[T], Dependency):
    """Lazy computed property with automatic dependency tracking.
    
    Usage:
        ```python
        nodeattrs = ReactiveDict({"x": 5})
        computed = ComputedProperty(
            lambda: nodeattrs["x"] * 2,
            dependencies=[nodeattrs]
        )
        
        print(computed.value)  # 10
        nodeattrs["x"] = 7     # Invalidates computed
        print(computed.value)  # 14 (recomputed)
        ```
    """
    
    compute_fn: Callable[[], T]
    dependencies: List[Dependency] = field(default_factory=list)
    _cached_value: Optional[T] = field(default=None, init=False, repr=False)
    _is_valid: bool = field(default=False, init=False, repr=False)
    
    def __post_init__(self):
        """Register with dependencies."""
        super().__init__()
        for dep in self.dependencies:
            dep.add_dependent(self)
    
    @property
    def value(self) -> T:
        """Get computed value, recomputing if invalid."""
        if not self._is_valid:
            self._cached_value = self.compute_fn()
            self._is_valid = True
        return self._cached_value
    
    def invalidate(self) -> None:
        """Invalidate this property and notify dependents."""
        if self._is_valid:
            self._is_valid = False
            self.notify_change()
    
    def is_valid(self) -> bool:
        """Check if cached value is still valid."""
        return self._is_valid


class ReactiveProperty:
    """Descriptor for reactive properties on classes.
    
    Usage:
        ```python
        class MyClass:
            def __init__(self):
                self._attrs = ReactiveDict()
                
            @ReactiveProperty
            def computed(self):
                return self._attrs.get("x", 0) * 2
        ```
    """
    
    def __init__(self, compute_fn: Callable[[Any], T]):
        """Initialize with compute function."""
        self.compute_fn = compute_fn
        self.name = compute_fn.__name__
        self.cache_attr = f"_reactive_cache_{self.name}"
    
    def __get__(self, instance: Any, owner: type) -> T:
        """Get computed value."""
        if instance is None:
            return self
        
        # Get or create cached property
        if not hasattr(instance, self.cache_attr):
            # Find dependencies by looking for ReactiveDict attributes
            deps = []
            for attr_name in dir(instance):
                attr = getattr(instance, attr_name, None)
                if isinstance(attr, (ReactiveDict, ComputedProperty)):
                    deps.append(attr)
            
            # Create computed property
            computed = ComputedProperty(
                lambda: self.compute_fn(instance),
                dependencies=deps
            )
            setattr(instance, self.cache_attr, computed)
        
        return getattr(instance, self.cache_attr).value


class ReactiveState:
    """Base class for objects with reactive state.
    
    Provides infrastructure for automatic dependency tracking and
    cache invalidation. Subclasses should define reactive properties
    using @ReactiveProperty or ComputedProperty.
    
    Example:
        ```python
        class KernelState(ReactiveState):
            def __init__(self, schema):
                super().__init__()
                self.schema = schema
                self.nodeattrs = self.create_reactive_dict()
                
                # Define computed properties
                self.resolved_config = ComputedProperty(
                    lambda: self._resolve_config(),
                    dependencies=[self.nodeattrs]
                )
                
                self.kernel_model = ComputedProperty(
                    lambda: self._create_model(),
                    dependencies=[self.resolved_config, self.tensor_context]
                )
        ```
    """
    
    def __init__(self):
        """Initialize reactive state."""
        self._reactive_dicts: List[ReactiveDict] = []
    
    def create_reactive_dict(self, initial: Optional[Dict[str, Any]] = None) -> ReactiveDict:
        """Create a reactive dictionary tracked by this state.
        
        Args:
            initial: Optional initial data
            
        Returns:
            New ReactiveDict instance
        """
        reactive = ReactiveDict(initial)
        self._reactive_dicts.append(reactive)
        return reactive
    
    def invalidate_all(self) -> None:
        """Invalidate all reactive properties."""
        # Trigger change on all reactive dicts
        for reactive_dict in self._reactive_dicts:
            reactive_dict.notify_change()


def reactive_method(dependencies: List[str]):
    """Decorator for methods that depend on reactive properties.
    
    Args:
        dependencies: List of attribute names this method depends on
        
    Example:
        ```python
        @reactive_method(["nodeattrs", "schema"])
        def compute_something(self):
            return process(self.nodeattrs, self.schema)
        ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get dependencies
            deps = []
            for dep_name in dependencies:
                dep = getattr(self, dep_name, None)
                if isinstance(dep, Dependency):
                    deps.append(dep)
            
            # Create computed property if needed
            cache_attr = f"_reactive_cache_{func.__name__}"
            if not hasattr(self, cache_attr):
                computed = ComputedProperty(
                    lambda: func(self, *args, **kwargs),
                    dependencies=deps
                )
                setattr(self, cache_attr, computed)
            
            return getattr(self, cache_attr).value
        
        return wrapper
    return decorator