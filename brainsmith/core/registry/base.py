"""
Base Registry Infrastructure

Provides the ComponentInfo base class and BaseRegistry abstract base class
for unified registry implementation across all Brainsmith components.
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Set, Any, TypeVar, Generic
from .exceptions import ComponentNotFoundError, ValidationError, ComponentLoadError

# Generic type for component objects
T = TypeVar('T')

logger = logging.getLogger(__name__)


class ComponentInfo(ABC):
    """Base class for all component information objects."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """The unique name of this component."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """A human-readable description of this component."""
        pass


class BaseRegistry(Generic[T], ABC):
    """Unified base class for all registry implementations."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize the registry.
        
        Args:
            search_dirs: List of directories to search for components.
                        If None, uses default directories for this registry type.
            config_manager: Optional configuration manager for component configuration.
        """
        self.search_dirs = search_dirs or self._get_default_dirs()
        self.config_manager = config_manager
        self._cache: Dict[str, T] = {}
        self._metadata_cache: Dict[str, Dict[str, Any]] = {}
        self._last_scan_time = 0
        
        logger.info(f"{self.__class__.__name__} initialized with dirs: {self.search_dirs}")
    
    # Standardized Discovery Interface
    @abstractmethod
    def discover_components(self, rescan: bool = False) -> Dict[str, T]:
        """
        Discover all available components.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping component names to component objects
        """
        pass
    
    # Standardized Retrieval Interface
    def get_component(self, name: str) -> Optional[T]:
        """
        Get a specific component by name.
        
        Args:
            name: Name of the component
            
        Returns:
            Component object or None if not found
        """
        components = self.discover_components()
        return components.get(name)
    
    def get_component_required(self, name: str) -> T:
        """
        Get a specific component by name, raising exception if not found.
        
        Args:
            name: Name of the component
            
        Returns:
            Component object
            
        Raises:
            ComponentNotFoundError: If component is not found
        """
        component = self.get_component(name)
        if component is None:
            raise ComponentNotFoundError(name, self.__class__.__name__)
        return component
    
    def list_component_names(self) -> List[str]:
        """Get list of all available component names."""
        components = self.discover_components()
        return list(components.keys())
    
    def get_component_info(self, name: str) -> Optional[Dict[str, Any]]:
        """
        Get summary information about a component.
        
        Args:
            name: Name of the component
            
        Returns:
            Dictionary with component summary or None if not found
        """
        component = self.get_component(name)
        if not component:
            return None
        return self._extract_info(component)
    
    # Standardized Search Interface
    @abstractmethod
    def find_components_by_type(self, component_type: Any) -> List[T]:
        """
        Find components by type/category.
        
        Args:
            component_type: Type or category to search for
            
        Returns:
            List of matching component objects
        """
        pass
    
    def find_components_by_attribute(self, attribute: str, value: Any) -> List[T]:
        """
        Find components by any attribute value.
        
        Args:
            attribute: Attribute name to check
            value: Value to match
            
        Returns:
            List of matching component objects
        """
        components = self.discover_components()
        matches = []
        for component in components.values():
            if hasattr(component, attribute) and getattr(component, attribute) == value:
                matches.append(component)
        return matches
    
    # Standardized Cache Management
    def refresh_cache(self):
        """Refresh the component cache."""
        self._cache.clear()
        self._metadata_cache.clear()
        self._last_scan_time = 0
        self._log_info("Registry cache refreshed")
    
    # Standardized Validation Interface
    def validate_component(self, name: str) -> tuple[bool, List[str]]:
        """
        Validate a component.
        
        Args:
            name: Name of the component to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        component = self.get_component(name)
        if not component:
            return False, [f"Component '{name}' not found"]
        return self._validate_component_implementation(component)
    
    def validate_all_components(self) -> Dict[str, tuple[bool, List[str]]]:
        """
        Validate all components in the registry.
        
        Returns:
            Dictionary mapping component names to validation results
        """
        components = self.discover_components()
        results = {}
        for name in components.keys():
            results[name] = self.validate_component(name)
        return results
    
    # Standardized Health Checking
    def health_check(self) -> Dict[str, Any]:
        """
        Perform registry health check.
        
        Returns:
            Dictionary with health check results
        """
        try:
            components = self.discover_components()
            total = len(components)
            valid_count = 0
            errors = []
            
            for name, component in components.items():
                is_valid, component_errors = self.validate_component(name)
                if is_valid:
                    valid_count += 1
                else:
                    errors.extend([f"{name}: {error}" for error in component_errors])
            
            return {
                'status': 'healthy' if len(errors) == 0 else 'degraded',
                'total_components': total,
                'component_count': total,  # Alias for backward compatibility
                'valid_components': valid_count,
                'success_rate': (valid_count / total * 100) if total > 0 else 0,
                'errors': errors,
                'registry_type': self.__class__.__name__,
                'search_dirs': self.search_dirs,
                'cache_size': len(self._cache),
                'last_scan_time': self._last_scan_time
            }
        except Exception as e:
            return {
                'status': 'failed',
                'total_components': 0,
                'valid_components': 0,
                'success_rate': 0,
                'errors': [f"Health check failed: {e}"],
                'registry_type': self.__class__.__name__,
                'search_dirs': self.search_dirs,
                'cache_size': len(self._cache)
            }
    
    # Abstract methods for registry-specific implementation
    @abstractmethod
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for this registry type."""
        pass
    
    @abstractmethod
    def _extract_info(self, component: T) -> Dict[str, Any]:
        """Extract standardized info from component."""
        pass
    
    @abstractmethod
    def _validate_component_implementation(self, component: T) -> tuple[bool, List[str]]:
        """Registry-specific validation logic."""
        pass
    
    def _log_debug(self, message: str):
        """Standardized debug logging."""
        component_logger = logging.getLogger(f"brainsmith.{self.__class__.__name__.lower()}")
        component_logger.debug(message)
    
    def _log_info(self, message: str):
        """Standardized logging."""
        component_logger = logging.getLogger(f"brainsmith.{self.__class__.__name__.lower()}")
        component_logger.info(message)
    
    def _log_warning(self, message: str):
        """Standardized warning logging."""
        component_logger = logging.getLogger(f"brainsmith.{self.__class__.__name__.lower()}")
        component_logger.warning(message)
    
    def _log_error(self, message: str):
        """Standardized error logging."""
        component_logger = logging.getLogger(f"brainsmith.{self.__class__.__name__.lower()}")
        component_logger.error(message)