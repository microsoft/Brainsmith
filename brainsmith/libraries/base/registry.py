"""
Library registry and management system.

Provides centralized registration and discovery of libraries.
"""

from typing import Dict, List, Any, Optional, Type, Tuple
import logging
from .library import BaseLibrary
from .exceptions import LibraryNotFoundError, LibraryInitializationError

logger = logging.getLogger(__name__)


class LibraryRegistry:
    """
    Registry for managing library types and instances.
    
    Provides centralized registration and discovery of available libraries.
    """
    
    def __init__(self):
        """Initialize library registry."""
        self._library_types: Dict[str, Type[BaseLibrary]] = {}
        self._instances: Dict[str, BaseLibrary] = {}
        self.logger = logging.getLogger("brainsmith.libraries.registry")
    
    def register_library_type(self, name: str, library_class: Type[BaseLibrary]):
        """
        Register a library type.
        
        Args:
            name: Library name
            library_class: Library class
        """
        self._library_types[name] = library_class
        self.logger.info(f"Registered library type: {name}")
    
    def create_library(self, name: str, config: Dict[str, Any] = None) -> BaseLibrary:
        """
        Create and initialize a library instance.
        
        Args:
            name: Library name
            config: Library configuration
            
        Returns:
            Initialized library instance
            
        Raises:
            LibraryNotFoundError: If library type not found
            LibraryInitializationError: If initialization fails
        """
        if name not in self._library_types:
            raise LibraryNotFoundError(f"Library type '{name}' not registered")
        
        library_class = self._library_types[name]
        
        try:
            instance = library_class(name)
            
            # Initialize the library
            if not instance.initialize(config or {}):
                raise LibraryInitializationError(f"Failed to initialize library '{name}'")
            
            self._instances[name] = instance
            self.logger.info(f"Created and initialized library: {name}")
            return instance
            
        except Exception as e:
            raise LibraryInitializationError(f"Error creating library '{name}': {e}")
    
    def get_library(self, name: str) -> Optional[BaseLibrary]:
        """
        Get an existing library instance.
        
        Args:
            name: Library name
            
        Returns:
            Library instance or None if not found
        """
        return self._instances.get(name)
    
    def list_available_types(self) -> List[str]:
        """
        List available library types.
        
        Returns:
            List of registered library type names
        """
        return list(self._library_types.keys())
    
    def list_instances(self) -> List[str]:
        """
        List created library instances.
        
        Returns:
            List of library instance names
        """
        return list(self._instances.keys())
    
    def cleanup_library(self, name: str):
        """
        Cleanup a library instance.
        
        Args:
            name: Library name
        """
        if name in self._instances:
            self._instances[name].cleanup()
            del self._instances[name]
            self.logger.info(f"Cleaned up library: {name}")
    
    def cleanup_all(self):
        """Cleanup all library instances."""
        for name in list(self._instances.keys()):
            self.cleanup_library(name)


class LibraryManager:
    """
    High-level library management system.
    
    Provides convenient methods for managing the four core libraries.
    """
    
    def __init__(self, registry: LibraryRegistry = None):
        """
        Initialize library manager.
        
        Args:
            registry: Library registry to use (creates new if None)
        """
        self.registry = registry or LibraryRegistry()
        self.core_libraries = ['kernels', 'transforms', 'hw_optim', 'analysis']
        self.logger = logging.getLogger("brainsmith.libraries.manager")
    
    def initialize_core_libraries(self, config: Dict[str, Any] = None) -> Dict[str, BaseLibrary]:
        """
        Initialize all core libraries.
        
        Args:
            config: Configuration for libraries
            
        Returns:
            Dictionary of initialized libraries
        """
        libraries = {}
        config = config or {}
        
        for lib_name in self.core_libraries:
            try:
                lib_config = config.get(lib_name, {})
                library = self.registry.create_library(lib_name, lib_config)
                libraries[lib_name] = library
                self.logger.info(f"Initialized core library: {lib_name}")
            except Exception as e:
                self.logger.error(f"Failed to initialize library {lib_name}: {e}")
                libraries[lib_name] = None
        
        return libraries
    
    def get_library_status(self) -> Dict[str, Any]:
        """
        Get status of all libraries.
        
        Returns:
            Status information for all libraries
        """
        status = {
            'available_types': self.registry.list_available_types(),
            'instances': self.registry.list_instances(),
            'core_libraries_status': {}
        }
        
        for lib_name in self.core_libraries:
            library = self.registry.get_library(lib_name)
            if library:
                status['core_libraries_status'][lib_name] = library.get_status()
            else:
                status['core_libraries_status'][lib_name] = {
                    'name': lib_name,
                    'initialized': False,
                    'available': False
                }
        
        return status
    
    def validate_library_parameters(self, lib_name: str, 
                                   parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for a specific library.
        
        Args:
            lib_name: Library name
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        library = self.registry.get_library(lib_name)
        if not library:
            return False, [f"Library '{lib_name}' not available"]
        
        return library.validate_parameters(parameters)
    
    def execute_library_operation(self, lib_name: str, operation: str,
                                 parameters: Dict[str, Any],
                                 context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute operation on a specific library.
        
        Args:
            lib_name: Library name
            operation: Operation to execute
            parameters: Operation parameters
            context: Execution context
            
        Returns:
            Operation results
            
        Raises:
            LibraryNotFoundError: If library not found
        """
        library = self.registry.get_library(lib_name)
        if not library:
            raise LibraryNotFoundError(f"Library '{lib_name}' not available")
        
        return library.execute(operation, parameters, context or {})
    
    def cleanup(self):
        """Cleanup all libraries."""
        self.registry.cleanup_all()
        self.logger.info("All libraries cleaned up")


# Global registry instance
_global_registry = LibraryRegistry()


def get_global_registry() -> LibraryRegistry:
    """Get the global library registry."""
    return _global_registry


def register_library(name: str, library_class: Type[BaseLibrary]):
    """Register a library type globally."""
    _global_registry.register_library_type(name, library_class)