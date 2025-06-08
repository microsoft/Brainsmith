"""
Base library interface for all Brainsmith libraries.

Provides the common interface that all libraries must implement
for consistent integration with the orchestrator.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaseLibrary(ABC):
    """
    Abstract base class for all Brainsmith libraries.
    
    Defines the common interface that kernels, transforms, hardware optimization,
    and analysis libraries must implement.
    """
    
    def __init__(self, name: str, version: str = "1.0.0"):
        """
        Initialize base library.
        
        Args:
            name: Library name
            version: Library version
        """
        self.name = name
        self.version = version
        self.initialized = False
        self.capabilities = set()
        self.metadata = {}
        self.logger = logging.getLogger(f"brainsmith.libraries.{name}")
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any] = None) -> bool:
        """
        Initialize the library with given configuration.
        
        Args:
            config: Library-specific configuration
            
        Returns:
            True if initialization successful
        """
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """
        Get list of capabilities provided by this library.
        
        Returns:
            List of capability names
        """
        pass
    
    @abstractmethod
    def get_design_space_parameters(self) -> Dict[str, Any]:
        """
        Get design space parameters supported by this library.
        
        Returns:
            Dictionary of parameter definitions
        """
        pass
    
    @abstractmethod
    def validate_parameters(self, parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate parameters for this library.
        
        Args:
            parameters: Parameters to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        pass
    
    @abstractmethod
    def execute(self, operation: str, parameters: Dict[str, Any], 
                context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Execute a library operation.
        
        Args:
            operation: Operation name to execute
            parameters: Operation parameters
            context: Execution context (model, previous results, etc.)
            
        Returns:
            Operation results
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get library information.
        
        Returns:
            Library information dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'initialized': self.initialized,
            'capabilities': list(self.capabilities),
            'metadata': self.metadata
        }
    
    def is_available(self) -> bool:
        """
        Check if library is available and ready to use.
        
        Returns:
            True if library is available
        """
        return self.initialized
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current library status.
        
        Returns:
            Status information
        """
        return {
            'name': self.name,
            'initialized': self.initialized,
            'capabilities_count': len(self.capabilities),
            'available': self.is_available()
        }
    
    def cleanup(self):
        """Cleanup library resources."""
        self.initialized = False
        self.logger.info(f"Library {self.name} cleaned up")


class LibraryComponent(ABC):
    """
    Base class for individual components within a library.
    
    Components are the specific implementations (kernels, transforms, etc.)
    that libraries manage.
    """
    
    def __init__(self, name: str, component_type: str):
        """
        Initialize component.
        
        Args:
            name: Component name
            component_type: Type of component
        """
        self.name = name
        self.component_type = component_type
        self.metadata = {}
        self.parameters = {}
    
    @abstractmethod
    def get_parameters(self) -> Dict[str, Any]:
        """
        Get component parameters.
        
        Returns:
            Parameter definitions
        """
        pass
    
    @abstractmethod
    def execute(self, inputs: Dict[str, Any], 
                parameters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute component operation.
        
        Args:
            inputs: Input data
            parameters: Execution parameters
            
        Returns:
            Execution results
        """
        pass
    
    def get_info(self) -> Dict[str, Any]:
        """Get component information."""
        return {
            'name': self.name,
            'type': self.component_type,
            'parameters': self.get_parameters(),
            'metadata': self.metadata
        }