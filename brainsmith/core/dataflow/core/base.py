############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Base classes for Definition/Model architecture"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Union
from .types import Shape, DataType, InterfaceDirection


class BaseDefinition(ABC):
    """Base class for all Definition classes
    
    Definitions specify constraints, relationships, and validation rules.
    They define "what should be" rather than "what is".
    """
    
    @abstractmethod
    def validate(self) -> List[str]:
        """Validate the definition for internal consistency
        
        Returns:
            List of validation errors (empty if valid)
        """
        pass
    
    @abstractmethod
    def create_model(self, **params) -> 'BaseModel':
        """Create a model instance from this definition
        
        Args:
            **params: Runtime parameters for the model
            
        Returns:
            Model instance
        """
        pass


class BaseModel(ABC):
    """Base class for all Model classes
    
    Models represent specific instantiated objects optimized for performance
    calculations. They assume valid configuration and focus on "what is".
    """
    
    def __init__(self, definition: Optional[BaseDefinition] = None):
        """Initialize model
        
        Args:
            definition: Optional reference to the definition this model implements
        """
        self._definition = definition
    
    @property
    def definition(self) -> Optional[BaseDefinition]:
        """Get the definition this model implements"""
        return self._definition
    
    @abstractmethod
    def calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate performance metrics for this model
        
        Returns:
            Dictionary of performance metrics
        """
        pass


@dataclass
class ParameterBinding:
    """Represents a binding of parameters to specific values
    
    Used to create model instances from definitions with specific parameter values.
    """
    parameters: Dict[str, Union[int, float, str]]
    constants: Dict[str, Union[int, float]] = None
    
    def __post_init__(self):
        if self.constants is None:
            self.constants = {}
    
    def get_value(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name"""
        if name in self.parameters:
            return self.parameters[name]
        if name in self.constants:
            return self.constants[name]
        return default
    
    def update(self, **kwargs) -> 'ParameterBinding':
        """Create new binding with updated parameters"""
        new_params = self.parameters.copy()
        new_params.update(kwargs)
        return ParameterBinding(new_params, self.constants.copy())


class ValidationContext:
    """Context for validation operations
    
    Provides access to interfaces, parameters, and other context needed
    for constraint validation.
    """
    
    def __init__(self):
        self.interfaces: Dict[str, Any] = {}
        self.parameters: Dict[str, Union[int, float]] = {}
        self.constants: Dict[str, Union[int, float]] = {}
        self.models: Dict[str, BaseModel] = {}
    
    def add_interface(self, name: str, interface: Any):
        """Add interface to context"""
        self.interfaces[name] = interface
    
    def add_parameter(self, name: str, value: Union[int, float]):
        """Add parameter to context"""
        self.parameters[name] = value
    
    def add_constant(self, name: str, value: Union[int, float]):
        """Add constant to context"""
        self.constants[name] = value
    
    def add_model(self, name: str, model: BaseModel):
        """Add model to context"""
        self.models[name] = model
    
    def get_interface(self, name: str) -> Any:
        """Get interface by name"""
        return self.interfaces.get(name)
    
    def get_parameter(self, name: str, default: Any = None) -> Any:
        """Get parameter value by name"""
        if name in self.parameters:
            return self.parameters[name]
        if name in self.constants:
            return self.constants[name]
        return default
    
    def get_model(self, name: str) -> Optional[BaseModel]:
        """Get model by name"""
        return self.models.get(name)
    
    def to_expression_context(self) -> Dict[str, Any]:
        """Convert to context dict for expression evaluation"""
        return {
            'interfaces': self.interfaces,
            'parameters': self.parameters,
            'constants': self.constants
        }


class ModelFactory:
    """Factory for creating model instances from definitions"""
    
    @staticmethod
    def create_interface_model(definition: 'InterfaceDefinition', 
                             tensor_dims: Shape,
                             block_dims: Union[Shape, List[Shape]],
                             stream_dims: Optional[Shape] = None,
                             **kwargs) -> 'InterfaceModel':
        """Create interface model from definition
        
        Args:
            definition: Interface definition
            tensor_dims: Actual tensor dimensions
            block_dims: Actual block dimensions
            stream_dims: Actual stream dimensions
            **kwargs: Additional model parameters
            
        Returns:
            InterfaceModel instance
        """
        # Will be implemented after we create the specific classes
        raise NotImplementedError("Will be implemented in Phase 3")
    
    @staticmethod
    def create_kernel_model(definition: 'KernelDefinition',
                          interface_models: List['InterfaceModel'],
                          parameter_binding: ParameterBinding,
                          **kwargs) -> 'KernelModel':
        """Create kernel model from definition
        
        Args:
            definition: Kernel definition
            interface_models: List of interface models
            parameter_binding: Parameter values
            **kwargs: Additional model parameters
            
        Returns:
            KernelModel instance
        """
        # Will be implemented after we create the specific classes
        raise NotImplementedError("Will be implemented in Phase 3")


class DefinitionRegistry:
    """Registry for storing and retrieving definitions
    
    Enables reuse of definitions across multiple model instances.
    """
    
    def __init__(self):
        self._interface_definitions: Dict[str, 'InterfaceDefinition'] = {}
        self._kernel_definitions: Dict[str, 'KernelDefinition'] = {}
    
    def register_interface_definition(self, name: str, definition: 'InterfaceDefinition'):
        """Register an interface definition"""
        self._interface_definitions[name] = definition
    
    def register_kernel_definition(self, name: str, definition: 'KernelDefinition'):
        """Register a kernel definition"""
        self._kernel_definitions[name] = definition
    
    def get_interface_definition(self, name: str) -> Optional['InterfaceDefinition']:
        """Get interface definition by name"""
        return self._interface_definitions.get(name)
    
    def get_kernel_definition(self, name: str) -> Optional['KernelDefinition']:
        """Get kernel definition by name"""
        return self._kernel_definitions.get(name)
    
    def list_interface_definitions(self) -> List[str]:
        """List all registered interface definition names"""
        return list(self._interface_definitions.keys())
    
    def list_kernel_definitions(self) -> List[str]:
        """List all registered kernel definition names"""
        return list(self._kernel_definitions.keys())


# Global registry instance
DEFINITION_REGISTRY = DefinitionRegistry()