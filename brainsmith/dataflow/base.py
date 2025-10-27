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
from .types import Shape


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


