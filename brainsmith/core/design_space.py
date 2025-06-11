"""
Essential Design Space Management for BrainSmith Core

Focused on blueprint instantiation and parameter management.
Removes complex library orchestration in favor of direct blueprint support.
"""

from typing import Dict, List, Any, Optional, Tuple
from enum import Enum
import json
import random


class ParameterType(Enum):
    """Parameter types for design space parameters."""
    CATEGORICAL = "categorical"
    INTEGER = "integer" 
    FLOAT = "float"
    CONTINUOUS = "float"  # Alias for compatibility
    BOOLEAN = "boolean"


class ParameterDefinition:
    """Definition of a design space parameter."""
    
    def __init__(self, name: str, param_type: str, 
                 values: List[Any] = None, range_min: float = None, range_max: float = None, 
                 default: Any = None):
        """Initialize parameter definition."""
        self.name = name
        self.type = param_type if isinstance(param_type, str) else param_type.value
        self.values = values
        self.range_min = range_min
        self.range_max = range_max
        self.default = default
    
    def validate_value(self, value: Any) -> bool:
        """Validate if value is valid for this parameter."""
        if self.type == 'categorical':
            return value in self.values if self.values else True
        elif self.type == 'boolean':
            return isinstance(value, bool)
        elif self.type in ['integer', 'float', 'continuous']:
            if self.range_min is not None and value < self.range_min:
                return False
            if self.range_max is not None and value > self.range_max:
                return False
            return True
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'type': self.type,
            'values': self.values,
            'range_min': self.range_min,
            'range_max': self.range_max,
            'default': self.default
        }


class DesignPoint:
    """Represents a single design configuration."""
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """Initialize design point with parameters."""
        self.parameters = parameters or {}
        self.results = {}
        self.metadata = {}
    
    def get_parameter(self, key: str, default=None):
        """Get parameter value."""
        return self.parameters.get(key, default)
    
    def set_parameter(self, key: str, value: Any):
        """Set parameter value."""
        self.parameters[key] = value
    
    def set_result(self, key: str, value: Any):
        """Set result value."""
        self.results[key] = value
    
    def get_result(self, key: str, default=None):
        """Get result value."""
        return self.results.get(key, default)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'parameters': self.parameters,
            'results': self.results,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignPoint':
        """Create from dictionary."""
        point = cls(data.get('parameters', {}))
        point.results = data.get('results', {})
        point.metadata = data.get('metadata', {})
        return point


class DesignSpace:
    """Manages design space for blueprint instantiation."""
    
    def __init__(self, name: str = "default"):
        """Initialize design space."""
        self.name = name
        self.parameters = {}  # Parameter definitions
        self.design_points = []
        self.blueprint_config = {}
    
    def add_parameter(self, param_def: ParameterDefinition):
        """Add parameter definition."""
        self.parameters[param_def.name] = param_def
    
    def get_parameter_names(self) -> List[str]:
        """Get parameter names."""
        return list(self.parameters.keys())
    
    def create_design_point(self, parameters: Dict[str, Any]) -> DesignPoint:
        """Create and validate a design point."""
        # Validate parameters
        for param_name, param_value in parameters.items():
            if param_name in self.parameters:
                param_def = self.parameters[param_name]
                if not param_def.validate_value(param_value):
                    raise ValueError(f"Invalid value {param_value} for parameter {param_name}")
        
        return DesignPoint(parameters)
    
    def sample_points(self, n_samples: int = 10, seed: Optional[int] = None) -> List[DesignPoint]:
        """Sample design points from parameter space."""
        if seed is not None:
            random.seed(seed)
        
        sampled_points = []
        
        for _ in range(n_samples):
            params = {}
            for param_name, param_def in self.parameters.items():
                if param_def.type == 'categorical' and param_def.values:
                    params[param_name] = random.choice(param_def.values)
                elif param_def.type == 'boolean':
                    params[param_name] = random.choice([True, False])
                elif param_def.type in ['integer', 'float', 'continuous']:
                    if param_def.range_min is not None and param_def.range_max is not None:
                        if param_def.type == 'integer':
                            params[param_name] = random.randint(param_def.range_min, param_def.range_max)
                        else:
                            params[param_name] = random.uniform(param_def.range_min, param_def.range_max)
                    else:
                        params[param_name] = param_def.default if param_def.default is not None else 0
                else:
                    params[param_name] = param_def.default if param_def.default is not None else None
            
            sampled_points.append(DesignPoint(params))
        
        return sampled_points
    
    @classmethod
    def from_blueprint_data(cls, blueprint_data: Dict[str, Any]) -> 'DesignSpace':
        """Create design space from blueprint configuration."""
        name = blueprint_data.get('name', 'blueprint_design_space')
        design_space = cls(name)
        design_space.blueprint_config = blueprint_data
        
        # Extract parameter definitions from blueprint
        # This would typically come from the blueprint's parameter specifications
        params_config = blueprint_data.get('parameters', {})
        for param_name, param_config in params_config.items():
            param_def = ParameterDefinition(
                name=param_name,
                param_type=param_config.get('type', 'float'),
                values=param_config.get('values'),
                range_min=param_config.get('range_min'),
                range_max=param_config.get('range_max'),
                default=param_config.get('default')
            )
            design_space.add_parameter(param_def)
        
        return design_space
    
    def validate(self) -> Tuple[bool, List[str]]:
        """Validate design space configuration."""
        errors = []
        
        # Check we have parameters
        if not self.parameters:
            errors.append("Design space has no parameters defined")
        
        # Validate parameter definitions
        for param_name, param_def in self.parameters.items():
            if param_def.type == 'categorical' and not param_def.values:
                errors.append(f"Categorical parameter {param_name} has no values")
            elif param_def.type in ['integer', 'float', 'continuous']:
                if param_def.range_min is not None and param_def.range_max is not None:
                    if param_def.range_min >= param_def.range_max:
                        errors.append(f"Parameter {param_name} has invalid range")
        
        return len(errors) == 0, errors
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'name': self.name,
            'parameters': {name: param.to_dict() for name, param in self.parameters.items()},
            'blueprint_config': self.blueprint_config,
            'design_points': [point.to_dict() for point in self.design_points]
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)


def create_parameter_sweep_points(parameters: Dict[str, List[Any]]) -> List[DesignPoint]:
    """Create design points for parameter sweep."""
    import itertools
    
    # Get all parameter combinations
    param_names = list(parameters.keys())
    param_values = list(parameters.values())
    
    design_points = []
    for combination in itertools.product(*param_values):
        params = dict(zip(param_names, combination))
        design_points.append(DesignPoint(params))
    
    return design_points


def sample_design_space(design_space: DesignSpace, n_samples: int = 10,
                       strategy: str = "random", seed: Optional[int] = None) -> List[DesignPoint]:
    """Sample points from design space (compatibility function)."""
    return design_space.sample_points(n_samples, seed)