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
        """Create design space from blueprint configuration with enhanced parameter extraction."""
        name = blueprint_data.get('name', 'blueprint_design_space')
        design_space = cls(name)
        design_space.blueprint_config = blueprint_data
        
        # Extract parameter definitions using enhanced logic
        parameter_definitions = design_space._extract_blueprint_parameters(blueprint_data)
        
        # Add all extracted parameters to design space
        for param_name, param_def in parameter_definitions.items():
            design_space.add_parameter(param_def)
        
        return design_space
    
    def _extract_blueprint_parameters(self, blueprint_data: Dict[str, Any]) -> Dict[str, ParameterDefinition]:
        """Extract all parameters from blueprint, handling nested sections."""
        parameter_definitions = {}
        
        if 'parameters' in blueprint_data:
            parameters = blueprint_data['parameters']
            
            for section_name, section_data in parameters.items():
                if self._is_nested_section(section_data):
                    # Handle nested sections like bert_config, folding_factors
                    for param_name, param_config in section_data.items():
                        if param_name != 'description':
                            full_param_name = f"{section_name}.{param_name}"
                            param_def = self._create_parameter_definition(full_param_name, param_config)
                            parameter_definitions[full_param_name] = param_def
                else:
                    # Handle direct parameters
                    param_def = self._create_parameter_definition(section_name, section_data)
                    parameter_definitions[section_name] = param_def
        
        return parameter_definitions
    
    def _is_nested_section(self, section_data: Any) -> bool:
        """Check if section data represents a nested parameter section."""
        return (isinstance(section_data, dict) and
                'description' in section_data and
                len(section_data) > 1)
    
    def _create_parameter_definition(self, param_name: str, param_config: Dict[str, Any]) -> ParameterDefinition:
        """Create parameter definition from blueprint parameter config."""
        if not isinstance(param_config, dict):
            # Simple value case
            return ParameterDefinition(
                name=param_name,
                param_type='categorical',
                values=[param_config],
                default=param_config
            )
        
        # Handle range-based parameters
        if 'range' in param_config:
            param_range = param_config['range']
            return ParameterDefinition(
                name=param_name,
                param_type='categorical',  # Treat ranges as categorical for discrete values
                values=param_range,
                default=param_config.get('default', param_range[0] if param_range else None)
            )
        
        # Handle values-based parameters
        elif 'values' in param_config:
            return ParameterDefinition(
                name=param_name,
                param_type='categorical',
                values=param_config['values'],
                default=param_config.get('default', param_config['values'][0] if param_config['values'] else None)
            )
        
        # Handle default-only parameters
        elif 'default' in param_config:
            default_value = param_config['default']
            param_type = 'boolean' if isinstance(default_value, bool) else \
                        'integer' if isinstance(default_value, int) else \
                        'float' if isinstance(default_value, float) else \
                        'categorical'
            
            return ParameterDefinition(
                name=param_name,
                param_type=param_type,
                values=[default_value] if param_type == 'categorical' else None,
                default=default_value
            )
        
        # Fallback case
        else:
            return ParameterDefinition(
                name=param_name,
                param_type='categorical',
                values=[None],
                default=None
            )
    
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
    
    def to_parameter_space(self) -> Dict[str, List[Any]]:
        """Convert DesignSpace to DSE ParameterSpace format."""
        parameter_space = {}
        
        for param_name, param_def in self.parameters.items():
            if param_def.type == 'categorical' and param_def.values:
                parameter_space[param_name] = param_def.values
            elif param_def.type in ['integer', 'float', 'continuous']:
                if param_def.range_min is not None and param_def.range_max is not None:
                    # Generate range values
                    if param_def.type == 'integer':
                        parameter_space[param_name] = list(range(
                            int(param_def.range_min),
                            int(param_def.range_max) + 1
                        ))
                    else:
                        # For float, use discrete steps
                        step = (param_def.range_max - param_def.range_min) / 10
                        values = []
                        current = param_def.range_min
                        while current <= param_def.range_max:
                            values.append(current)
                            current += step
                        parameter_space[param_name] = values
                else:
                    parameter_space[param_name] = [param_def.default] if param_def.default is not None else [0]
            elif param_def.type == 'boolean':
                parameter_space[param_name] = [True, False]
            else:
                parameter_space[param_name] = [param_def.default] if param_def.default is not None else [None]
        
        return parameter_space
    
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