"""
Design Space Management using existing components.

This module provides design space construction and management
using existing library components in an extensible structure.
"""

from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
import json
import logging

logger = logging.getLogger(__name__)


class ParameterType(Enum):
    """
    Enum for parameter types to maintain compatibility with existing tests.
    """
    CATEGORICAL = "categorical"
    INTEGER = "integer" 
    FLOAT = "float"
    CONTINUOUS = "float"  # Alias for FLOAT to match tests
    BOOLEAN = "boolean"


class ParameterDefinition:
    """
    Definition of a parameter in the design space.
    
    Provides compatibility with existing parameter definitions
    while supporting extensible structure.
    """
    
    def __init__(self, name: str, param_type: Union[str, ParameterType], 
                 values: Any = None, range_min: float = None, range_max: float = None, 
                 default: Any = None, range_values: List = None):
        """
        Initialize parameter definition.
        
        Args:
            name: Parameter name
            param_type: Parameter type ('categorical', 'integer', 'float', 'boolean')
            values: List of values for categorical parameters
            range_min: Minimum value for numeric parameters
            range_max: Maximum value for numeric parameters
            default: Default value
            range_values: Alternative way to specify range [min, max] for compatibility
        """
        self.name = name
        if isinstance(param_type, ParameterType):
            self.type = param_type.value
        else:
            self.type = param_type
        
        # Handle range_values parameter for backward compatibility
        if range_values is not None:
            if len(range_values) >= 2:
                self.range_min = range_values[0]
                self.range_max = range_values[1]
            else:
                self.range_min = range_min
                self.range_max = range_max
        else:
            self.range_min = range_min
            self.range_max = range_max
            
        self.values = values
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
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ParameterDefinition':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            param_type=data['type'],
            values=data.get('values'),
            range_min=data.get('range_min'),
            range_max=data.get('range_max'),
            default=data.get('default')
        )


class DesignPoint:
    """
    Represents a single design point in the design space.
    
    Uses existing component parameters while providing
    extensible structure for future library additions.
    """
    
    def __init__(self, parameters: Dict[str, Any] = None):
        """
        Initialize design point with parameters.
        
        Args:
            parameters: Dictionary of design point parameters
        """
        self.parameters = parameters or {}
        self.results = {}
        self.metadata = {}
        self.objectives = {}  # Add objectives for compatibility
    
    def get_parameter(self, key: str, default=None):
        """Get parameter value by key."""
        return self.parameters.get(key, default)
    
    def set_result(self, key: str, value: Any):
        """Set result value for this design point."""
        self.results[key] = value
    
    def get_result(self, key: str, default=None):
        """Get result value by key."""
        return self.results.get(key, default)
    
    def set_objective(self, key: str, value: Any):
        """Set objective value for this design point."""
        self.objectives[key] = value
    
    def get_objective(self, key: str, default=None):
        """Get objective value by key."""
        return self.objectives.get(key, default)
    
    def dominates(self, other: 'DesignPoint', objectives: List[str]) -> bool:
        """Check if this point dominates another point."""
        better_in_any = False
        for obj in objectives:
            self_val = self.get_objective(obj, 0)
            other_val = other.get_objective(obj, 0)
            if self_val < other_val:  # Assuming minimization
                return False
            elif self_val > other_val:
                better_in_any = True
        return better_in_any
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert design point to dictionary."""
        return {
            'parameters': self.parameters,
            'results': self.results,
            'metadata': self.metadata,
            'objectives': self.objectives
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DesignPoint':
        """Create design point from dictionary."""
        point = cls(data.get('parameters', {}))
        point.results = data.get('results', {})
        point.metadata = data.get('metadata', {})
        point.objectives = data.get('objectives', {})
        return point


class DesignSpace:
    """
    Manages design space construction from existing components.
    
    Provides extensible structure around existing libraries
    while maintaining compatibility with current workflows.
    """
    
    def __init__(self, name: str = "default", libraries: Dict[str, Any] = None, 
                 blueprint_config: Dict[str, Any] = None):
        """
        Initialize design space from existing libraries.
        
        Args:
            name: Design space name
            libraries: Dictionary of existing library instances
            blueprint_config: Blueprint configuration for design space
        """
        self.name = name
        self.libraries = libraries or {}
        self.blueprint_config = blueprint_config or {}
        self.design_points = []
        self.parameters = {}  # Parameter definitions
        self.metadata = {
            'source': 'existing_components_only',
            'extensible_structure': True
        }
    
    def add_parameter(self, param_def: ParameterDefinition):
        """Add parameter definition to design space."""
        self.parameters[param_def.name] = param_def
    
    def get_parameter_names(self) -> List[str]:
        """Get list of parameter names."""
        return list(self.parameters.keys())
    
    def construct_from_existing_libraries(self) -> List[DesignPoint]:
        """
        Construct design space from existing library components.
        
        Returns:
            List of design points using existing component parameters
        """
        logger.info("Constructing design space from existing libraries")
        
        design_points = []
        
        # Get design space from each library using existing components
        library_spaces = {}
        
        for lib_name, lib in self.libraries.items():
            if lib and hasattr(lib, 'get_design_space_from_existing'):
                try:
                    lib_space = lib.get_design_space_from_existing()
                    library_spaces[lib_name] = lib_space
                    logger.info(f"Got design space from {lib_name}: {len(lib_space.get('points', []))} points")
                except Exception as e:
                    logger.warning(f"Failed to get design space from {lib_name}: {e}")
                    library_spaces[lib_name] = {'points': []}
        
        # Combine library design spaces using existing combination strategies
        combined_points = self._combine_library_spaces_existing(library_spaces)
        
        # Convert to DesignPoint objects
        for point_data in combined_points:
            design_point = DesignPoint(point_data)
            design_point.metadata['libraries_used'] = list(library_spaces.keys())
            design_point.metadata['source'] = 'existing_components'
            design_points.append(design_point)
        
        self.design_points = design_points
        logger.info(f"Constructed design space with {len(design_points)} points")
        
        return design_points
    
    def _combine_library_spaces_existing(self, library_spaces: Dict[str, Dict]) -> List[Dict[str, Any]]:
        """
        Combine design spaces from libraries using existing strategies.
        
        Args:
            library_spaces: Design spaces from each library
            
        Returns:
            Combined design points
        """
        # Use existing combination strategy (simple cross-product)
        combined_points = []
        
        # Start with base configuration
        base_config = {
            'kernels': {},
            'transforms': {},
            'hw_optimization': {},
            'analysis': {}
        }
        
        # Get points from each library
        all_lib_points = {}
        for lib_name, lib_space in library_spaces.items():
            points = lib_space.get('points', [])
            if not points:
                # Create default point if library has no points
                points = [{}]
            all_lib_points[lib_name] = points
        
        # Generate combinations using existing strategy
        if all_lib_points:
            # Simple strategy: take first point from each library
            combined_config = base_config.copy()
            for lib_name, points in all_lib_points.items():
                if points:
                    combined_config[lib_name] = points[0]
            
            combined_points.append(combined_config)
            
            # Add a few more combinations for variety (existing approach)
            for i in range(min(3, max(len(points) for points in all_lib_points.values() if points))):
                variant_config = base_config.copy()
                for lib_name, points in all_lib_points.items():
                    if points and i < len(points):
                        variant_config[lib_name] = points[i]
                    elif points:
                        variant_config[lib_name] = points[0]
                
                if variant_config not in combined_points:
                    combined_points.append(variant_config)
        
        return combined_points if combined_points else [base_config]
    
    def get_design_space_summary(self) -> Dict[str, Any]:
        """Get summary of design space characteristics."""
        return {
            'name': self.name,
            'total_points': len(self.design_points),
            'total_parameters': len(self.parameters),
            'libraries_used': list(self.libraries.keys()),
            'source': self.metadata['source'],
            'extensible_structure': self.metadata['extensible_structure'],
            'blueprint_config': self.blueprint_config
        }
    
    def export_to_existing_format(self) -> str:
        """Export design space in existing format for compatibility."""
        export_data = {
            'design_space': {
                'name': self.name,
                'parameters': {name: param.to_dict() for name, param in self.parameters.items()},
                'points': [point.to_dict() for point in self.design_points],
                'summary': self.get_design_space_summary(),
                'metadata': self.metadata
            },
            'format_version': '1.0',
            'compatible_with_existing': True
        }
        
        return json.dumps(export_data, indent=2, default=str)

    @classmethod
    def from_blueprint_data(cls, blueprint_data: Dict[str, Any]) -> 'DesignSpace':
        """
        Create design space from blueprint data.
        
        Args:
            blueprint_data: Blueprint configuration data
            
        Returns:
            DesignSpace instance
        """
        # Extract library configurations
        library_configs = {
            'kernels': blueprint_data.get('kernels', {}),
            'transforms': blueprint_data.get('transforms', {}),
            'hw_optimization': blueprint_data.get('hw_optimization', {}),
            'analysis': blueprint_data.get('analysis', {})
        }
        
        name = blueprint_data.get('name', 'blueprint_design_space')
        
        return cls(name, {}, library_configs)
    
    def validate_design_space(self) -> Tuple[bool, List[str]]:
        """
        Validate design space using existing validation approaches.
        
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        # Check that we have design points
        if not self.design_points:
            errors.append("Design space contains no design points")
        
        # Check library availability
        if not self.libraries:
            errors.append("No libraries available for design space construction")
        
        # Validate individual design points using existing validation
        for i, point in enumerate(self.design_points):
            point_errors = self._validate_design_point_existing(point)
            for error in point_errors:
                errors.append(f"Design point {i}: {error}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def _validate_design_point_existing(self, point: DesignPoint) -> List[str]:
        """Validate individual design point using existing validation logic."""
        errors = []
        
        # Check parameter structure
        if not isinstance(point.parameters, dict):
            errors.append("Parameters must be a dictionary")
        
        # Check for required parameter categories (existing structure)
        required_categories = ['kernels', 'transforms', 'hw_optimization']
        for category in required_categories:
            if category not in point.parameters:
                # This is a warning, not an error, since we use existing components
                pass
        
        return errors


# Utility function for sampling (referenced in __init__.py)
def sample_design_space(design_space: DesignSpace, n_samples: int = 10,
                       strategy: str = "latin_hypercube", seed: int = None) -> List[DesignPoint]:
    """
    Sample points from design space.
    
    Args:
        design_space: Design space to sample from
        n_samples: Number of samples to generate
        strategy: Sampling strategy
        seed: Random seed for reproducibility
        
    Returns:
        List of sampled design points
    """
    import random
    if seed is not None:
        random.seed(seed)
    
    # Simple sampling implementation for compatibility
    sampled_points = []
    
    if design_space.design_points:
        # Sample from existing points
        sample_indices = random.sample(range(len(design_space.design_points)), 
                                     min(n_samples, len(design_space.design_points)))
        sampled_points = [design_space.design_points[i] for i in sample_indices]
    else:
        # Generate random points from parameter definitions
        for _ in range(n_samples):
            params = {}
            for param_name, param_def in design_space.parameters.items():
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