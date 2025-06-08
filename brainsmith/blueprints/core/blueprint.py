"""
Core Blueprint class implementation.

Represents a complete blueprint specification for FPGA accelerator design
space exploration, integrating with the Week 2 library system.
"""

import json
import yaml
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class Blueprint:
    """
    Core blueprint class for FPGA accelerator design specifications.
    
    A blueprint defines the complete specification for design space exploration,
    including library configurations, constraints, and optimization objectives.
    """
    
    def __init__(self, name: str, version: str = "1.0.0", 
                 description: str = "", metadata: Dict[str, Any] = None):
        """
        Initialize a blueprint.
        
        Args:
            name: Blueprint name
            version: Blueprint version
            description: Blueprint description  
            metadata: Additional metadata
        """
        self.name = name
        self.version = version
        self.description = description
        self.metadata = metadata or {}
        
        # Core blueprint sections
        self.libraries = {}
        self.design_space = {}
        self.constraints = {}
        self.objectives = []
        self.parameters = {}
        
        # Blueprint management
        self.created_at = datetime.now().isoformat()
        self.updated_at = self.created_at
        self._validated = False
        self._validation_errors = []
        
        self.logger = logging.getLogger(f"brainsmith.blueprints.{name}")
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Blueprint':
        """
        Create blueprint from dictionary representation.
        
        Args:
            data: Blueprint data dictionary
            
        Returns:
            Blueprint instance
        """
        # Extract basic information
        name = data.get('name', 'unnamed_blueprint')
        version = data.get('version', '1.0.0')
        description = data.get('description', '')
        metadata = data.get('metadata', {})
        
        # Create blueprint instance
        blueprint = cls(name, version, description, metadata)
        
        # Load blueprint sections
        blueprint.libraries = data.get('libraries', {})
        blueprint.design_space = data.get('design_space', {})
        blueprint.constraints = data.get('constraints', {})
        blueprint.objectives = data.get('objectives', [])
        blueprint.parameters = data.get('parameters', {})
        
        # Load timestamps if present
        if 'created_at' in data:
            blueprint.created_at = data['created_at']
        if 'updated_at' in data:
            blueprint.updated_at = data['updated_at']
        
        return blueprint
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Blueprint':
        """
        Create blueprint from JSON string.
        
        Args:
            json_str: JSON string representation
            
        Returns:
            Blueprint instance
        """
        try:
            data = json.loads(json_str)
            return cls.from_dict(data)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
    
    @classmethod
    def from_yaml(cls, yaml_str: str) -> 'Blueprint':
        """
        Create blueprint from YAML string.
        
        Args:
            yaml_str: YAML string representation
            
        Returns:
            Blueprint instance
        """
        try:
            data = yaml.safe_load(yaml_str)
            return cls.from_dict(data)
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML format: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert blueprint to dictionary representation.
        
        Returns:
            Blueprint data dictionary
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'metadata': self.metadata,
            'libraries': self.libraries,
            'design_space': self.design_space,
            'constraints': self.constraints,
            'objectives': self.objectives,
            'parameters': self.parameters,
            'created_at': self.created_at,
            'updated_at': self.updated_at
        }
    
    def to_json(self, indent: int = 2) -> str:
        """
        Convert blueprint to JSON string.
        
        Args:
            indent: JSON indentation
            
        Returns:
            JSON string representation
        """
        return json.dumps(self.to_dict(), indent=indent)
    
    def to_yaml(self) -> str:
        """
        Convert blueprint to YAML string.
        
        Returns:
            YAML string representation
        """
        return yaml.dump(self.to_dict(), default_flow_style=False, indent=2)
    
    def add_library_config(self, library_name: str, config: Dict[str, Any]):
        """
        Add configuration for a library.
        
        Args:
            library_name: Name of the library (kernels, transforms, hw_optim, analysis)
            config: Library configuration
        """
        self.libraries[library_name] = config
        self._mark_updated()
    
    def add_constraint(self, constraint_name: str, constraint_value: Any):
        """
        Add a constraint to the blueprint.
        
        Args:
            constraint_name: Name of the constraint
            constraint_value: Constraint specification
        """
        self.constraints[constraint_name] = constraint_value
        self._mark_updated()
    
    def add_objective(self, objective: str, optimization_type: str = "maximize"):
        """
        Add an optimization objective.
        
        Args:
            objective: Objective name (e.g., 'throughput', 'resource_efficiency')
            optimization_type: 'maximize' or 'minimize'
        """
        objective_spec = {
            'name': objective,
            'type': optimization_type
        }
        self.objectives.append(objective_spec)
        self._mark_updated()
    
    def set_design_space_config(self, config: Dict[str, Any]):
        """
        Set design space exploration configuration.
        
        Args:
            config: Design space configuration
        """
        self.design_space = config
        self._mark_updated()
    
    def get_library_config(self, library_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific library.
        
        Args:
            library_name: Library name
            
        Returns:
            Library configuration or None
        """
        return self.libraries.get(library_name)
    
    def get_kernels_config(self) -> Dict[str, Any]:
        """Get kernels library configuration."""
        return self.get_library_config('kernels') or {}
    
    def get_transforms_config(self) -> Dict[str, Any]:
        """Get transforms library configuration."""
        return self.get_library_config('transforms') or {}
    
    def get_hw_optim_config(self) -> Dict[str, Any]:
        """Get hardware optimization library configuration."""
        return self.get_library_config('hw_optim') or {}
    
    def get_analysis_config(self) -> Dict[str, Any]:
        """Get analysis library configuration."""
        return self.get_library_config('analysis') or {}
    
    def get_resource_constraints(self) -> Dict[str, Any]:
        """Get resource constraints."""
        return self.constraints.get('resource_limits', {})
    
    def get_performance_requirements(self) -> Dict[str, Any]:
        """Get performance requirements."""
        return self.constraints.get('performance_requirements', {})
    
    def get_optimization_objectives(self) -> List[Dict[str, Any]]:
        """Get optimization objectives."""
        return self.objectives
    
    def validate(self) -> bool:
        """
        Validate the blueprint.
        
        Returns:
            True if blueprint is valid
        """
        self._validation_errors = []
        
        # Validate basic structure
        if not self.name:
            self._validation_errors.append("Blueprint name is required")
        
        if not self.version:
            self._validation_errors.append("Blueprint version is required")
        
        # Validate library configurations
        valid_libraries = {'kernels', 'transforms', 'hw_optim', 'analysis'}
        for lib_name in self.libraries:
            if lib_name not in valid_libraries:
                self._validation_errors.append(f"Unknown library: {lib_name}")
        
        # Validate objectives
        valid_optimization_types = {'maximize', 'minimize'}
        for obj in self.objectives:
            if isinstance(obj, dict):
                if 'name' not in obj:
                    self._validation_errors.append("Objective must have 'name' field")
                if obj.get('type', 'maximize') not in valid_optimization_types:
                    self._validation_errors.append(f"Invalid optimization type: {obj.get('type')}")
        
        # Validate design space configuration
        if self.design_space:
            if 'exploration_strategy' in self.design_space:
                valid_strategies = {'pareto_optimal', 'random', 'grid', 'genetic', 'bayesian'}
                strategy = self.design_space['exploration_strategy']
                if strategy not in valid_strategies:
                    self._validation_errors.append(f"Unknown exploration strategy: {strategy}")
        
        self._validated = len(self._validation_errors) == 0
        return self._validated
    
    def get_validation_errors(self) -> List[str]:
        """Get validation errors."""
        return self._validation_errors.copy()
    
    def is_valid(self) -> bool:
        """Check if blueprint is valid."""
        return self._validated
    
    def clone(self, new_name: str = None) -> 'Blueprint':
        """
        Create a copy of this blueprint.
        
        Args:
            new_name: Name for the cloned blueprint
            
        Returns:
            New blueprint instance
        """
        data = self.to_dict()
        if new_name:
            data['name'] = new_name
        
        cloned = Blueprint.from_dict(data)
        cloned.created_at = datetime.now().isoformat()
        cloned.updated_at = cloned.created_at
        
        return cloned
    
    def merge_with(self, other: 'Blueprint', strategy: str = "override"):
        """
        Merge with another blueprint.
        
        Args:
            other: Blueprint to merge with
            strategy: Merge strategy ('override', 'merge', 'append')
        """
        if strategy == "override":
            # Other blueprint overrides this one
            self.libraries.update(other.libraries)
            self.constraints.update(other.constraints)
            self.design_space.update(other.design_space)
            self.objectives = other.objectives
            
        elif strategy == "merge":
            # Deep merge configurations
            self._deep_merge_dict(self.libraries, other.libraries)
            self._deep_merge_dict(self.constraints, other.constraints)
            self._deep_merge_dict(self.design_space, other.design_space)
            
        elif strategy == "append":
            # Append objectives, override others
            self.libraries.update(other.libraries)
            self.constraints.update(other.constraints)
            self.design_space.update(other.design_space)
            self.objectives.extend(other.objectives)
        
        self._mark_updated()
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get blueprint summary.
        
        Returns:
            Summary information
        """
        return {
            'name': self.name,
            'version': self.version,
            'description': self.description,
            'libraries_configured': list(self.libraries.keys()),
            'num_constraints': len(self.constraints),
            'num_objectives': len(self.objectives),
            'has_design_space_config': bool(self.design_space),
            'created_at': self.created_at,
            'updated_at': self.updated_at,
            'is_valid': self._validated
        }
    
    def _mark_updated(self):
        """Mark blueprint as updated."""
        self.updated_at = datetime.now().isoformat()
        self._validated = False  # Require re-validation after changes
    
    def _deep_merge_dict(self, target: Dict, source: Dict):
        """Deep merge source dict into target dict."""
        for key, value in source.items():
            if key in target and isinstance(target[key], dict) and isinstance(value, dict):
                self._deep_merge_dict(target[key], value)
            else:
                target[key] = value
    
    def __str__(self) -> str:
        """String representation."""
        return f"Blueprint(name='{self.name}', version='{self.version}', libraries={list(self.libraries.keys())})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return (f"Blueprint(name='{self.name}', version='{self.version}', "
                f"libraries={list(self.libraries.keys())}, "
                f"constraints={len(self.constraints)}, "
                f"objectives={len(self.objectives)})")