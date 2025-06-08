"""
Enhanced Blueprint base class with design space support.

This module provides the Blueprint class that integrates with the extensible
platform's design space exploration capabilities.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..core.design_space import DesignSpace


@dataclass
class Blueprint:
    """Enhanced Blueprint class with design space support."""
    
    # Core blueprint information
    name: str
    description: str
    architecture: str
    build_steps: List[str]
    parameters: Optional[Dict[str, Any]] = None
    
    # Enhanced fields for DSE
    design_space_data: Optional[Dict[str, Any]] = None
    finn_hooks_config: Optional[Dict[str, Any]] = None
    constraints: Optional[Dict[str, Any]] = None
    metrics_config: Optional[Dict[str, Any]] = None
    research_config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None
    
    # Raw YAML data for extensibility
    yaml_data: Optional[Dict[str, Any]] = None
    
    # Cached design space
    _design_space: Optional[DesignSpace] = None
    
    def __post_init__(self):
        """Initialize default values and validate blueprint."""
        if self.parameters is None:
            self.parameters = {}
        
        # Store YAML data if not provided
        if self.yaml_data is None:
            self.yaml_data = {
                'name': self.name,
                'description': self.description,
                'architecture': self.architecture,
                'build_steps': self.build_steps,
                'parameters': self.parameters,
                'design_space': self.design_space_data,
                'finn_hooks_config': self.finn_hooks_config,
                'constraints': self.constraints,
                'metrics_config': self.metrics_config,
                'research_config': self.research_config,
                'metadata': self.metadata
            }
    
    def has_design_space(self) -> bool:
        """Check if blueprint includes design space definition."""
        return (self.design_space_data is not None and 
                'dimensions' in self.design_space_data)
    
    def get_design_space(self) -> Optional[DesignSpace]:
        """
        Get design space for this blueprint.
        
        Returns cached design space or creates new one from blueprint data.
        """
        if self._design_space is not None:
            return self._design_space
        
        if not self.has_design_space():
            return None
        
        # Create design space from blueprint YAML data
        self._design_space = DesignSpace.from_blueprint_data(self.yaml_data)
        return self._design_space
    
    def get_dimension_ranges(self) -> Dict[str, Any]:
        """Get parameter dimension ranges for this blueprint."""
        design_space = self.get_design_space()
        if design_space:
            return design_space.get_dimension_ranges()
        return {}
    
    def get_constraints(self) -> Dict[str, Any]:
        """Get constraint definitions for this blueprint."""
        return self.constraints or {}
    
    def get_finn_hooks_config(self) -> Dict[str, Any]:
        """Get FINN hooks configuration (placeholder for future use)."""
        return self.finn_hooks_config or {}
    
    def get_metrics_config(self) -> Dict[str, Any]:
        """Get metrics collection configuration."""
        return self.metrics_config or {}
    
    def supports_parameter_sweep(self) -> bool:
        """Check if blueprint supports parameter sweep operations."""
        return self.has_design_space()
    
    def supports_dse(self) -> bool:
        """Check if blueprint supports design space exploration."""
        return (self.has_design_space() and 
                self.research_config is not None and
                self.research_config.get('dse_enabled', False))
    
    def get_recommended_parameters(self) -> Dict[str, List[Any]]:
        """
        Get recommended parameter ranges for parameter sweeps.
        
        Returns a dictionary suitable for use with parameter_sweep().
        """
        design_space = self.get_design_space()
        if not design_space:
            # Fallback to basic parameters
            return {
                'target_fps': [1000, 3000, 5000, 7500, 10000],
                'clk_period_ns': [2.5, 3.33, 4.0, 5.0]
            }
        
        # Extract reasonable ranges from design space
        recommended = {}
        for name, param_def in design_space.parameters.items():
            if param_def.type.value == 'categorical':
                recommended[name] = param_def.values
            elif param_def.type.value in ['integer', 'continuous']:
                # Create 5 evenly spaced values in range
                min_val, max_val = param_def.range
                if param_def.type.value == 'integer':
                    values = [int(min_val + i * (max_val - min_val) / 4) for i in range(5)]
                else:
                    values = [min_val + i * (max_val - min_val) / 4 for i in range(5)]
                recommended[name] = values
            elif param_def.type.value == 'boolean':
                recommended[name] = [True, False]
        
        return recommended
    
    def export_for_research(self) -> Dict[str, Any]:
        """Export blueprint data for research purposes."""
        return {
            'blueprint_info': {
                'name': self.name,
                'description': self.description,
                'architecture': self.architecture,
                'version': self.metadata.get('version') if self.metadata else None
            },
            'design_space': self.get_design_space().export_for_dse() if self.has_design_space() else None,
            'constraints': self.get_constraints(),
            'metrics_config': self.get_metrics_config(),
            'research_config': self.research_config,
            'finn_hooks_config': self.get_finn_hooks_config(),
            'recommended_parameters': self.get_recommended_parameters()
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert blueprint to dictionary representation."""
        return {
            'name': self.name,
            'description': self.description,
            'architecture': self.architecture,
            'build_steps': self.build_steps,
            'parameters': self.parameters,
            'design_space': self.design_space_data,
            'finn_hooks_config': self.finn_hooks_config,
            'constraints': self.constraints,
            'metrics_config': self.metrics_config,
            'research_config': self.research_config,
            'metadata': self.metadata,
            'has_design_space': self.has_design_space(),
            'supports_dse': self.supports_dse()
        }
    
    @classmethod
    def from_yaml_data(cls, yaml_data: Dict[str, Any]) -> 'Blueprint':
        """Create Blueprint from YAML data dictionary."""
        return cls(
            name=yaml_data['name'],
            description=yaml_data['description'],
            architecture=yaml_data['architecture'],
            build_steps=yaml_data['build_steps'],
            parameters=yaml_data.get('parameters'),
            design_space_data=yaml_data.get('design_space'),
            finn_hooks_config=yaml_data.get('finn_hooks_config'),
            constraints=yaml_data.get('constraints'),
            metrics_config=yaml_data.get('metrics_config'),
            research_config=yaml_data.get('research_config'),
            metadata=yaml_data.get('metadata'),
            yaml_data=yaml_data
        )
    
    @classmethod
    def from_yaml_file(cls, yaml_file: Path) -> 'Blueprint':
        """Create Blueprint from YAML file."""
        with open(yaml_file, 'r') as f:
            yaml_data = yaml.safe_load(f)
        
        return cls.from_yaml_data(yaml_data)
    
    def save_to_yaml(self, filepath: Path):
        """Save blueprint to YAML file."""
        # Ensure directory exists
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.yaml_data, f, default_flow_style=False, indent=2)
    
    def __str__(self) -> str:
        """String representation of blueprint."""
        design_space_info = ""
        if self.has_design_space():
            ds = self.get_design_space()
            param_count = len(ds.parameters) if ds else 0
            design_space_info = f" (DSE: {param_count} parameters)"
        
        return f"Blueprint(name='{self.name}', arch='{self.architecture}', steps={len(self.build_steps)}{design_space_info})"
    
    def __repr__(self) -> str:
        """Detailed representation of blueprint."""
        return self.__str__()


# Backward compatibility alias
BlueprintConfig = Blueprint