"""
Simplified Blueprint Functions - North Star Aligned Implementation

This module provides simple functions for loading and working with blueprint YAML files,
replacing the complex Blueprint dataclass with functions that follow the North Star axioms:
- Functions Over Frameworks
- Simplicity Over Sophistication 
- Focus Over Feature Creep
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


def load_blueprint_yaml(blueprint_path: str) -> Dict[str, Any]:
    """
    Load blueprint YAML file as a simple dictionary.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Dictionary containing blueprint configuration
        
    Raises:
        FileNotFoundError: If blueprint file doesn't exist
        ValueError: If YAML is invalid or missing required fields
    """
    blueprint_file = Path(blueprint_path)
    
    if not blueprint_file.exists():
        raise FileNotFoundError(f"Blueprint file not found: {blueprint_path}")
    
    if not blueprint_path.lower().endswith(('.yaml', '.yml')):
        raise ValueError(f"Blueprint must be YAML format, got: {blueprint_path}")
    
    try:
        with open(blueprint_file, 'r') as f:
            blueprint_data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ValueError(f"Invalid YAML in blueprint '{blueprint_path}': {str(e)}")
    
    if not isinstance(blueprint_data, dict):
        raise ValueError(f"Blueprint must be a dictionary, got {type(blueprint_data)}")
    
    return blueprint_data


def validate_blueprint_yaml(blueprint_data: Dict[str, Any]) -> tuple[bool, List[str]]:
    """
    Validate blueprint configuration with graceful error handling.
    
    Args:
        blueprint_data: Blueprint dictionary from load_blueprint_yaml()
        
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    
    # Check required fields with defaults
    required_fields = {
        'name': 'unnamed_blueprint',
        'build_steps': []
    }
    
    for field, default_value in required_fields.items():
        if field not in blueprint_data:
            if field == 'name':
                errors.append(f"Missing required field: {field}")
            else:
                # Provide default for build_steps
                blueprint_data[field] = default_value
    
    # Validate build_steps format
    if 'build_steps' in blueprint_data:
        build_steps = blueprint_data['build_steps']
        if not isinstance(build_steps, list):
            errors.append("build_steps must be a list")
        elif len(build_steps) == 0:
            errors.append("build_steps cannot be empty")
        else:
            for i, step in enumerate(build_steps):
                if not isinstance(step, str):
                    errors.append(f"build_steps[{i}] must be a string, got {type(step)}")
    
    # Validate optional fields if present
    if 'objectives' in blueprint_data:
        objectives = blueprint_data['objectives']
        if isinstance(objectives, dict):
            for obj_name, obj_config in objectives.items():
                if isinstance(obj_config, dict) and 'direction' in obj_config:
                    if obj_config['direction'] not in ['maximize', 'minimize']:
                        errors.append(f"Objective '{obj_name}' direction must be 'maximize' or 'minimize'")
    
    return len(errors) == 0, errors


def get_build_steps(blueprint_data: Dict[str, Any]) -> List[str]:
    """
    Extract build steps from blueprint data.
    
    Args:
        blueprint_data: Blueprint dictionary
        
    Returns:
        List of build step names
    """
    return blueprint_data.get('build_steps', [])


def get_objectives(blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract optimization objectives from blueprint data.
    
    Args:
        blueprint_data: Blueprint dictionary
        
    Returns:
        Dictionary of objectives with defaults
    """
    objectives = blueprint_data.get('objectives', {})
    
    # Provide sensible defaults if none specified
    if not objectives:
        objectives = {
            'throughput': {'direction': 'maximize', 'weight': 1.0},
            'latency': {'direction': 'minimize', 'weight': 0.8}
        }
    
    return objectives


def get_constraints(blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract hardware constraints from blueprint data.
    
    Args:
        blueprint_data: Blueprint dictionary
        
    Returns:
        Dictionary of constraints with defaults
    """
    constraints = blueprint_data.get('constraints', {})
    
    # Provide sensible defaults if none specified
    if not constraints:
        constraints = {
            'max_luts': 0.8,
            'max_dsps': 0.8,
            'max_brams': 0.8
        }
    
    return constraints


def get_kernels(blueprint_data: Dict[str, Any]) -> List[str]:
    """
    Extract kernel list from blueprint data.
    
    Args:
        blueprint_data: Blueprint dictionary
        
    Returns:
        List of kernel names, empty if not specified
    """
    return blueprint_data.get('kernels', [])


def get_transforms(blueprint_data: Dict[str, Any]) -> List[str]:
    """
    Extract transform list from blueprint data.
    
    Args:
        blueprint_data: Blueprint dictionary
        
    Returns:
        List of transform names, empty if not specified
    """
    return blueprint_data.get('transforms', [])


def create_simple_blueprint(
    name: str,
    build_steps: List[str],
    objectives: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    kernels: Optional[List[str]] = None,
    transforms: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Create a simple blueprint dictionary programmatically.
    
    Args:
        name: Blueprint name
        build_steps: List of build step names
        objectives: Optional optimization objectives
        constraints: Optional hardware constraints
        kernels: Optional kernel list
        transforms: Optional transform list
        
    Returns:
        Blueprint dictionary ready for use
    """
    blueprint = {
        'name': name,
        'build_steps': build_steps
    }
    
    if objectives:
        blueprint['objectives'] = objectives
    
    if constraints:
        blueprint['constraints'] = constraints
    
    if kernels:
        blueprint['kernels'] = kernels
    
    if transforms:
        blueprint['transforms'] = transforms
    
    return blueprint


def save_blueprint_yaml(blueprint_data: Dict[str, Any], output_path: str):
    """
    Save blueprint dictionary to YAML file.
    
    Args:
        blueprint_data: Blueprint dictionary
        output_path: Path to save YAML file
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(blueprint_data, f, default_flow_style=False, indent=2)