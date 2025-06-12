"""
Blueprint Management Infrastructure

Manages blueprint collections and provides design space exploration
functionality. This module handles blueprint loading, validation,
and integration with DSE operations.
"""

import os
import yaml
import logging
from typing import Dict, Any, List, Optional, Union
from pathlib import Path

# Import required modules
from .design_space import DesignPoint
from .types import ParameterSpace

logger = logging.getLogger(__name__)


class BlueprintManager:
    """Manages blueprint collections for design space exploration."""
    
    def __init__(self, blueprint_dirs: Optional[List[str]] = None):
        """
        Initialize blueprint manager.
        
        Args:
            blueprint_dirs: List of directories to search for blueprints.
                          Defaults to ['libraries/blueprints/']
        """
        self.blueprint_dirs = blueprint_dirs or ['libraries/blueprints/']
        self.blueprints_cache = {}
        self.blueprint_metadata = {}
        
        logger.info(f"Blueprint manager initialized with dirs: {self.blueprint_dirs}")
    
    def discover_blueprints(self) -> Dict[str, str]:
        """
        Discover all available blueprints.
        
        Returns:
            Dictionary mapping blueprint names to file paths
        """
        discovered = {}
        
        for blueprint_dir in self.blueprint_dirs:
            if not os.path.exists(blueprint_dir):
                logger.warning(f"Blueprint directory not found: {blueprint_dir}")
                continue
                
            for root, dirs, files in os.walk(blueprint_dir):
                for file in files:
                    if file.endswith(('.yaml', '.yml')):
                        file_path = os.path.join(root, file)
                        blueprint_name = self._extract_blueprint_name(file_path)
                        discovered[blueprint_name] = file_path
                        
        logger.info(f"Discovered {len(discovered)} blueprints")
        return discovered
    
    def load_blueprint(self, blueprint_name: str) -> Dict[str, Any]:
        """
        Load a specific blueprint by name.
        
        Args:
            blueprint_name: Name of the blueprint to load
            
        Returns:
            Blueprint configuration dictionary
            
        Raises:
            FileNotFoundError: If blueprint not found
            yaml.YAMLError: If blueprint file is invalid
        """
        if blueprint_name in self.blueprints_cache:
            return self.blueprints_cache[blueprint_name]
        
        # Discover blueprints if not already done
        discovered = self.discover_blueprints()
        
        if blueprint_name not in discovered:
            raise FileNotFoundError(f"Blueprint '{blueprint_name}' not found")
        
        blueprint_path = discovered[blueprint_name]
        
        try:
            with open(blueprint_path, 'r') as f:
                blueprint_config = yaml.safe_load(f)
            
            # Validate blueprint structure
            self._validate_blueprint(blueprint_config, blueprint_name)
            
            # Cache the blueprint
            self.blueprints_cache[blueprint_name] = blueprint_config
            self.blueprint_metadata[blueprint_name] = {
                'path': blueprint_path,
                'last_loaded': os.path.getmtime(blueprint_path)
            }
            
            logger.info(f"Loaded blueprint: {blueprint_name}")
            return blueprint_config
            
        except yaml.YAMLError as e:
            logger.error(f"Invalid YAML in blueprint {blueprint_name}: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading blueprint {blueprint_name}: {e}")
            raise
    
    def get_blueprint_parameter_space(self, blueprint_name: str) -> ParameterSpace:
        """
        Extract parameter space from a blueprint.
        
        Args:
            blueprint_name: Name of the blueprint
            
        Returns:
            ParameterSpace object for DSE operations
        """
        blueprint = self.load_blueprint(blueprint_name)
        
        # Extract parameter ranges from blueprint
        parameter_space = {}
        
        if 'parameters' in blueprint:
            for param_name, param_config in blueprint['parameters'].items():
                if isinstance(param_config, dict):
                    if 'range' in param_config:
                        parameter_space[param_name] = param_config['range']
                    elif 'values' in param_config:
                        parameter_space[param_name] = param_config['values']
                    else:
                        parameter_space[param_name] = param_config.get('default', [1])
                else:
                    parameter_space[param_name] = [param_config]
        
        return ParameterSpace(parameter_space)
    
    def create_design_point(self, blueprint_name: str,
                          parameter_values: Dict[str, Any]) -> Optional[Any]:
        """
        Create a design point from a blueprint and parameter values.
        
        Args:
            blueprint_name: Name of the blueprint
            parameter_values: Parameter values for this design point
            
        Returns:
            DesignPoint object or None if DesignPoint class not available
        """
        blueprint = self.load_blueprint(blueprint_name)
        
        # Merge blueprint base config with parameter values
        design_config = blueprint.copy()
        if 'parameters' in design_config:
            del design_config['parameters']  # Remove parameter definitions
        
        # Apply parameter values
        design_config.update(parameter_values)
        
        return DesignPoint(
            parameters=parameter_values
        )
    
    def validate_design_point(self, blueprint_name: str,
                            design_point: Any) -> tuple[bool, List[str]]:
        """
        Validate a design point against its blueprint.
        
        Args:
            blueprint_name: Name of the blueprint
            design_point: Design point to validate
            
        Returns:
            Tuple of (is_valid, list_of_errors)
        """
        errors = []
        
        if design_point is None:
            return False, ["Design point is None"]
        
        try:
            blueprint = self.load_blueprint(blueprint_name)
        except Exception as e:
            return False, [f"Could not load blueprint: {e}"]
        
        # Get parameters from design point (handle both dict and object)
        if hasattr(design_point, 'parameters'):
            parameters = design_point.parameters
        elif isinstance(design_point, dict):
            parameters = design_point.get('parameters', {})
        else:
            return False, ["Invalid design point format"]
        
        # Validate required parameters
        if 'parameters' in blueprint:
            for param_name, param_config in blueprint['parameters'].items():
                if isinstance(param_config, dict) and param_config.get('required', False):
                    if param_name not in parameters:
                        errors.append(f"Required parameter '{param_name}' missing")
        
        # Validate parameter values against ranges/choices
        for param_name, value in parameters.items():
            if 'parameters' in blueprint and param_name in blueprint['parameters']:
                param_config = blueprint['parameters'][param_name]
                if isinstance(param_config, dict):
                    # Validate against range
                    if 'range' in param_config:
                        param_range = param_config['range']
                        if isinstance(param_range, list) and len(param_range) >= 2:
                            if not (param_range[0] <= value <= param_range[-1]):
                                errors.append(
                                    f"Parameter '{param_name}' value {value} "
                                    f"outside range {param_range}"
                                )
                    
                    # Validate against choices
                    if 'values' in param_config:
                        if value not in param_config['values']:
                            errors.append(
                                f"Parameter '{param_name}' value {value} "
                                f"not in allowed values {param_config['values']}"
                            )
        
        return len(errors) == 0, errors
    
    def list_blueprints(self) -> List[str]:
        """List all available blueprint names."""
        return list(self.discover_blueprints().keys())
    
    def get_blueprint_info(self, blueprint_name: str) -> Dict[str, Any]:
        """
        Get metadata information about a blueprint.
        
        Args:
            blueprint_name: Name of the blueprint
            
        Returns:
            Dictionary with blueprint metadata
        """
        blueprint = self.load_blueprint(blueprint_name)
        
        return {
            'name': blueprint_name,
            'description': blueprint.get('description', 'No description'),
            'version': blueprint.get('version', '1.0'),
            'parameters': list(blueprint.get('parameters', {}).keys()),
            'path': self.blueprint_metadata.get(blueprint_name, {}).get('path', ''),
            'parameter_count': len(blueprint.get('parameters', {}))
        }
    
    def refresh_cache(self):
        """Refresh the blueprint cache by clearing it."""
        self.blueprints_cache.clear()
        self.blueprint_metadata.clear()
        logger.info("Blueprint cache refreshed")
    
    def _extract_blueprint_name(self, file_path: str) -> str:
        """Extract blueprint name from file path."""
        return Path(file_path).stem
    
    def _validate_blueprint(self, blueprint: Dict[str, Any], name: str):
        """
        Validate blueprint structure.
        
        Args:
            blueprint: Blueprint configuration
            name: Blueprint name for error messages
            
        Raises:
            ValueError: If blueprint structure is invalid
        """
        if not isinstance(blueprint, dict):
            raise ValueError(f"Blueprint '{name}' must be a dictionary")
        
        # Check for required top-level keys
        # Allow flexible structure but validate parameter definitions if present
        if 'parameters' in blueprint:
            if not isinstance(blueprint['parameters'], dict):
                raise ValueError(f"Blueprint '{name}' parameters must be a dictionary")


# Convenience functions for blueprint management
def load_blueprint(blueprint_name: str, blueprint_dirs: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Load a blueprint configuration.
    
    Args:
        blueprint_name: Name of the blueprint to load
        blueprint_dirs: Optional list of directories to search
        
    Returns:
        Blueprint configuration dictionary
    """
    manager = BlueprintManager(blueprint_dirs)
    return manager.load_blueprint(blueprint_name)


def list_available_blueprints(blueprint_dirs: Optional[List[str]] = None) -> List[str]:
    """
    List all available blueprints.
    
    Args:
        blueprint_dirs: Optional list of directories to search
        
    Returns:
        List of blueprint names
    """
    manager = BlueprintManager(blueprint_dirs)
    return manager.list_blueprints()


def create_design_point_from_blueprint(blueprint_name: str,
                                     parameter_values: Dict[str, Any],
                                     blueprint_dirs: Optional[List[str]] = None) -> Optional[Any]:
    """
    Create a design point from a blueprint.
    
    Args:
        blueprint_name: Name of the blueprint
        parameter_values: Parameter values for the design point
        blueprint_dirs: Optional list of directories to search
        
    Returns:
        DesignPoint object or None if not available
    """
    manager = BlueprintManager(blueprint_dirs)
    return manager.create_design_point(blueprint_name, parameter_values)