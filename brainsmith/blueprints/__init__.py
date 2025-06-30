"""
Blueprint Library

Provides access to declarative blueprint YAML configurations.
Blueprints define compilation flows without requiring custom Python code.
"""

import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


class BlueprintLibrary:
    """Library of available blueprints.
    
    Blueprints are YAML configurations that define:
    - hw_compiler: Configuration for DSE v3 backend (kernels, transforms)
    - build_steps: Steps for legacy FINN backend
    - finn_config: FINN-specific settings used by both backends
    """
    
    def __init__(self):
        """Initialize the blueprint library."""
        self.blueprint_dir = Path(__file__).parent
        self._cache = {}
    
    def list_blueprints(self) -> List[str]:
        """List all available blueprint names.
        
        Returns:
            List of blueprint names (without .yaml extension)
        """
        blueprints = []
        for yaml_file in self.blueprint_dir.glob("*.yaml"):
            blueprints.append(yaml_file.stem)
        return sorted(blueprints)
    
    def load_blueprint(self, name: str) -> Dict[str, Any]:
        """Load a blueprint by name.
        
        Args:
            name: Blueprint name (without .yaml extension)
            
        Returns:
            Parsed blueprint data
            
        Raises:
            FileNotFoundError: If blueprint doesn't exist
            yaml.YAMLError: If blueprint is invalid YAML
        """
        if name in self._cache:
            return self._cache[name]
        
        path = self.blueprint_dir / f"{name}.yaml"
        if not path.exists():
            available = self.list_blueprints()
            raise FileNotFoundError(
                f"Blueprint '{name}' not found. Available blueprints: {available}"
            )
        
        logger.info(f"Loading blueprint: {name}")
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        self._cache[name] = data
        return data
    
    def get_build_steps(self, blueprint_data: Dict[str, Any]) -> List[str]:
        """Extract build steps for legacy FINN backend.
        
        Args:
            blueprint_data: Loaded blueprint data
            
        Returns:
            List of build step names
        """
        return blueprint_data.get("build_steps", [])
    
    def get_finn_config(self, blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract FINN configuration.
        
        This configuration is used by both legacy and DSE v3 backends.
        
        Args:
            blueprint_data: Loaded blueprint data
            
        Returns:
            FINN configuration dictionary
        """
        return blueprint_data.get("finn_config", {})
    
    def get_hw_compiler_config(self, blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract hardware compiler configuration for DSE v3.
        
        Args:
            blueprint_data: Loaded blueprint data
            
        Returns:
            hw_compiler section containing kernels and transforms
        """
        return blueprint_data.get("hw_compiler", {})
    
    def get_description(self, blueprint_data: Dict[str, Any]) -> str:
        """Get blueprint description.
        
        Args:
            blueprint_data: Loaded blueprint data
            
        Returns:
            Blueprint description or empty string
        """
        return blueprint_data.get("description", "")
    
    def get_version(self, blueprint_data: Dict[str, Any]) -> str:
        """Get blueprint version.
        
        Args:
            blueprint_data: Loaded blueprint data
            
        Returns:
            Blueprint version (e.g., "3.0")
        """
        return blueprint_data.get("version", "")
    
    def validate_version(self, blueprint_data: Dict[str, Any], expected: str = "3.0") -> bool:
        """Validate blueprint version.
        
        Args:
            blueprint_data: Loaded blueprint data
            expected: Expected version string
            
        Returns:
            True if version matches expected
        """
        version = self.get_version(blueprint_data)
        return version == expected


# Global instance for convenience
_library = BlueprintLibrary()

# Convenience functions
def list_blueprints() -> List[str]:
    """List all available blueprints."""
    return _library.list_blueprints()

def load_blueprint(name: str) -> Dict[str, Any]:
    """Load a blueprint by name."""
    return _library.load_blueprint(name)

def get_build_steps(blueprint_data: Dict[str, Any]) -> List[str]:
    """Extract build steps from blueprint data."""
    return _library.get_build_steps(blueprint_data)

def get_finn_config(blueprint_data: Dict[str, Any]) -> Dict[str, Any]:
    """Extract FINN config from blueprint data."""
    return _library.get_finn_config(blueprint_data)