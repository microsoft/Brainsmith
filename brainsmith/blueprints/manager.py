"""
Blueprint Manager for loading and executing YAML blueprints.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from brainsmith.steps import get_step, validate_steps


@dataclass
class BlueprintConfig:
    """Configuration data for a blueprint."""
    name: str
    description: str
    architecture: str
    build_steps: List[str]
    parameters: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}


class BlueprintManager:
    """Manager for loading and executing YAML blueprints."""
    
    def __init__(self):
        self._blueprints: Dict[str, BlueprintConfig] = {}
        self._blueprint_dirs = [
            Path(__file__).parent / "yaml",  # Default YAML blueprint directory
        ]
    
    def add_blueprint_directory(self, path: Path):
        """Add a directory to search for blueprint YAML files."""
        self._blueprint_dirs.append(Path(path))
    
    def load_blueprint(self, name: str) -> BlueprintConfig:
        """Load a blueprint by name from YAML files."""
        if name in self._blueprints:
            return self._blueprints[name]
        
        # Search for blueprint YAML file
        for blueprint_dir in self._blueprint_dirs:
            yaml_file = blueprint_dir / f"{name}.yaml"
            if yaml_file.exists():
                return self._load_blueprint_from_file(yaml_file)
        
        raise ValueError(f"Blueprint '{name}' not found in any blueprint directory")
    
    def _load_blueprint_from_file(self, yaml_file: Path) -> BlueprintConfig:
        """Load blueprint from a YAML file."""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['name', 'description', 'architecture', 'build_steps']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Blueprint '{yaml_file}' missing required field: {field}")
        
        config = BlueprintConfig(
            name=data['name'],
            description=data['description'],
            architecture=data['architecture'],
            build_steps=data['build_steps'],
            parameters=data.get('parameters', {})
        )
        
        # Validate build steps
        errors = validate_steps(config.build_steps)
        if errors:
            raise ValueError(f"Blueprint '{config.name}' has invalid steps:\n" + "\n".join(errors))
        
        # Cache the blueprint
        self._blueprints[config.name] = config
        return config
    
    def list_available_blueprints(self) -> List[str]:
        """List all available blueprint names."""
        blueprints = set()
        
        for blueprint_dir in self._blueprint_dirs:
            if blueprint_dir.exists():
                for yaml_file in blueprint_dir.glob("*.yaml"):
                    blueprints.add(yaml_file.stem)
        
        return sorted(blueprints)
    
    def execute_blueprint(self, blueprint_name: str, model, cfg):
        """Execute a blueprint on a model."""
        blueprint = self.load_blueprint(blueprint_name)
        
        print(f"Executing blueprint: {blueprint.name}")
        print(f"Description: {blueprint.description}")
        print(f"Architecture: {blueprint.architecture}")
        print(f"Steps: {len(blueprint.build_steps)}")
        
        # Execute each step in sequence
        for i, step_name in enumerate(blueprint.build_steps):
            print(f"  [{i+1}/{len(blueprint.build_steps)}] {step_name}")
            step_func = get_step(step_name)
            model = step_func(model, cfg)
        
        return model
    
    def get_build_steps(self, blueprint_name: str) -> List[str]:
        """Get the build steps for a blueprint (for backward compatibility)."""
        blueprint = self.load_blueprint(blueprint_name)
        
        # Convert step names to actual functions for backward compatibility
        build_steps = []
        for step_name in blueprint.build_steps:
            build_steps.append(get_step(step_name))
        
        return build_steps


# Global blueprint manager instance
BLUEPRINT_MANAGER = BlueprintManager()

# Convenience functions
def load_blueprint(name: str) -> BlueprintConfig:
    """Load a blueprint by name."""
    return BLUEPRINT_MANAGER.load_blueprint(name)

def execute_blueprint(blueprint_name: str, model, cfg):
    """Execute a blueprint on a model."""
    return BLUEPRINT_MANAGER.execute_blueprint(blueprint_name, model, cfg)

def get_build_steps(blueprint_name: str) -> List[str]:
    """Get build steps for a blueprint (backward compatibility)."""
    return BLUEPRINT_MANAGER.get_build_steps(blueprint_name)

def list_blueprints() -> List[str]:
    """List available blueprints."""
    return BLUEPRINT_MANAGER.list_available_blueprints()