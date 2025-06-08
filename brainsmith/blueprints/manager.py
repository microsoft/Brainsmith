"""
Enhanced Blueprint Manager with design space support.

This module provides blueprint loading and management capabilities integrated
with the extensible platform's design space exploration features.
"""

import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

from brainsmith.steps import get_step, validate_steps
from .base import Blueprint
from ..core.design_space import DesignSpace


class BlueprintManager:
    """Enhanced manager for loading and executing YAML blueprints with DSE support."""
    
    def __init__(self):
        self._blueprints: Dict[str, Blueprint] = {}
        self._design_spaces: Dict[str, DesignSpace] = {}
        self._blueprint_dirs = [
            Path(__file__).parent / "yaml",  # Default YAML blueprint directory
        ]
    
    def add_blueprint_directory(self, path: Path):
        """Add a directory to search for blueprint YAML files."""
        self._blueprint_dirs.append(Path(path))
    
    def load_blueprint(self, name: str) -> Blueprint:
        """Load a blueprint by name from YAML files."""
        if name in self._blueprints:
            return self._blueprints[name]
        
        # Search for blueprint YAML file
        for blueprint_dir in self._blueprint_dirs:
            yaml_file = blueprint_dir / f"{name}.yaml"
            if yaml_file.exists():
                return self._load_blueprint_from_file(yaml_file)
        
        raise ValueError(f"Blueprint '{name}' not found in any blueprint directory")
    
    def _load_blueprint_from_file(self, yaml_file: Path) -> Blueprint:
        """Load blueprint from a YAML file."""
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        
        # Validate required fields
        required_fields = ['name', 'description', 'architecture', 'build_steps']
        for field in required_fields:
            if field not in data:
                raise ValueError(f"Blueprint '{yaml_file}' missing required field: {field}")
        
        # Create enhanced blueprint
        blueprint = Blueprint.from_yaml_data(data)
        
        # Validate build steps
        errors = validate_steps(blueprint.build_steps)
        if errors:
            raise ValueError(f"Blueprint '{blueprint.name}' has invalid steps:\n" + "\n".join(errors))
        
        # Cache the blueprint
        self._blueprints[blueprint.name] = blueprint
        
        # Cache design space if available
        if blueprint.has_design_space():
            design_space = blueprint.get_design_space()
            if design_space:
                self._design_spaces[blueprint.name] = design_space
        
        return blueprint
    
    def get_design_space(self, blueprint_name: str) -> Optional[DesignSpace]:
        """Get design space for a blueprint."""
        if blueprint_name in self._design_spaces:
            return self._design_spaces[blueprint_name]
        
        # Load blueprint and extract design space
        blueprint = self.load_blueprint(blueprint_name)
        design_space = blueprint.get_design_space()
        
        if design_space:
            self._design_spaces[blueprint_name] = design_space
        
        return design_space
    
    def list_available_blueprints(self) -> List[str]:
        """List all available blueprint names."""
        blueprints = set()
        
        for blueprint_dir in self._blueprint_dirs:
            if blueprint_dir.exists():
                for yaml_file in blueprint_dir.glob("*.yaml"):
                    blueprints.add(yaml_file.stem)
        
        return sorted(blueprints)
    
    def list_blueprints_with_dse(self) -> List[str]:
        """List blueprints that support design space exploration."""
        dse_blueprints = []
        
        for blueprint_name in self.list_available_blueprints():
            try:
                blueprint = self.load_blueprint(blueprint_name)
                if blueprint.supports_dse():
                    dse_blueprints.append(blueprint_name)
            except Exception:
                # Skip blueprints that fail to load
                continue
        
        return dse_blueprints
    
    def get_blueprint_info(self) -> List[Dict[str, Any]]:
        """Get detailed information about all available blueprints."""
        blueprint_info = []
        
        for blueprint_name in self.list_available_blueprints():
            try:
                blueprint = self.load_blueprint(blueprint_name)
                info = {
                    'name': blueprint.name,
                    'description': blueprint.description,
                    'architecture': blueprint.architecture,
                    'step_count': len(blueprint.build_steps),
                    'has_design_space': blueprint.has_design_space(),
                    'supports_dse': blueprint.supports_dse(),
                    'parameter_count': 0
                }
                
                if blueprint.has_design_space():
                    design_space = blueprint.get_design_space()
                    if design_space:
                        info['parameter_count'] = len(design_space.parameters)
                        info['estimated_space_size'] = design_space.estimate_space_size()
                
                blueprint_info.append(info)
                
            except Exception as e:
                # Add error info for failed blueprints
                blueprint_info.append({
                    'name': blueprint_name,
                    'description': f"Failed to load: {str(e)}",
                    'architecture': 'unknown',
                    'step_count': 0,
                    'has_design_space': False,
                    'supports_dse': False,
                    'parameter_count': 0,
                    'error': str(e)
                })
        
        return blueprint_info
    
    def execute_blueprint(self, blueprint_name: str, model, cfg):
        """Execute a blueprint on a model."""
        blueprint = self.load_blueprint(blueprint_name)
        
        print(f"Executing blueprint: {blueprint.name}")
        print(f"Description: {blueprint.description}")
        print(f"Architecture: {blueprint.architecture}")
        print(f"Steps: {len(blueprint.build_steps)}")
        
        if blueprint.has_design_space():
            design_space = blueprint.get_design_space()
            if design_space:
                print(f"Design space: {len(design_space.parameters)} parameters")
        
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
    
    def validate_blueprint(self, blueprint_name: str) -> List[str]:
        """Validate a blueprint and return any issues."""
        issues = []
        
        try:
            blueprint = self.load_blueprint(blueprint_name)
            
            # Validate build steps
            step_errors = validate_steps(blueprint.build_steps)
            issues.extend(step_errors)
            
            # Validate design space if present
            if blueprint.has_design_space():
                design_space = blueprint.get_design_space()
                if design_space:
                    # Try to sample a point to validate design space
                    try:
                        test_point = design_space.sample_random_point()
                        valid, violations = design_space.validate_design_point(test_point)
                        if not valid:
                            issues.extend([f"Design space validation: {v}" for v in violations])
                    except Exception as e:
                        issues.append(f"Design space sampling error: {str(e)}")
            
        except Exception as e:
            issues.append(f"Blueprint loading error: {str(e)}")
        
        return issues
    
    def export_blueprint_for_research(self, blueprint_name: str) -> Dict[str, Any]:
        """Export blueprint data for research purposes."""
        blueprint = self.load_blueprint(blueprint_name)
        return blueprint.export_for_research()
    
    def create_blueprint_from_template(self, name: str, template_name: str, 
                                     modifications: Dict[str, Any]) -> Blueprint:
        """Create a new blueprint based on an existing template with modifications."""
        template = self.load_blueprint(template_name)
        
        # Start with template data
        new_data = template.yaml_data.copy()
        
        # Apply modifications
        new_data['name'] = name
        new_data.update(modifications)
        
        # Create new blueprint
        new_blueprint = Blueprint.from_yaml_data(new_data)
        
        # Cache the new blueprint
        self._blueprints[name] = new_blueprint
        
        return new_blueprint


# Global blueprint manager instance
BLUEPRINT_MANAGER = BlueprintManager()

# Enhanced convenience functions
def load_blueprint(name: str) -> Blueprint:
    """Load a blueprint by name."""
    return BLUEPRINT_MANAGER.load_blueprint(name)

def get_blueprint(name: str) -> Blueprint:
    """Get a blueprint by name (alias for load_blueprint)."""
    return BLUEPRINT_MANAGER.load_blueprint(name)

def get_design_space(blueprint_name: str) -> Optional[DesignSpace]:
    """Get design space for a blueprint."""
    return BLUEPRINT_MANAGER.get_design_space(blueprint_name)

def execute_blueprint(blueprint_name: str, model, cfg):
    """Execute a blueprint on a model."""
    return BLUEPRINT_MANAGER.execute_blueprint(blueprint_name, model, cfg)

def get_build_steps(blueprint_name: str) -> List[str]:
    """Get build steps for a blueprint (backward compatibility)."""
    return BLUEPRINT_MANAGER.get_build_steps(blueprint_name)

def list_blueprints() -> List[str]:
    """List available blueprints."""
    return BLUEPRINT_MANAGER.list_available_blueprints()

def list_dse_blueprints() -> List[str]:
    """List blueprints that support design space exploration."""
    return BLUEPRINT_MANAGER.list_blueprints_with_dse()

def get_blueprint_info() -> List[Dict[str, Any]]:
    """Get detailed information about all blueprints."""
    return BLUEPRINT_MANAGER.get_blueprint_info()

def validate_blueprint(blueprint_name: str) -> List[str]:
    """Validate a blueprint and return any issues."""
    return BLUEPRINT_MANAGER.validate_blueprint(blueprint_name)

# Backward compatibility alias
BlueprintConfig = Blueprint