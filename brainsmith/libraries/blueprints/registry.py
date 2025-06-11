"""
Blueprint Libraries Registry System

Auto-discovery and management of blueprint YAML collections.
Provides registration, caching, and lookup functionality for blueprint templates
in the libraries directory structure.

BREAKING CHANGE: Now uses unified BaseRegistry interface with standardized method names.
"""

import os
import yaml
import logging
from typing import Dict, List, Optional, Set, Any
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from brainsmith.core.registry import BaseRegistry, ComponentInfo

logger = logging.getLogger(__name__)


class BlueprintCategory(Enum):
    """Categories of blueprint templates."""
    BASIC = "basic"
    ADVANCED = "advanced"
    EXPERIMENTAL = "experimental"
    CUSTOM = "custom"


@dataclass
class BlueprintInfo:
    """Information about a discovered blueprint."""
    name: str
    category: BlueprintCategory
    file_path: str
    version: str = "1.0"
    description: str = ""
    model_type: str = "unknown"
    target_platform: str = "unknown"
    parameters: Dict[str, Any] = None
    targets: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
        if self.targets is None:
            self.targets = {}


class BlueprintLibraryRegistry(BaseRegistry[BlueprintInfo]):
    """Registry for auto-discovery and management of blueprint libraries."""
    
    def __init__(self, search_dirs: Optional[List[str]] = None, config_manager=None):
        """
        Initialize blueprint library registry.
        
        Args:
            search_dirs: List of directories to search for blueprints.
                        If None, uses default blueprint directories.
            config_manager: Optional configuration manager.
        """
        super().__init__(search_dirs, config_manager)
        # For backward compatibility, maintain blueprint_cache reference
        self.blueprint_cache = self._cache
        self.metadata_cache = self._metadata_cache
    
    def discover_components(self, rescan: bool = False) -> Dict[str, BlueprintInfo]:
        """
        Discover all available blueprint YAML files.
        
        Args:
            rescan: Force rescan even if cache exists
            
        Returns:
            Dictionary mapping blueprint names to BlueprintInfo objects
        """
        if not rescan and self._cache:
            return self._cache
        
        discovered = {}
        
        for blueprint_dir in self.search_dirs:
            if not os.path.exists(blueprint_dir):
                self._log_warning(f"Blueprint directory not found: {blueprint_dir}")
                continue
            
            # Look for category directories
            for category_name in os.listdir(blueprint_dir):
                category_path = os.path.join(blueprint_dir, category_name)
                
                if not os.path.isdir(category_path):
                    continue
                
                # Try to map directory name to category enum
                try:
                    category = BlueprintCategory(category_name.lower())
                except ValueError:
                    # Unknown category, treat as custom
                    category = BlueprintCategory.CUSTOM
                
                # Look for YAML files in category directory
                for item in os.listdir(category_path):
                    if item.endswith(('.yaml', '.yml')):
                        file_path = os.path.join(category_path, item)
                        blueprint_name = Path(item).stem
                        
                        try:
                            blueprint_info = self._load_blueprint_info(file_path, category)
                            blueprint_info.name = blueprint_name
                            discovered[blueprint_name] = blueprint_info
                            logger.debug(f"Discovered blueprint: {blueprint_name} ({category.value})")
                            
                        except Exception as e:
                            self._log_warning(f"Failed to load blueprint {file_path}: {e}")
        
        # Cache the results
        self._cache = discovered
        self.blueprint_cache = self._cache  # Maintain backward compatibility reference
        
        self._log_info(f"Discovered {len(discovered)} blueprint templates")
        return discovered
    
    def find_components_by_type(self, category: BlueprintCategory) -> List[BlueprintInfo]:
        """
        Find blueprints by category.
        
        Args:
            category: Category to search for
            
        Returns:
            List of matching BlueprintInfo objects
        """
        blueprints = self.discover_components()
        matches = []
        
        for blueprint in blueprints.values():
            if blueprint.category == category:
                matches.append(blueprint)
        
        return matches
    
    def find_blueprints_by_model_type(self, model_type: str) -> List[BlueprintInfo]:
        """
        Find blueprints by model type.
        
        Args:
            model_type: Model type to search for
            
        Returns:
            List of matching BlueprintInfo objects
        """
        blueprints = self.discover_components()
        matches = []
        
        for blueprint in blueprints.values():
            if blueprint.model_type.lower() == model_type.lower():
                matches.append(blueprint)
        
        return matches
    
    def find_blueprints_by_platform(self, platform: str) -> List[BlueprintInfo]:
        """
        Find blueprints by target platform.
        
        Args:
            platform: Target platform to search for
            
        Returns:
            List of matching BlueprintInfo objects
        """
        blueprints = self.discover_components()
        matches = []
        
        for blueprint in blueprints.values():
            if platform.lower() in blueprint.target_platform.lower():
                matches.append(blueprint)
        
        return matches
    
    def list_categories(self) -> Set[BlueprintCategory]:
        """Get set of all available categories."""
        blueprints = self.discover_components()
        return {blueprint.category for blueprint in blueprints.values()}
    
    def list_model_types(self) -> Set[str]:
        """Get set of all available model types."""
        blueprints = self.discover_components()
        return {blueprint.model_type for blueprint in blueprints.values()}
    
    def list_platforms(self) -> Set[str]:
        """Get set of all available target platforms."""
        blueprints = self.discover_components()
        return {blueprint.target_platform for blueprint in blueprints.values()}
    
    def load_blueprint_yaml(self, blueprint_name: str) -> Optional[Dict[str, Any]]:
        """
        Load the full YAML content of a blueprint.
        
        Args:
            blueprint_name: Name of the blueprint
            
        Returns:
            Blueprint YAML content as dictionary or None if not found
        """
        blueprint = self.get_component(blueprint_name)
        if not blueprint:
            return None
        
        try:
            with open(blueprint.file_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load blueprint YAML {blueprint.file_path}: {e}")
            return None
    
    def _get_default_dirs(self) -> List[str]:
        """Get default search directories for blueprint registry."""
        current_dir = Path(__file__).parent
        return [str(current_dir)]
    
    def _extract_info(self, component: BlueprintInfo) -> Dict[str, Any]:
        """Extract standardized info from blueprint component."""
        return {
            'name': component.name,
            'type': 'blueprint',
            'category': component.category.value,
            'version': component.version,
            'description': component.description,
            'model_type': component.model_type,
            'target_platform': component.target_platform,
            'file_path': component.file_path,
            'parameter_count': len(component.parameters),
            'has_targets': bool(component.targets)
        }
    
    def _validate_component_implementation(self, component: BlueprintInfo) -> tuple[bool, List[str]]:
        """Blueprint-specific validation logic."""
        errors = []
        
        # Check file exists
        if not os.path.exists(component.file_path):
            errors.append(f"Blueprint file not found: {component.file_path}")
            return False, errors
        
        # Load and validate YAML structure
        try:
            yaml_content = self.load_blueprint_yaml(component.name)
            if not yaml_content:
                errors.append("Could not load blueprint YAML content")
                return False, errors
            
            # Check required fields
            if not yaml_content.get('name'):
                errors.append("Blueprint name is required")
            
            if not yaml_content.get('description'):
                errors.append("Blueprint description is required")
            
            # Validate parameters structure if present
            if 'parameters' in yaml_content:
                params = yaml_content['parameters']
                if not isinstance(params, dict):
                    errors.append("Parameters must be a dictionary")
                else:
                    for param_name, param_config in params.items():
                        if isinstance(param_config, dict):
                            # Check for valid parameter configuration
                            if not any(key in param_config for key in ['range', 'values', 'default']):
                                errors.append(f"Parameter '{param_name}' missing range, values, or default")
            
        except yaml.YAMLError as e:
            errors.append(f"Invalid YAML syntax: {e}")
        except Exception as e:
            errors.append(f"Validation error: {e}")
        
        return len(errors) == 0, errors
    
    def _load_blueprint_info(self, file_path: str, category: BlueprintCategory) -> BlueprintInfo:
        """
        Load blueprint information from a YAML file.
        
        Args:
            file_path: Path to the blueprint YAML file
            category: Category of the blueprint
            
        Returns:
            BlueprintInfo object
        """
        with open(file_path, 'r') as f:
            yaml_content = yaml.safe_load(f)
        
        return BlueprintInfo(
            name=yaml_content.get('name', ''),
            category=category,
            file_path=file_path,
            version=yaml_content.get('version', '1.0'),
            description=yaml_content.get('description', ''),
            model_type=yaml_content.get('model_type', 'unknown'),
            target_platform=yaml_content.get('target_platform', 'unknown'),
            parameters=yaml_content.get('parameters', {}),
            targets=yaml_content.get('targets', {})
        )


# Global registry instance
_global_registry = None


def get_blueprint_library_registry() -> BlueprintLibraryRegistry:
    """Get the global blueprint library registry instance."""
    global _global_registry
    if _global_registry is None:
        _global_registry = BlueprintLibraryRegistry()
    return _global_registry


# BREAKING CHANGE: Updated convenience functions to use new unified interface
def discover_all_blueprints(rescan: bool = False) -> Dict[str, BlueprintInfo]:
    """
    Discover all available blueprint templates.
    
    Args:
        rescan: Force rescan even if cache exists
        
    Returns:
        Dictionary mapping blueprint names to BlueprintInfo objects
    """
    registry = get_blueprint_library_registry()
    return registry.discover_components(rescan)


def get_blueprint_by_name(blueprint_name: str) -> Optional[BlueprintInfo]:
    """
    Get a blueprint by name.
    
    Args:
        blueprint_name: Name of the blueprint
        
    Returns:
        BlueprintInfo object or None if not found
    """
    registry = get_blueprint_library_registry()
    return registry.get_component(blueprint_name)


def find_blueprints_by_category(category: BlueprintCategory) -> List[BlueprintInfo]:
    """
    Find all blueprints in a specific category.
    
    Args:
        category: Category to search for
        
    Returns:
        List of matching BlueprintInfo objects
    """
    registry = get_blueprint_library_registry()
    return registry.find_components_by_type(category)


def list_available_blueprints() -> List[str]:
    """
    Get list of all available blueprint names.
    
    Returns:
        List of blueprint names
    """
    registry = get_blueprint_library_registry()
    return registry.list_component_names()


def refresh_blueprint_library_registry():
    """Refresh the blueprint library registry cache."""
    registry = get_blueprint_library_registry()
    registry.refresh_cache()