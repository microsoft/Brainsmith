"""
Blueprint Plugin Manager

Provides blueprint-driven plugin loading for selective plugin discovery
and loading based on blueprint specifications.
"""

import logging
import yaml
from typing import Dict, List, Optional, Any, Set
from pathlib import Path

from .manager import get_plugin_manager
from .data_models import PluginInfo

logger = logging.getLogger(__name__)


class BlueprintPluginManager:
    """
    Blueprint-driven plugin manager for selective plugin loading.
    
    This manager provides efficient plugin loading based on blueprint
    specifications, avoiding the overhead of loading all available plugins.
    """
    
    def __init__(self):
        self._base_manager = get_plugin_manager()
        self._loaded_blueprints: Dict[str, Dict[str, List[PluginInfo]]] = {}
    
    def load_for_blueprint(self, blueprint_path: str) -> Dict[str, List[PluginInfo]]:
        """
        Load plugins specified in a blueprint file.
        
        Args:
            blueprint_path: Path to blueprint YAML file
            
        Returns:
            Dict mapping plugin types to lists of loaded plugins
        """
        # Check cache first
        if blueprint_path in self._loaded_blueprints:
            logger.debug(f"Using cached plugins for blueprint: {blueprint_path}")
            return self._loaded_blueprints[blueprint_path]
        
        # Parse blueprint requirements
        requirements = self._parse_blueprint_requirements(blueprint_path)
        
        # Load plugins using base manager
        loaded_plugins = self._base_manager.load_for_blueprint(requirements)
        
        # Cache results
        self._loaded_blueprints[blueprint_path] = loaded_plugins
        
        logger.info(f"Loaded {sum(len(plugins) for plugins in loaded_plugins.values())} plugins for blueprint: {blueprint_path}")
        return loaded_plugins
    
    def load_for_requirements(self, requirements: Dict[str, List[str]]) -> Dict[str, List[PluginInfo]]:
        """
        Load plugins based on explicit requirements dictionary.
        
        Args:
            requirements: Dict with plugin types as keys and plugin names as values
            
        Returns:
            Dict mapping plugin types to lists of loaded plugins
        """
        return self._base_manager.load_for_blueprint(requirements)
    
    def _parse_blueprint_requirements(self, blueprint_path: str) -> Dict[str, List[str]]:
        """
        Parse blueprint YAML to extract plugin requirements.
        
        Args:
            blueprint_path: Path to blueprint file
            
        Returns:
            Dict with plugin types as keys and plugin names as values
        """
        try:
            with open(blueprint_path, 'r') as f:
                blueprint = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load blueprint {blueprint_path}: {e}")
            return {}
        
        requirements = {
            'transforms': [],
            'kernels': [],
            'backends': [],
            'steps': [],
            'kernel_inference': []
        }
        
        # Extract from hw_compiler section
        hw_compiler = blueprint.get('hw_compiler', {})
        
        # Parse kernels
        kernels = hw_compiler.get('kernels', [])
        for kernel_spec in kernels:
            kernel_names = self._parse_kernel_specification(kernel_spec)
            requirements['kernels'].extend(kernel_names)
        
        # Parse transforms
        transforms = hw_compiler.get('transforms', [])
        for transform_spec in transforms:
            transform_name = self._parse_transform_specification(transform_spec)
            if transform_name:
                requirements['transforms'].append(transform_name)
        
        # Parse phased transforms
        transforms_phased = hw_compiler.get('transforms_phased', {})
        for phase, phase_transforms in transforms_phased.items():
            for transform_spec in phase_transforms:
                transform_name = self._parse_transform_specification(transform_spec)
                if transform_name:
                    requirements['transforms'].append(transform_name)
        
        # Parse backends (usually derived from kernels, but can be explicit)
        backends = hw_compiler.get('backends', [])
        for backend_spec in backends:
            backend_name = self._parse_backend_specification(backend_spec)
            if backend_name:
                requirements['backends'].append(backend_name)
        
        # Parse steps
        steps = hw_compiler.get('steps', [])
        for step_spec in steps:
            step_name = self._parse_step_specification(step_spec)
            if step_name:
                requirements['steps'].append(step_name)
        
        # Remove duplicates and filter empty
        for key in requirements:
            requirements[key] = list(set(filter(None, requirements[key])))
        
        logger.debug(f"Blueprint requirements: {requirements}")
        return requirements
    
    def _parse_kernel_specification(self, kernel_spec: Any) -> List[str]:
        """Parse various kernel specification formats."""
        if isinstance(kernel_spec, str):
            # Simple kernel name, strip optional prefix
            return [kernel_spec.lstrip("~")]
        
        elif isinstance(kernel_spec, dict):
            # Dict format: {"kernel": "name", "backends": [...]}
            kernel_name = kernel_spec.get("kernel", "").lstrip("~")
            return [kernel_name] if kernel_name else []
        
        elif isinstance(kernel_spec, list):
            # Mutually exclusive group - extract all alternatives
            names = []
            for item in kernel_spec:
                names.extend(self._parse_kernel_specification(item))
            return names
        
        return []
    
    def _parse_transform_specification(self, transform_spec: Any) -> Optional[str]:
        """Parse transform specification."""
        if isinstance(transform_spec, str):
            return transform_spec.lstrip("~")
        elif isinstance(transform_spec, dict):
            return transform_spec.get("transform", "").lstrip("~")
        return None
    
    def _parse_backend_specification(self, backend_spec: Any) -> Optional[str]:
        """Parse backend specification."""
        if isinstance(backend_spec, str):
            return backend_spec.lstrip("~")
        elif isinstance(backend_spec, dict):
            return backend_spec.get("backend", "").lstrip("~")
        return None
    
    def _parse_step_specification(self, step_spec: Any) -> Optional[str]:
        """Parse step specification."""
        if isinstance(step_spec, str):
            return step_spec.lstrip("~")
        elif isinstance(step_spec, dict):
            return step_spec.get("step", "").lstrip("~")
        return None
    
    def clear_cache(self) -> None:
        """Clear cached blueprint results."""
        self._loaded_blueprints.clear()
        logger.debug("Cleared blueprint plugin cache")
    
    def get_loaded_blueprints(self) -> List[str]:
        """Get list of currently loaded blueprint paths."""
        return list(self._loaded_blueprints.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about loaded blueprints and plugins."""
        stats = {
            'loaded_blueprints': len(self._loaded_blueprints),
            'blueprint_details': {}
        }
        
        for blueprint_path, plugins in self._loaded_blueprints.items():
            total_plugins = sum(len(plugin_list) for plugin_list in plugins.values())
            stats['blueprint_details'][blueprint_path] = {
                'total_plugins': total_plugins,
                'by_type': {ptype: len(plist) for ptype, plist in plugins.items() if plist}
            }
        
        return stats


# Global instance for convenience
_blueprint_manager: Optional[BlueprintPluginManager] = None


def get_blueprint_manager() -> BlueprintPluginManager:
    """Get the global blueprint plugin manager instance."""
    global _blueprint_manager
    if _blueprint_manager is None:
        _blueprint_manager = BlueprintPluginManager()
    return _blueprint_manager


def load_blueprint_plugins(blueprint_path: str) -> Dict[str, List[PluginInfo]]:
    """Convenience function to load plugins for a blueprint."""
    manager = get_blueprint_manager()
    return manager.load_for_blueprint(blueprint_path)