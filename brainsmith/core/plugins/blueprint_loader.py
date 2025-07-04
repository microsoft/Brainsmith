"""
Blueprint Plugin Loader - Perfect Code Implementation

Load only required plugins based on blueprint specification.
Optimizes memory usage and improves startup performance.
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any, Union, List, Optional

logger = logging.getLogger(__name__)


class BlueprintPluginLoader:
    """
    Load plugins based on blueprint requirements.
    
    Perfect Code approach: Create subset registries containing only
    required plugins for maximum performance.
    """
    
    def __init__(self, registry=None):
        """Initialize with main registry reference."""
        if registry is None:
            from .registry import get_registry
            registry = get_registry()
        
        self.registry = registry
        self._blueprint_cache = {}
    
    def load_from_blueprint(self, blueprint_path_or_requirements: Union[str, Path, Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Load plugins based on blueprint specification.
        
        Args:
            blueprint_path_or_requirements: Path to YAML file or parsed requirements dict
            
        Returns:
            Dict mapping plugin types to lists of loaded plugin names
        """
        # Parse requirements
        if isinstance(blueprint_path_or_requirements, (str, Path)):
            requirements = self._parse_blueprint_file(blueprint_path_or_requirements)
        else:
            requirements = blueprint_path_or_requirements
        
        return self._load_plugins_for_requirements(requirements)
    
    def _parse_blueprint_file(self, blueprint_path: Union[str, Path]) -> Dict[str, Any]:
        """Parse blueprint YAML file to extract plugin requirements."""
        blueprint_path = Path(blueprint_path)
        
        try:
            with open(blueprint_path, 'r') as f:
                blueprint = yaml.safe_load(f)
        except Exception as e:
            logger.error(f"Failed to load blueprint {blueprint_path}: {e}")
            return {}
        
        return self._extract_plugin_requirements(blueprint)
    
    def _extract_plugin_requirements(self, blueprint: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract plugin requirements from blueprint structure.
        
        Perfect Code approach: Simple, direct extraction without complex nesting.
        """
        requirements = {
            'transforms': [],
            'kernels': [],
            'backends': [],
            'steps': []
        }
        
        hw_compiler = blueprint.get('hw_compiler', {})
        
        # Extract transforms
        transforms = hw_compiler.get('transforms', {})
        if isinstance(transforms, dict):
            # Staged transforms: {stage: [names]}
            for transform_list in transforms.values():
                if isinstance(transform_list, list):
                    requirements['transforms'].extend(
                        self._normalize_name(t) for t in transform_list
                    )
        elif isinstance(transforms, list):
            # Flat list of transforms
            requirements['transforms'].extend(
                self._normalize_name(t) for t in transforms
            )
        
        # Extract kernels
        kernels = hw_compiler.get('kernels', [])
        if isinstance(kernels, list):
            requirements['kernels'].extend(
                self._normalize_name(k) for k in kernels if k
            )
        
        # Extract backends (simplified)
        backends = hw_compiler.get('backends', [])
        if isinstance(backends, list):
            for backend_spec in backends:
                if isinstance(backend_spec, str):
                    # Simple backend name
                    requirements['backends'].append(backend_spec)
                elif isinstance(backend_spec, dict):
                    # Backend with metadata
                    name = backend_spec.get('name')
                    if name:
                        requirements['backends'].append(name)
        
        # Extract steps
        steps = hw_compiler.get('steps', [])
        if isinstance(steps, list):
            requirements['steps'].extend(
                self._normalize_name(s) for s in steps if s
            )
        
        return requirements
    
    def _normalize_name(self, spec: Any) -> Optional[str]:
        """Extract name from various specification formats."""
        if isinstance(spec, str):
            return spec.strip('~')  # Remove optional prefix
        elif isinstance(spec, dict):
            return spec.get('name', '').strip('~')
        elif isinstance(spec, (list, tuple)) and spec:
            # Take first element
            return self._normalize_name(spec[0])
        return None
    
    def _load_plugins_for_requirements(self, requirements: Dict[str, Any]) -> Dict[str, List[str]]:
        """Load plugins based on requirements using direct registry lookups."""
        loaded = {
            'transforms': [],
            'kernels': [],
            'backends': [],
            'steps': []
        }
        
        # Load transforms
        for name in requirements.get('transforms', []):
            if name in self.registry.transforms:
                loaded['transforms'].append(name)
            else:
                logger.warning(f"Transform '{name}' not found")
        
        # Load kernels
        for name in requirements.get('kernels', []):
            if name in self.registry.kernels:
                loaded['kernels'].append(name)
            else:
                logger.warning(f"Kernel '{name}' not found")
        
        # Load backends
        for name in requirements.get('backends', []):
            if name in self.registry.backends:
                loaded['backends'].append(name)
            else:
                logger.warning(f"Backend '{name}' not found")
        
        # Load steps  
        for name in requirements.get('steps', []):
            if name in self.registry.steps:
                loaded['steps'].append(name)
            else:
                logger.warning(f"Step '{name}' not found")
        
        return loaded
    
    def create_optimized_collections(self, blueprint_path_or_requirements: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Create collections with only blueprint-required plugins.
        
        Perfect Code approach: Create subset registry containing only required plugins
        for maximum performance and minimal memory usage.
        """
        from .plugin_collections import create_collections
        from .decorators import filter_plugin_metadata
        
        # Parse requirements
        if isinstance(blueprint_path_or_requirements, (str, Path)):
            requirements = self._parse_blueprint_file(blueprint_path_or_requirements)
        else:
            requirements = blueprint_path_or_requirements
        
        # Load required plugins
        loaded = self._load_plugins_for_requirements(requirements)
        
        # Create subset registry with only loaded plugins
        from .registry import BrainsmithPluginRegistry
        
        subset_registry = BrainsmithPluginRegistry()
        
        # Copy only required plugins
        for name in loaded['transforms']:
            transform_cls = self.registry.transforms[name]
            metadata = self.registry.get_plugin_metadata(name)
            subset_registry.register_transform(
                name, 
                transform_cls,
                stage=metadata.get('stage'),
                framework=metadata.get('framework', 'brainsmith'),
                **filter_plugin_metadata(metadata, 'transform')
            )
        
        for name in loaded['kernels']:
            kernel_cls = self.registry.kernels[name]
            metadata = self.registry.get_plugin_metadata(name)
            subset_registry.register_kernel(
                name,
                kernel_cls,
                framework=metadata.get('framework', 'brainsmith'),
                **filter_plugin_metadata(metadata, 'kernel')
            )
        
        for name in loaded['backends']:
            backend_cls = self.registry.backends[name]
            metadata = self.registry.get_plugin_metadata(name)
            subset_registry.register_backend(
                name,
                backend_cls,
                kernel=metadata.get('kernel'),
                framework=metadata.get('framework', 'brainsmith'),
                **filter_plugin_metadata(metadata, 'backend')
            )
        
        for name in loaded['steps']:
            step_cls = self.registry.steps[name]
            metadata = self.registry.get_plugin_metadata(name)
            subset_registry.register_step(
                name,
                step_cls,
                category=metadata.get('category'),
                framework=metadata.get('framework', 'brainsmith'),
                **filter_plugin_metadata(metadata, 'step')
            )
        
        # Create collections from subset registry
        collections = create_collections(subset_registry)
        
        logger.info(f"Created optimized collections with:")
        logger.info(f"  - {len(loaded['transforms'])} transforms")
        logger.info(f"  - {len(loaded['kernels'])} kernels")
        logger.info(f"  - {len(loaded['backends'])} backends")
        logger.info(f"  - {len(loaded['steps'])} steps")
        
        return collections


def load_blueprint_plugins(blueprint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Convenience function to load plugins from blueprint.
    
    Args:
        blueprint_path: Path to blueprint YAML file
        
    Returns:
        Dict of plugin collections
    """
    loader = BlueprintPluginLoader()
    return loader.create_optimized_collections(blueprint_path)