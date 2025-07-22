"""
Blueprint Parser - YAML to DesignSpace

This module parses blueprint YAML files and creates DesignSpace objects
with all plugins resolved from the registry.
"""

import os
import yaml
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type, Union
)

from .design_space import DesignSpace, GlobalConfig, OutputStage
from .execution_tree import TransformStage

if TYPE_CHECKING:
    from qonnx.transformation.base import Transformation


class BlueprintParser:
    """Parse blueprint YAML into DesignSpace with resolved plugins."""
    

    def parse(self, blueprint_path: str, model_path: str) -> DesignSpace:
        """
        Parse blueprint and create design space.
        
        Args:
            blueprint_path: Path to blueprint YAML file
            model_path: Path to ONNX model
            
        Returns:
            DesignSpace with all plugins resolved
        """
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Parse global config
        global_config = self._parse_global_config(data.get('global_config', {}))
        
        # Parse transform stages
        transform_stages = self._parse_transform_stages(
            data.get('design_space', {}).get('transforms', {})
        )
        
        # Parse kernels with backend resolution
        kernel_backends = self._parse_kernels(
            data.get('design_space', {}).get('kernels', [])
        )
        
        # Get build pipeline
        build_pipeline = data.get('build_pipeline', {}).get('steps', [])
        
        # Validate pipeline references
        for step in build_pipeline:
            # Extract stage name from various formats
            stage_name = None
            if isinstance(step, str) and step.startswith("{") and step.endswith("}"):
                stage_name = step[1:-1]
            elif isinstance(step, dict) and len(step) == 1:
                stage_name = next(iter(step.keys()))
            
            # Validate if it's a stage reference
            if stage_name and stage_name not in transform_stages:
                raise ValueError(
                    f"Pipeline references stage '{stage_name}' which is not defined. "
                    f"Available stages: {list(transform_stages.keys())}"
                )
        
        # Create design space
        design_space = DesignSpace(
            model_path=model_path,
            transform_stages=transform_stages,
            kernel_backends=kernel_backends,
            build_pipeline=build_pipeline,
            global_config=global_config
        )
        
        # Validate size constraints
        design_space.validate_size()
        
        return design_space
    
    def parse_with_inheritance(self, blueprint_path: str, model_path: str) -> DesignSpace:
        """
        Parse blueprint with inheritance support.
        
        Args:
            blueprint_path: Path to blueprint YAML file (may have 'extends')
            model_path: Path to ONNX model
            
        Returns:
            DesignSpace with all plugins resolved
        """
        blueprint_data = self._load_with_inheritance(blueprint_path)
        
        # Use regular parse but with loaded data
        # Extract just the parse logic
        # Parse global config
        global_config = self._parse_global_config(blueprint_data.get('global_config', {}))
        
        # Parse transform stages
        transform_stages = self._parse_transform_stages(
            blueprint_data.get('design_space', {}).get('transforms', {})
        )
        
        # Parse kernels with backend resolution
        kernel_backends = self._parse_kernels(
            blueprint_data.get('design_space', {}).get('kernels', [])
        )
        
        # Get build pipeline
        build_pipeline = blueprint_data.get('build_pipeline', {}).get('steps', [])
        
        # Validate pipeline references
        for step in build_pipeline:
            # Extract stage name from various formats
            stage_name = None
            if isinstance(step, str) and step.startswith("{") and step.endswith("}"):
                stage_name = step[1:-1]
            elif isinstance(step, dict) and len(step) == 1:
                stage_name = next(iter(step.keys()))
            
            # Validate if it's a stage reference
            if stage_name and stage_name not in transform_stages:
                raise ValueError(
                    f"Pipeline references stage '{stage_name}' which is not defined. "
                    f"Available stages: {list(transform_stages.keys())}"
                )
        
        # Create design space
        design_space = DesignSpace(
            model_path=model_path,
            transform_stages=transform_stages,
            kernel_backends=kernel_backends,
            build_pipeline=build_pipeline,
            global_config=global_config
        )
        
        # Validate size constraints
        design_space.validate_size()
        
        return design_space
    
    def _load_with_inheritance(self, blueprint_path: str) -> Dict[str, Any]:
        """
        Load blueprint and merge with parent if extends is specified.
        
        Args:
            blueprint_path: Path to blueprint YAML file
            
        Returns:
            Merged blueprint data
        """
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        # Handle inheritance
        if 'extends' in data:
            # Resolve parent path relative to child
            parent_path = os.path.join(
                os.path.dirname(blueprint_path), 
                data['extends']
            )
            parent_data = self._load_with_inheritance(parent_path)
            
            # Deep merge parent and child
            return self._deep_merge(parent_data, data)
        
        return data
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two dictionaries.
        
        Args:
            base: Base dictionary (parent blueprint)
            override: Override dictionary (child blueprint)
            
        Returns:
            Merged dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _parse_global_config(self, config_data: Dict[str, Any]) -> GlobalConfig:
        """Parse global configuration section."""
        global_config = GlobalConfig()
        
        # Map string to enum for output_stage
        if 'output_stage' in config_data:
            stage_str = config_data['output_stage']
            try:
                global_config.output_stage = OutputStage(stage_str)
            except ValueError:
                # Try to map old names to new ones
                stage_map = {
                    'stitched_ip': OutputStage.SYNTHESIZE_BITSTREAM,
                    'dataflow_graph': OutputStage.COMPILE_AND_PACKAGE,
                    'rtl': OutputStage.COMPILE_AND_PACKAGE
                }
                if stage_str in stage_map:
                    global_config.output_stage = stage_map[stage_str]
                else:
                    raise ValueError(f"Unknown output stage: {stage_str}")
        
        # Set other fields
        for field in ['working_directory', 'save_intermediate_models', 
                      'max_combinations', 'timeout_minutes']:
            if field in config_data:
                setattr(global_config, field, config_data[field])
        
        return global_config
    
    def _parse_transform_stages(self, transforms_data: Dict) -> Dict[str, Any]:
        """Parse transform stages section."""
        from .plugins.registry import get_registry
        registry = get_registry()
        transform_stages = {}
        
        for stage_name, stage_spec in transforms_data.items():
            # Ensure stage_spec is a list
            if not isinstance(stage_spec, list):
                stage_spec = [stage_spec]
            
            # Parse transform stage inline
            transform_steps = []
            
            for spec in stage_spec:
                # Normalize spec to always be a list
                specs = spec if isinstance(spec, list) else [spec]
                
                # Process all specs uniformly
                options = []
                for name in specs:
                    if name == "~" or name is None:
                        options.append(None)  # Skip option
                    else:
                        transform_class = registry.get_transform(name)
                        if not transform_class:
                            raise ValueError(f"Transform '{name}' not found in registry")
                        options.append(transform_class)
                
                transform_steps.append(options)
            
            transform_stages[stage_name] = TransformStage(stage_name, transform_steps)
        
        return transform_stages
    
    def _parse_kernels(self, kernels_data: list) -> list:
        """Parse kernels section."""
        from .plugins.registry import get_registry
        registry = get_registry()
        kernel_backends = []
        
        for spec in kernels_data:
            # Parse spec format
            if isinstance(spec, str):
                kernel_name = spec
                backend_names = registry.list_backends_by_kernel(kernel_name)
            elif isinstance(spec, dict) and len(spec) == 1:
                kernel_name, backend_specs = next(iter(spec.items()))
                backend_names = backend_specs if isinstance(backend_specs, list) else [backend_specs]
            else:
                raise ValueError(f"Invalid kernel spec: {spec}")
            
            if not backend_names:
                raise ValueError(f"No backends found for kernel '{kernel_name}'")
            
            # Resolve and validate backends
            available = registry.list_backends_by_kernel(kernel_name)
            backend_classes = []
            
            for name in backend_names:
                if backend_class := registry.get_backend(name):
                    if name not in available:
                        raise ValueError(
                            f"Backend '{name}' does not support kernel '{kernel_name}'. "
                            f"Available backends: {available}"
                        )
                    backend_classes.append(backend_class)
                else:
                    raise ValueError(f"Backend '{name}' not found in registry")
            
            kernel_backends.append((kernel_name, backend_classes))
        
        return kernel_backends