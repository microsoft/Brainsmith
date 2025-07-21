"""
Blueprint Parser - YAML to DesignSpace

This module parses blueprint YAML files and creates DesignSpace objects
with all plugins resolved from the registry.
"""

import yaml
from typing import Dict, Any

from .design_space import DesignSpace, GlobalConfig, OutputStage
from .resolution import parse_transform_stage, resolve_kernel_spec, validate_pipeline_steps


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
        validate_pipeline_steps(build_pipeline, transform_stages)
        
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
        transform_stages = {}
        
        for stage_name, stage_spec in transforms_data.items():
            # Ensure stage_spec is a list
            if not isinstance(stage_spec, list):
                stage_spec = [stage_spec]
            
            transform_stages[stage_name] = parse_transform_stage(stage_name, stage_spec)
        
        return transform_stages
    
    def _parse_kernels(self, kernels_data: list) -> list:
        """Parse kernels section."""
        kernel_backends = []
        
        for spec in kernels_data:
            kernel_name, backend_classes = resolve_kernel_spec(spec)
            kernel_backends.append((kernel_name, backend_classes))
        
        return kernel_backends