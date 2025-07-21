"""
Blueprint parser for v4.0 format.

This module handles parsing Blueprint YAML files into structured DesignSpace objects.
Supports blueprint inheritance and dynamic transform stage registration.
"""

import os
import yaml
import logging
from typing import Dict, List, Any, Union, Tuple
from pathlib import Path

from .data_structures import (
    DesignSpace,
    HWCompilerSpace,
    GlobalConfig,
    OutputStage,
)
from .exceptions import BlueprintParseError

logger = logging.getLogger(__name__)


def load_blueprint(blueprint_path: str) -> Dict[str, Any]:
    """Load a blueprint YAML file."""
    try:
        with open(blueprint_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise BlueprintParseError(f"Failed to load blueprint {blueprint_path}: {e}")


class BlueprintParser:
    """
    Parse Blueprint YAML files into DesignSpace objects.
    
    This parser handles the V4 blueprint format with support for:
    - Blueprint inheritance via 'extends'
    - Direct kernel format: ["KernelName", ["backend1", "backend2"]]
    - Transform stages with dynamic registration
    - Direct limits on design space
    """
    
    def __init__(self):
        """Initialize parser."""
        pass
    
    def parse(self, blueprint_data: Dict[str, Any], model_path: str) -> DesignSpace:
        """
        Parse blueprint data into a DesignSpace object.
        
        Args:
            blueprint_data: Parsed YAML data from blueprint file
            model_path: Path to the ONNX model
            
        Returns:
            Fully constructed DesignSpace object
            
        Raises:
            BlueprintParseError: If parsing fails
        """
        try:
            # For v4.0, we expect a different structure
            # The blueprint_data passed in is just the initial load,
            # but we need the path for inheritance
            # This is a limitation - for now assume no inheritance
            
            # Parse transforms including custom stages
            all_transforms = self._parse_all_transforms(blueprint_data)
            
            # Register transform stages as steps
            self._register_transform_stages(all_transforms)
            
            # Parse and expand build pipeline
            build_steps = self._parse_build_pipeline(blueprint_data, all_transforms)
            
            # Parse kernels (simple format)
            kernels = self._parse_kernels(blueprint_data)
            
            # Create HWCompilerSpace
            hw_compiler_space = HWCompilerSpace(
                kernels=kernels,
                transforms=all_transforms,
                build_steps=build_steps,
                config_flags=blueprint_data.get("finn_config", {})
            )
            
            # Create GlobalConfig
            global_config = GlobalConfig(
                output_stage=OutputStage(blueprint_data.get("output_stage", "rtl")),
                working_directory=blueprint_data.get("working_directory", "./dse_work"),
                cache_results=blueprint_data.get("preserve_intermediate_models", False),
                save_artifacts=blueprint_data.get("preserve_intermediate_models", False),
                start_step=blueprint_data.get("start_step"),
                stop_step=blueprint_data.get("stop_step")
            )
            
            # Create DesignSpace with direct limits
            design_space = DesignSpace(
                model_path=model_path,
                hw_compiler_space=hw_compiler_space,
                global_config=global_config,
                max_combinations=blueprint_data.get("max_combinations", 100000),
                timeout_minutes=blueprint_data.get("timeout_minutes", 60)
            )
            
            return design_space
            
        except Exception as e:
            if isinstance(e, BlueprintParseError):
                raise
            raise BlueprintParseError(f"Failed to parse blueprint: {str(e)}")
    
    def parse_with_inheritance(self, blueprint_path: str, model_path: str) -> DesignSpace:
        """Parse blueprint with inheritance support."""
        blueprint = self._load_blueprint_with_inheritance(blueprint_path)
        return self.parse(blueprint, model_path)
    
    def _load_blueprint_with_inheritance(self, blueprint_path: str) -> Dict[str, Any]:
        """Load blueprint with inheritance support."""
        with open(blueprint_path, 'r') as f:
            blueprint = yaml.safe_load(f)
        
        # Handle inheritance
        if "extends" in blueprint:
            # Resolve parent path relative to child
            parent_path = self._resolve_path(blueprint["extends"], blueprint_path)
            parent = self._load_blueprint_with_inheritance(parent_path)
            
            # Deep merge parent and child
            blueprint = self._deep_merge(parent, blueprint)
        
        return blueprint
    
    def _resolve_path(self, path: str, relative_to: str) -> str:
        """Resolve a path relative to another file."""
        base_dir = os.path.dirname(relative_to)
        return os.path.join(base_dir, path)
    
    def _deep_merge(self, parent: Dict, child: Dict) -> Dict:
        """Deep merge child blueprint into parent."""
        result = parent.copy()
        
        for key, value in child.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _parse_all_transforms(self, blueprint: Dict) -> Dict[str, List]:
        """Parse all transforms including custom stages."""
        # Start with standard stages
        transforms = blueprint.get("design_space", {}).get("transforms", {})
        
        # Add custom stages
        if "custom_stages" in blueprint:
            for stage_name, config in blueprint["custom_stages"].items():
                transforms[stage_name] = config["transforms"]
        
        return transforms
    
    def _register_transform_stages(self, all_transforms: Dict[str, List]):
        """Register each transform stage as a build step."""
        try:
            from brainsmith.core.plugins import register_step, get_transform
        except ImportError:
            logger.warning("Plugin system not available, skipping transform registration")
            return
        
        for stage_name, stage_transforms in all_transforms.items():
            step_name = f"brainsmith_stage_{stage_name}"
            
            def make_stage_step(stage, transforms):
                def step_func(model, cfg):
                    # Get selected transforms or use all
                    if hasattr(cfg, 'transform_selections'):
                        selected = cfg.transform_selections.get(stage, transforms)
                    else:
                        selected = transforms
                    
                    # Apply each transform
                    for t in selected:
                        if t and t != "~":
                            transform = get_transform(t)
                            if transform:
                                model = model.transform(transform())
                    
                    return model
                
                # Set function name for debugging
                step_func.__name__ = f"stage_{stage}"
                return step_func
            
            # Register the step
            try:
                register_step(step_name, make_stage_step(stage_name, stage_transforms))
            except Exception as e:
                logger.warning(f"Failed to register stage {stage_name}: {e}")
    
    def _parse_build_pipeline(self, blueprint: Dict, all_transforms: Dict) -> List[str]:
        """Parse build pipeline with stage expansion."""
        steps = blueprint.get("build_pipeline", {}).get("steps", [])
        custom_stages = blueprint.get("custom_stages", {})
        
        # Track positions for custom stages
        insertions = []
        for stage_name, config in custom_stages.items():
            if "after" in config:
                insertions.append((stage_name, "after", config["after"]))
            elif "before" in config:
                insertions.append((stage_name, "before", config["before"]))
        
        # Expand steps
        expanded_steps = []
        for step in steps:
            # Add the step
            expanded_steps.append(step)
            
            # Check for insertions after this step
            step_stage = step.strip("{}")
            for stage_name, position, target in insertions:
                if position == "after" and target == step_stage:
                    expanded_steps.append(f"{{{stage_name}}}")
        
        # Replace placeholders with actual step names
        final_steps = []
        for step in expanded_steps:
            if step.startswith("{") and step.endswith("}"):
                stage = step[1:-1]
                if stage in all_transforms:
                    final_steps.append(f"brainsmith_stage_{stage}")
            else:
                final_steps.append(step)
        
        return final_steps
    
    def _parse_kernels(self, blueprint: Dict) -> List[Tuple[str, Any]]:
        """Parse kernel specifications."""
        kernels = []
        kernel_specs = blueprint.get("design_space", {}).get("kernels", [])
        
        for spec in kernel_specs:
            if isinstance(spec, list) and len(spec) == 2:
                kernel_name, backends = spec
                kernels.append((kernel_name, backends))
            else:
                raise BlueprintParseError(
                    f"Invalid kernel specification: {spec}. "
                    f"Expected format: ['KernelName', ['backend1', 'backend2']]"
                )
        
        return kernels