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
from .execution_tree import TransformStage, ExecutionNode
from .explorer.utils import StageWrapperFactory
from .plugins.registry import BrainsmithPluginRegistry

if TYPE_CHECKING:
    from qonnx.transformation.base import Transformation


class BlueprintParser:
    """Parse blueprint YAML into DesignSpace with resolved plugins."""
    

    
    def parse(self, blueprint_path: str, model_path: str) -> Tuple[DesignSpace, ExecutionNode]:
        """
        Parse blueprint with inheritance support.
        
        Args:
            blueprint_path: Path to blueprint YAML file (may have 'extends')
            model_path: Path to ONNX model
            
        Returns:
            Tuple of (DesignSpace, ExecutionNode root) with all plugins resolved
        """
        blueprint_data = self._load_with_inheritance(blueprint_path)
        
        # Extract flat config and smithy mappings
        flat_config, finn_mappings = self._extract_flat_config(blueprint_data)
        
        # Parse global config from flat parameters
        global_config = self._parse_global_config(flat_config)
        
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
        
        # Extract FINN config and merge with smart mappings
        finn_config = blueprint_data.get('finn_config', {}).copy()
        finn_config.update(finn_mappings)
        
        # Validate required FINN fields only when needed for synthesis
        if global_config.output_stage != OutputStage.GENERATE_REPORTS:
            if 'synth_clk_period_ns' not in finn_config:
                raise ValueError("Hardware synthesis requires synth_clk_period_ns (or target_clk)")
            if 'board' not in finn_config:
                raise ValueError("Hardware synthesis requires board (or platform)")
        
        # Create design space
        design_space = DesignSpace(
            model_path=model_path,
            transform_stages=transform_stages,
            kernel_backends=kernel_backends,
            build_pipeline=build_pipeline,
            global_config=global_config,
            finn_config=finn_config
        )
        
        # Validate size constraints
        design_space.validate_size()
        
        # Build execution tree with pre-computed wrappers
        registry = BrainsmithPluginRegistry()
        wrapper_factory = StageWrapperFactory(registry)
        tree = self._build_execution_tree(design_space, wrapper_factory)
        
        return design_space, tree
    
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
    
    def _parse_time_with_units(self, value: Union[str, float, int]) -> float:
        """Parse time value with unit suffix to nanoseconds.
        
        Supports: ns, us, ms, ps
        Default unit is ns if no suffix.
        
        Examples:
            "5" -> 5.0
            "5ns" -> 5.0
            "5000ps" -> 5.0
            "0.005us" -> 5.0
            "0.000005ms" -> 5.0
        """
        if isinstance(value, (int, float)):
            return float(value)
        
        value_str = str(value).strip()
        
        # Unit conversion map to nanoseconds
        units = {
            'ps': 0.001,
            'ns': 1.0,
            'us': 1000.0,
            'ms': 1000000.0
        }
        
        # Check for unit suffix
        for unit, multiplier in units.items():
            if value_str.endswith(unit):
                numeric_part = value_str[:-len(unit)].strip()
                return float(numeric_part) * multiplier
        
        # No unit suffix, assume ns
        return float(value_str)
    
    def _extract_flat_config(self, data: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Extract flat top-level parameters as global config.
        
        Maps smithy parameters:
        - platform -> board
        - target_clk -> synth_clk_period_ns (with unit conversion)
        """
        # Start with explicit global_config if present
        config = data.get('global_config', {}).copy()
        
        # Top-level parameters that map directly to global config
        direct_params = [
            'output_stage', 'working_directory', 'save_intermediate_models',
            'max_combinations', 'timeout_minutes', 'fail_fast'
        ]
        
        for param in direct_params:
            if param in data and param not in config:
                config[param] = data[param]
        
        # Smithy parameter mappings for FINN config
        finn_mappings = {}
        
        # platform -> board
        if 'platform' in data:
            finn_mappings['board'] = data['platform']
        
        # target_clk -> synth_clk_period_ns with unit conversion
        if 'target_clk' in data:
            finn_mappings['synth_clk_period_ns'] = self._parse_time_with_units(data['target_clk'])
        
        return config, finn_mappings
    
    def _parse_global_config(self, config_data: Dict[str, Any]) -> GlobalConfig:
        """Parse global configuration section."""
        global_config = GlobalConfig()
        
        # Map string to enum for output_stage
        if 'output_stage' in config_data:
            stage_str = config_data['output_stage']
            global_config.output_stage = OutputStage(stage_str)
        
        # Set other fields
        for field in ['working_directory', 'save_intermediate_models', 
                      'max_combinations', 'timeout_minutes', 'fail_fast']:
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
    
    def _build_execution_tree(self, space: DesignSpace, wrapper_factory: StageWrapperFactory) -> ExecutionNode:
        """Build execution tree with segments at branch points."""
        # Track all wrapped functions for FINN registration
        wrapped_functions = {}
        
        # Root node starts empty, will accumulate initial steps
        root = ExecutionNode(
            segment_steps=[],
            finn_config=self._extract_finn_config(space.global_config)
        )
        
        current_segments = [root]
        pending_steps = []
        
        for pipeline_step in space.build_pipeline:
            step_name = self._extract_step_name(pipeline_step)
            
            if step_name in space.transform_stages:
                stage = space.transform_stages[step_name]
                combinations = stage.get_combinations()
                
                if len(combinations) <= 1:
                    # Linear - accumulate with pre-computed wrapper
                    transforms = combinations[0] if combinations else []
                    transform_names = [t.__name__ if t else None for t in transforms]
                    wrapper_name, wrapper_fn = wrapper_factory.create_stage_wrapper(
                        step_name, transform_names, 0
                    )
                    wrapped_functions[wrapper_name] = wrapper_fn
                    
                    pending_steps.append({
                        "transforms": transforms,
                        "stage_name": step_name,
                        "finn_step_name": wrapper_name
                    })
                else:
                    # Branch point - flush pending and split
                    self._flush_steps(current_segments, pending_steps)
                    current_segments = self._create_branches_with_wrappers(
                        current_segments, stage, step_name, combinations,
                        wrapper_factory, wrapped_functions
                    )
                    pending_steps = []
                    
            elif step_name == "infer_kernels":
                if self._has_kernel_choices(space.kernel_backends):
                    # Branch for kernel choices
                    self._flush_steps(current_segments, pending_steps)
                    current_segments = self._create_kernel_branches(
                        current_segments, space.kernel_backends
                    )
                    pending_steps = []
                else:
                    # Linear kernel assignment
                    pending_steps.append({
                        "kernel_backends": space.kernel_backends,
                        "name": "infer_kernels"
                    })
            else:
                # Regular step
                pending_steps.append({"name": step_name})
        
        # Flush final steps
        self._flush_steps(current_segments, pending_steps)
        
        # Register all wrapped functions with FINN at once
        self._register_wrapped_functions(wrapped_functions)
        
        # Validate tree size
        self._validate_tree_size(root, space.global_config.max_combinations)
        
        return root
    
    def _extract_step_name(self, pipeline_step) -> str:
        """Extract step name from pipeline step format."""
        if isinstance(pipeline_step, str):
            if pipeline_step.startswith("{") and pipeline_step.endswith("}"):
                return pipeline_step[1:-1]
            return pipeline_step
        elif isinstance(pipeline_step, dict):
            # YAML dict format {stage_name: null}
            return list(pipeline_step.keys())[0]
        else:
            raise ValueError(f"Unknown pipeline step format: {pipeline_step}")
    
    def _extract_finn_config(self, global_config) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from global config."""
        # Convert GlobalConfig to dict, excluding non-FINN fields
        config_dict = {}
        
        if hasattr(global_config, '__dict__'):
            for key, value in global_config.__dict__.items():
                # Skip internal fields and non-FINN config
                if not key.startswith('_') and key not in ['max_combinations']:
                    config_dict[key] = value
        
        return config_dict
    
    def _flush_steps(self, segments: List[ExecutionNode], steps: List[Dict]) -> None:
        """Add accumulated steps to segments."""
        if steps:
            for segment in segments:
                segment.segment_steps.extend(steps)
    
    def _create_branches_with_wrappers(self, segments: List[ExecutionNode], stage, 
                                      step_name: str, combinations: List,
                                      wrapper_factory: StageWrapperFactory,
                                      wrapped_functions: Dict[str, callable]) -> List[ExecutionNode]:
        """Create child segments for each combination with pre-computed wrappers."""
        new_segments = []
        
        for segment in segments:
            for i, transforms in enumerate(combinations):
                # Create wrapper with simple numeric index
                transform_names = [t.__name__ if t else None for t in transforms]
                wrapper_name, wrapper_fn = wrapper_factory.create_stage_wrapper(
                    step_name, transform_names, i
                )
                wrapped_functions[wrapper_name] = wrapper_fn
                
                # Use simple opt{i} naming for branches
                branch_id = f"{step_name}_opt{i}"
                child = segment.add_child(branch_id, [{
                    "transforms": transforms,
                    "stage_name": step_name,
                    "finn_step_name": wrapper_name
                }])
                new_segments.append(child)
        
        return new_segments
    
    def _has_kernel_choices(self, kernel_backends: List[Tuple[str, List[Type]]]) -> bool:
        """Check if kernel configuration has multiple choices."""
        # For now, we don't branch on kernel backends
        return False
    
    def _create_kernel_branches(self, segments: List[ExecutionNode], 
                               kernel_backends: List[Tuple[str, List[Type]]]) -> List[ExecutionNode]:
        """Create branches for kernel choices (if any)."""
        # Currently not implemented as kernels don't branch in current design
        raise NotImplementedError("Kernel branching not yet implemented")
    
    def _register_wrapped_functions(self, wrapped_functions: Dict[str, callable]) -> None:
        """Register all wrapped transform stages with FINN."""
        if not wrapped_functions:
            return
            
        from finn.builder.build_dataflow_steps import build_dataflow_step_lookup
        
        # Batch register all wrappers at once
        build_dataflow_step_lookup.update(wrapped_functions)
        
        print(f"Registered {len(wrapped_functions)} transform stage wrappers with FINN")
    
    def _validate_tree_size(self, root: ExecutionNode, max_combinations: int) -> None:
        """Validate tree doesn't exceed maximum combinations."""
        from .execution_tree import count_leaves
        
        leaf_count = count_leaves(root)
        if leaf_count > max_combinations:
            raise ValueError(
                f"Execution tree has {leaf_count} paths, exceeds limit of "
                f"{max_combinations}. Reduce design space or increase limit."
            )