# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Blueprint Parser - YAML to DesignSpace

This module parses blueprint YAML files and creates DesignSpace objects
with all plugins resolved from the registry.
"""

import os
import yaml
from pathlib import Path
from typing import (
    TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, Literal
)
from dataclasses import dataclass

from .design_space import DesignSpace, ForgeConfig, OutputStage
from .execution_tree import ExecutionNode

if TYPE_CHECKING:
    from qonnx.transformation.base import Transformation

# Type definitions
StepDef = Union[str, List[str]]

@dataclass
class StepOperation:
    """Represents a step manipulation operation"""
    op_type: Literal["after", "before", "replace", "remove", "at_start", "at_end"]
    target: Optional[StepDef] = None
    insert: Optional[StepDef] = None
    with_step: Optional[StepDef] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['StepOperation']:
        """Parse operation from YAML dict"""
        if "after" in data:
            return cls(op_type="after", target=data["after"], insert=data.get("insert"))
        elif "before" in data:
            return cls(op_type="before", target=data["before"], insert=data.get("insert"))
        elif "replace" in data:
            return cls(op_type="replace", target=data["replace"], with_step=data.get("with"))
        elif "remove" in data:
            return cls(op_type="remove", target=data["remove"])
        elif "at_start" in data:
            return cls(op_type="at_start", insert=data["at_start"]["insert"])
        elif "at_end" in data:
            return cls(op_type="at_end", insert=data["at_end"]["insert"])
        else:
            return None


class BlueprintParser:
    """Parse blueprint YAML into DesignSpace with resolved plugins."""

    def parse(self, blueprint_path: str, model_path: str) -> Tuple[DesignSpace, ExecutionNode]:
        """
        Parse blueprint YAML and return DesignSpace and ExecutionNode root.
        Steps:
        1. Load blueprint YAML (with inheritance)
        2. Extract global config and FINN mappings
        3. Parse steps and kernels
        4. Validate required fields
        5. Build DesignSpace and execution tree
        """
        blueprint_data, parent_data = self._load_with_inheritance(blueprint_path, return_parent=True)
        forge_config = self._extract_config_and_mappings(blueprint_data)
        
        # Parse steps with inheritance support
        parent_steps = None
        if parent_data:
            parent_steps_data = parent_data.get('design_space', {}).get('steps', [])
            if parent_steps_data:
                parent_steps = self._parse_steps(parent_steps_data, skip_operations=True)
        
        steps = self._parse_steps(
            blueprint_data.get('design_space', {}).get('steps', []),
            parent_steps=parent_steps
        )
        
        kernel_backends = self._parse_kernels(blueprint_data.get('design_space', {}).get('kernels', []))

        design_space = DesignSpace(
            model_path=model_path,
            steps=steps,
            kernel_backends=kernel_backends,
            config=forge_config
        )
        design_space.validate_size()
        tree = self._build_execution_tree(design_space)
        return design_space, tree

    def _extract_config_and_mappings(self, data: Dict[str, Any]) -> ForgeConfig:
        """Extract ForgeConfig from blueprint data."""
        # Merge all config sources
        all_config = {
            **data.get('global_config', {}),
            **{k: v for k, v in data.items() if k not in ['design_space', 'extends']}
        }
        
        # Extract ForgeConfig fields
        forge_fields = {}
        for field in ForgeConfig.__dataclass_fields__:
            if field in all_config:
                value = all_config.pop(field)
                if field == 'output_stage' and isinstance(value, str):
                    value = OutputStage(value)
                forge_fields[field] = value
        
        # Handle legacy mappings
        finn_params = data.get('finn_config', {})
        if 'platform' in all_config:
            finn_params['board'] = all_config.pop('platform')
        if 'target_clk' in all_config:
            finn_params['synth_clk_period_ns'] = self._parse_time_with_units(all_config.pop('target_clk'))
        
        return ForgeConfig(**forge_fields, finn_params=finn_params)
    
    def _load_with_inheritance(self, blueprint_path: str, return_parent: bool = False) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Optional[Dict[str, Any]]]]:
        """
        Load blueprint and merge with parent if extends is specified.
        
        Args:
            blueprint_path: Path to blueprint YAML file
            return_parent: If True, also return the parent data
            
        Returns:
            If return_parent is False: Merged blueprint data
            If return_parent is True: Tuple of (merged data, parent data)
        """
        with open(blueprint_path, 'r') as f:
            data = yaml.safe_load(f)
        
        parent_data = None
        
        # Handle inheritance
        if 'extends' in data:
            # Resolve parent path relative to child
            parent_path = str(Path(blueprint_path).parent / data['extends'])
            parent_data = self._load_with_inheritance(parent_path, return_parent=False)
            
            # Deep merge parent and child
            merged = self._deep_merge(parent_data, data)
            
            if return_parent:
                return merged, parent_data
            return merged
        
        # No inheritance
        if return_parent:
            return data, None
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
    
    def _parse_steps(
        self, 
        steps_data: List[Any], 
        parent_steps: Optional[List[Union[str, List[Optional[str]]]]] = None,
        skip_operations: bool = False
    ) -> List[Union[str, List[Optional[str]]]]:
        """Parse steps from design_space, preserving variations and supporting operations."""
        from .plugins.registry import get_registry
        registry = get_registry()
        
        # Helper to validate a single step specification
        def validate_spec(spec):
            if isinstance(spec, str):
                return self._validate_step(spec, registry)
            elif isinstance(spec, list):
                return [self._validate_step(opt, registry) for opt in spec]
            else:
                raise ValueError(f"Invalid step specification: {spec}")
        
        # Skip operations for parent parsing
        if skip_operations:
            return [validate_spec(spec) for spec in steps_data if not isinstance(spec, dict)]
        
        # Separate operations from direct steps
        operations = []
        direct_steps = []
        for item in steps_data:
            if isinstance(item, dict):
                op = StepOperation.from_dict(item)
                if op:
                    operations.append(op)
            else:
                direct_steps.append(item)
        
        # Determine base steps
        if direct_steps:
            # Direct steps specified: use them (child replaces parent)
            result = [validate_spec(spec) for spec in direct_steps]
        elif parent_steps:
            # No direct steps but have parent: start with parent
            result = parent_steps.copy()
        else:
            # No steps at all: empty
            result = []
        
        # Apply operations
        for op in operations:
            result = self._apply_step_operation(result, op)
        
        # Validate result (operations might have added unvalidated steps)
        return [validate_spec(spec) for spec in result]

    def _apply_step_operation(self, steps: List[StepDef], op: StepOperation) -> List[StepDef]:
        """Apply a single operation to the step list"""
        
        if op.op_type == "remove":
            return [s for s in steps if not self._step_matches(s, op.target)]
        
        elif op.op_type == "replace":
            new_steps = []
            for step in steps:
                if self._step_matches(step, op.target):
                    # If replacing with a list of steps that are all strings (no ~), extend
                    # Otherwise keep as branching list
                    if (isinstance(op.with_step, list) and 
                        all(isinstance(s, str) for s in op.with_step) and
                        "~" not in op.with_step):
                        new_steps.extend(op.with_step)
                    else:
                        new_steps.append(op.with_step)
                else:
                    new_steps.append(step)
            return new_steps
        
        elif op.op_type == "after":
            new_steps = []
            for step in steps:
                new_steps.append(step)
                if self._step_matches(step, op.target):
                    # Handle list inserts carefully
                    if isinstance(op.insert, list):
                        if all(isinstance(item, str) for item in op.insert):
                            # Sequential steps
                            new_steps.extend(op.insert)
                        else:
                            # Branch
                            new_steps.append(op.insert)
                    else:
                        new_steps.append(op.insert)
            return new_steps
        
        elif op.op_type == "before":
            new_steps = []
            for step in steps:
                if self._step_matches(step, op.target):
                    # Handle list inserts carefully
                    if isinstance(op.insert, list):
                        if all(isinstance(item, str) for item in op.insert):
                            # Sequential steps
                            new_steps.extend(op.insert)
                        else:
                            # Branch
                            new_steps.append(op.insert)
                    else:
                        new_steps.append(op.insert)
                new_steps.append(step)
            return new_steps
        
        elif op.op_type == "at_start":
            # Handle list inserts carefully:
            # - ["a", "b", "c"] -> insert 3 sequential steps
            # - [["a", "b", "c"]] -> insert 1 branch with 3 options
            if isinstance(op.insert, list):
                # Check if it's a list of strings (sequential) or contains lists (branch)
                if all(isinstance(item, str) for item in op.insert):
                    # Sequential steps
                    return op.insert + steps
                else:
                    # Contains branches or mixed content
                    return [op.insert] + steps
            else:
                return [op.insert] + steps
        
        elif op.op_type == "at_end":
            # Handle list inserts carefully
            if isinstance(op.insert, list):
                # Check if it's a list of strings (sequential) or contains lists (branch)
                if all(isinstance(item, str) for item in op.insert):
                    # Sequential steps
                    return steps + op.insert
                else:
                    # Contains branches or mixed content
                    return steps + [op.insert]
            else:
                return steps + [op.insert]
        
        return steps
    
    def _step_matches(self, step: StepDef, target: Optional[StepDef]) -> bool:
        """Check if a step matches the target pattern"""
        if target is None:
            return False
            
        # Simple string match
        if isinstance(step, str) and isinstance(target, str):
            return step == target
        
        # List match (for branching steps)
        if isinstance(step, list) and isinstance(target, list):
            return set(step) == set(target)
        
        # No match for mismatched types
        return False

    def _validate_step(self, step: Optional[str], registry) -> str:
        """Validate a step name against the registry, handle skip."""
        if step in [None, "~", ""]:
            return "~"
        from .plugins.registry import has_step
        if not has_step(step):
            raise ValueError(f"Step '{step}' not found in registry")
        return step
    
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
    
    
    def _extract_kernel_spec(self, spec) -> Tuple[str, Optional[List[str]]]:
        """Extract kernel name and optional backend names from spec."""
        if isinstance(spec, str):
            return spec, None
        elif isinstance(spec, dict) and len(spec) == 1:
            kernel_name, backend_specs = next(iter(spec.items()))
            backend_names = backend_specs if isinstance(backend_specs, list) else [backend_specs]
            return kernel_name, backend_names
        else:
            raise ValueError(f"Invalid kernel spec: {spec}")
    
    def _parse_kernels(self, kernels_data: list) -> list:
        """Parse kernels section."""
        from .plugins.registry import list_backends_by_kernel, get_backend
        kernel_backends = []
        
        for spec in kernels_data:
            kernel_name, backend_names = self._extract_kernel_spec(spec)
            
            # If no backends specified, get all available
            if not backend_names:
                backend_names = list_backends_by_kernel(kernel_name)
            
            # Skip if no backends available
            if not backend_names:
                continue
            
            # Resolve backend classes
            backend_classes = []
            for name in backend_names:
                backend_class = get_backend(name)
                if not backend_class:
                    raise ValueError(f"Backend '{name}' not found in registry")
                backend_classes.append(backend_class)
            
            kernel_backends.append((kernel_name, backend_classes))
        
        return kernel_backends

    def _build_execution_tree(self, space: DesignSpace) -> ExecutionNode:
        """Build execution tree with unified branching.
        
        Steps can now be direct strings or lists for variations.
        """
        # Root node starts empty, will accumulate initial steps
        root = ExecutionNode(
            segment_steps=[],
            finn_config=self._extract_finn_config(space.config)
        )
        
        current_segments = [root]
        pending_steps = []
        
        for step_spec in space.steps:
            if isinstance(step_spec, list):
                # Branch point - flush and split
                self._flush_steps(current_segments, pending_steps)
                current_segments = self._create_branches(current_segments, step_spec)
                pending_steps = []
            else:
                # Linear step - accumulate
                if step_spec == "infer_kernels" and hasattr(space, 'kernel_backends'):
                    # Special handling for kernel inference
                    pending_steps.append({
                        "kernel_backends": space.kernel_backends,
                        "name": "infer_kernels"
                    })
                else:
                    # Regular step
                    pending_steps.append({"name": step_spec})
        
        # Flush final steps
        self._flush_steps(current_segments, pending_steps)
        
        # Validate tree size
        self._validate_tree_size(root, space.config.max_combinations)
        
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
    
    def _extract_finn_config(self, forge_config: ForgeConfig) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from ForgeConfig."""
        # Build FINN config from ForgeConfig fields and finn_params
        config_dict = {
            'output_stage': forge_config.output_stage,
            'working_directory': forge_config.working_directory,
            'save_intermediate_models': forge_config.save_intermediate_models,
            'fail_fast': forge_config.fail_fast,
            'timeout_minutes': forge_config.timeout_minutes,
            **forge_config.finn_params  # Include all FINN-specific params
        }
        
        return config_dict
    
    def _flush_steps(self, segments: List[ExecutionNode], steps: List[Dict]) -> None:
        """Add accumulated steps to segments."""
        if steps:
            for segment in segments:
                segment.segment_steps.extend(steps)
    
    def _create_branches(self, segments: List[ExecutionNode], 
                        branch_options: List[str]) -> List[ExecutionNode]:
        """Create child segments for branch options.
        
        Unified handling for all branches - no special transform stage logic.
        """
        new_segments = []
        
        for segment in segments:
            for i, option in enumerate(branch_options):
                if option == "~":
                    # Skip branch
                    branch_id = f"skip_{i}"
                    child = segment.add_child(branch_id, [])
                else:
                    # Regular branch with step
                    branch_id = option  # Use step name as branch ID
                    child = segment.add_child(branch_id, [{"name": option}])
                new_segments.append(child)
        
        return new_segments
    
    def _validate_tree_size(self, root: ExecutionNode, max_combinations: int) -> None:
        """Validate tree doesn't exceed maximum combinations."""
        from .execution_tree import count_leaves
        
        leaf_count = count_leaves(root)
        if leaf_count > max_combinations:
            raise ValueError(
                f"Execution tree has {leaf_count} paths, exceeds limit of "
                f"{max_combinations}. Reduce design space or increase limit."
            )