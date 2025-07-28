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
from typing import Any, Dict, List, Optional, Tuple, Union, Literal
from dataclasses import dataclass

from .design_space import DesignSpace
from .config import ForgeConfig
from .execution_tree import ExecutionNode
from .plugins.registry import get_registry, has_step, list_backends_by_kernel, get_backend

# Type definitions
StepSpec = Union[str, List[Optional[str]]]

@dataclass
class StepOperation:
    """Represents a step manipulation operation"""
    op_type: Literal["after", "before", "replace", "remove", "at_start", "at_end"]
    target: Optional[StepSpec] = None
    insert: Optional[StepSpec] = None
    with_step: Optional[StepSpec] = None

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Optional['StepOperation']:
        """Parse operation from YAML dict"""
        op_mappings = {
            "after": lambda d: cls(op_type="after", target=d["after"], insert=d.get("insert")),
            "before": lambda d: cls(op_type="before", target=d["before"], insert=d.get("insert")),
            "replace": lambda d: cls(op_type="replace", target=d["replace"], with_step=d.get("with")),
            "remove": lambda d: cls(op_type="remove", target=d["remove"]),
            "at_start": lambda d: cls(op_type="at_start", insert=d["at_start"]["insert"]),
            "at_end": lambda d: cls(op_type="at_end", insert=d["at_end"]["insert"]),
        }
        
        for key, factory in op_mappings.items():
            if key in data:
                return factory(data)
        return None


class BlueprintParser:
    """Parse blueprint YAML into DesignSpace with resolved plugins."""

    def parse(self, blueprint_path: str, model_path: str) -> Tuple[DesignSpace, ExecutionNode, ForgeConfig]:
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
                parent_steps = self._parse_steps_raw(parent_steps_data)
        
        steps = self._parse_steps(
            blueprint_data.get('design_space', {}).get('steps', []),
            parent_steps=parent_steps
        )
        
        kernel_backends = self._parse_kernels(blueprint_data.get('design_space', {}).get('kernels', []))

        # Get max_combinations from environment or use default
        max_combinations = int(os.environ.get("BRAINSMITH_MAX_COMBINATIONS", "100000"))
        
        design_space = DesignSpace(
            model_path=model_path,
            steps=steps,
            kernel_backends=kernel_backends,
            max_combinations=max_combinations
        )
        design_space.validate_size()
        tree = self._build_execution_tree(design_space, forge_config)
        return design_space, tree, forge_config

    def _extract_config_and_mappings(self, data: Dict[str, Any]) -> ForgeConfig:
        """Extract ForgeConfig from blueprint data."""
        # Extract config - check both flat and global_config
        config_data = {**data.get('global_config', {}), **data}
        
        # Validate required field
        if 'clock_ns' not in config_data:
            raise ValueError("Missing required field 'clock_ns' in blueprint")
        
        return ForgeConfig(
            clock_ns=float(config_data['clock_ns']),
            output=config_data.get('output', 'estimates'),
            board=config_data.get('board'),
            verify=config_data.get('verify', False),
            verify_data=Path(config_data['verify_data']) if 'verify_data' in config_data else None,
            parallel_builds=config_data.get('parallel_builds', 4),
            debug=config_data.get('debug', False),
            save_intermediate_models=config_data.get('save_intermediate_models', False),
            finn_overrides=data.get('finn_config', {})
        )
    
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
    
    def _parse_steps_raw(self, steps_data: List[Any]) -> List[Union[str, List[Optional[str]]]]:
        """Parse steps without operations (for parent blueprints)."""
        registry = get_registry()
        return [self._validate_spec(spec, registry) for spec in steps_data if not isinstance(spec, dict)]
    
    def _parse_steps(
        self, 
        steps_data: List[Any], 
        parent_steps: Optional[List[Union[str, List[Optional[str]]]]] = None
    ) -> List[Union[str, List[Optional[str]]]]:
        """Parse steps from design_space, preserving variations and supporting operations."""
        registry = get_registry()
        
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
            result = [self._validate_spec(spec, registry) for spec in direct_steps]
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
        return [self._validate_spec(spec, registry) for spec in result]

    def _apply_step_operation(self, steps: List[StepSpec], op: StepOperation) -> List[StepSpec]:
        """Apply a single operation to the step list"""
        
        if op.op_type == "remove":
            return [s for s in steps if not self._step_matches(s, op.target)]
        
        elif op.op_type == "replace":
            new_steps = []
            for step in steps:
                if self._step_matches(step, op.target):
                    self._insert_steps(new_steps, op.with_step, check_skip=True)
                else:
                    new_steps.append(step)
            return new_steps
        
        elif op.op_type == "after":
            new_steps = []
            for step in steps:
                new_steps.append(step)
                if self._step_matches(step, op.target):
                    self._insert_steps(new_steps, op.insert)
            return new_steps
        
        elif op.op_type == "before":
            new_steps = []
            for step in steps:
                if self._step_matches(step, op.target):
                    self._insert_steps(new_steps, op.insert)
                new_steps.append(step)
            return new_steps
        
        elif op.op_type == "at_start":
            new_steps = []
            self._insert_steps(new_steps, op.insert)
            new_steps.extend(steps)
            return new_steps
        
        elif op.op_type == "at_end":
            new_steps = steps.copy()
            self._insert_steps(new_steps, op.insert)
            return new_steps
        
        return steps
    
    def _insert_steps(self, target_list: List[StepSpec], steps: StepSpec, check_skip: bool = False) -> None:
        """Insert steps as sequential or branch based on content.
        
        Args:
            target_list: List to insert steps into
            steps: Steps to insert (string or list)
            check_skip: If True, also check for "~" in list before extending
        """
        if isinstance(steps, list) and all(isinstance(s, str) for s in steps):
            if check_skip and "~" in steps:
                target_list.append(steps)
            else:
                target_list.extend(steps)
        else:
            target_list.append(steps)
    
    def _step_matches(self, step: StepSpec, target: Optional[StepSpec]) -> bool:
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
    
    def _validate_spec(self, spec: Union[str, List[str]], registry=None) -> Union[str, List[str]]:
        """Validate a step specification (string or list)."""
        if isinstance(spec, str):
            return self._validate_step(spec)
        elif isinstance(spec, list):
            return [self._validate_step(opt) for opt in spec]
        else:
            raise ValueError(f"Invalid step specification: {spec}")

    def _validate_step(self, step: Optional[str]) -> str:
        """Validate a step name against the registry, handle skip."""
        if step in [None, "~", ""]:
            return "~"
        if not has_step(step):
            raise ValueError(f"Step '{step}' not found in registry")
        return step    
    
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

    def _build_execution_tree(self, space: DesignSpace, forge_config: ForgeConfig) -> ExecutionNode:
        """Build execution tree with unified branching.
        
        Steps can now be direct strings or lists for variations.
        """
        # Root node starts empty, will accumulate initial steps
        root = ExecutionNode(
            segment_steps=[],
            finn_config=self._extract_finn_config(forge_config)
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
                if self._is_kernel_inference_step(step_spec, space):
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
        self._validate_tree_size(root, space.max_combinations)
        
        return root
    
    def _is_kernel_inference_step(self, step_spec: str, space: DesignSpace) -> bool:
        """Check if this is a kernel inference step requiring special handling."""
        return step_spec == "infer_kernels" and hasattr(space, 'kernel_backends')
    
    def _extract_finn_config(self, forge_config: ForgeConfig) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from ForgeConfig."""
        # Map ForgeConfig to FINN's expected format
        output_products = []
        if forge_config.output == "estimates":
            output_products = ["estimates"]
        elif forge_config.output == "rtl":
            output_products = ["rtl_sim", "ip_gen"]  
        elif forge_config.output == "bitfile":
            output_products = ["bitfile"]
        
        finn_config = {
            'output_products': output_products,
            'board': forge_config.board,
            'synth_clk_period_ns': forge_config.clock_ns,
            'save_intermediate_models': forge_config.save_intermediate_models
        }
        
        # Apply any finn_config overrides from blueprint
        finn_config.update(forge_config.finn_overrides)
        
        return finn_config
    
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