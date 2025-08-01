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

from .space import DesignSpace
from brainsmith.core.config import ForgeConfig
from brainsmith.core.plugins.registry import get_registry, has_step, list_backends_by_kernel, get_backend

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
    
    # Skip indicators
    SKIP_VALUES = frozenset([None, "~", ""])
    SKIP_NORMALIZED = "~"

    def parse(self, blueprint_path: str, model_path: str) -> Tuple[DesignSpace, ForgeConfig]:
        """
        Parse blueprint YAML and return DesignSpace and ForgeConfig.
        Steps:
        1. Load blueprint YAML (with inheritance)
        2. Extract global config and FINN mappings
        3. Parse steps and kernels
        4. Validate required fields
        5. Build DesignSpace
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
        return design_space, forge_config

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
        
        # Get registry for normalization
        registry = get_registry()
        
        # Normalize the operation target to match already-normalized steps
        normalized_target = None
        if op.target is not None:
            normalized_target = self._validate_spec(op.target, registry)
        
        # Validate nested lists in operation specs
        self._validate_nested_lists(op.insert, registry)
        self._validate_nested_lists(op.with_step, registry)
        
        # Dispatch to specific handler
        handlers = {
            "remove": self._apply_remove,
            "replace": self._apply_replace,
            "after": self._apply_after,
            "before": self._apply_before,
            "at_start": self._apply_at_start,
            "at_end": self._apply_at_end,
        }
        
        handler = handlers.get(op.op_type)
        if handler:
            return handler(steps, op, normalized_target)
        return steps
    
    def _validate_nested_lists(self, spec: Optional[StepSpec], registry) -> None:
        """Validate nested lists in step specifications"""
        if spec is not None and isinstance(spec, list):
            for item in spec:
                if isinstance(item, list):
                    self._validate_spec(item, registry)
    
    def _apply_remove(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply remove operation"""
        return [s for s in steps if not self._step_matches(s, target)]
    
    def _apply_replace(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply replace operation"""
        new_steps = []
        for step in steps:
            if self._step_matches(step, target):
                self._insert_steps(new_steps, op.with_step)
            else:
                new_steps.append(step)
        return new_steps
    
    def _apply_after(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply after operation"""
        new_steps = []
        for step in steps:
            new_steps.append(step)
            if self._step_matches(step, target):
                self._insert_steps(new_steps, op.insert)
        return new_steps
    
    def _apply_before(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply before operation"""
        new_steps = []
        for step in steps:
            if self._step_matches(step, target):
                self._insert_steps(new_steps, op.insert)
            new_steps.append(step)
        return new_steps
    
    def _apply_at_start(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply at_start operation"""
        new_steps = []
        self._insert_steps(new_steps, op.insert)
        new_steps.extend(steps)
        return new_steps
    
    def _apply_at_end(self, steps: List[StepSpec], op: StepOperation, target: Optional[StepSpec]) -> List[StepSpec]:
        """Apply at_end operation"""
        new_steps = steps.copy()
        self._insert_steps(new_steps, op.insert)
        return new_steps
    
    def _insert_steps(self, target_list: List[StepSpec], steps: StepSpec) -> None:
        """Insert steps as sequential or branch based on content.
        
        Args:
            target_list: List to insert steps into
            steps: Steps to insert (string or list)
        """
        if isinstance(steps, list):
            # Handle lists that may contain mixed types (strings and sublists)
            for step in steps:
                if isinstance(step, list):
                    # This is a branch point - append as-is
                    target_list.append(step)
                else:
                    # This is a regular step - append directly
                    target_list.append(step)
        else:
            # Single step (string) or list to be treated as branching point
            target_list.append(steps)
    
    def _step_matches(self, step: StepSpec, target: Optional[StepSpec]) -> bool:
        """Check if a step matches the target pattern"""
        if target is None:
            return False
        elif isinstance(step, str) and isinstance(target, str):
            return step == target
        elif isinstance(step, list) and isinstance(target, list):
            return set(step) == set(target)
        return False
    
    def _validate_spec(self, spec: Union[str, List[Optional[str]], None], registry=None) -> Union[str, List[str]]:
        """Validate a step specification (string or list).
        
        Rules:
        - Strings are regular steps
        - Lists are branch points (can only contain strings or None/~)
        - No nested lists allowed within branch points
        - Branch points must have at least one non-skip option
        - Branch points can have at most one skip option
        """
        if isinstance(spec, str):
            return self._validate_step(spec)
        elif isinstance(spec, list):
            # This is a branch point - validate each option
            validated = []
            skip_count = 0
            non_skip_count = 0
            
            for opt in spec:
                if isinstance(opt, str) or opt is None:
                    validated_opt = self._validate_step(opt)
                    validated.append(validated_opt)
                    if validated_opt == self.SKIP_NORMALIZED:
                        skip_count += 1
                    else:
                        non_skip_count += 1
                elif isinstance(opt, list):
                    raise ValueError(
                        f"Invalid branch point: contains nested list {opt}. "
                        "Branch points can only contain strings or skip (~). "
                        "To insert a branch point via operations, use double brackets: [[option1, option2]]"
                    )
                else:
                    raise ValueError(f"Invalid option in branch point: {opt}. Expected string or None, got {type(opt)}")
            
            # Validate branch point constraints
            if skip_count > 1:
                raise ValueError(
                    f"Invalid branch point {spec}: contains {skip_count} skip options. "
                    "Branch points can have at most one skip option."
                )
            if non_skip_count == 0:
                raise ValueError(
                    f"Invalid branch point {spec}: contains only skip options. "
                    "Branch points must have at least one non-skip step."
                )
            
            return validated
        elif spec is None:
            # Handle bare None values
            return self._validate_step(None)
        else:
            raise ValueError(f"Invalid step specification: {spec}")

    def _validate_step(self, step: Optional[str]) -> str:
        """Validate a step name against the registry, handle skip."""
        if step in self.SKIP_VALUES:
            return self.SKIP_NORMALIZED
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