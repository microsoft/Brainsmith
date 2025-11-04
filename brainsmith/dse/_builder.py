# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSE Tree Builder - Constructs DSESegment tree from GlobalDesignSpace

This module is responsible for building the segment-based DSE tree
from a parsed GlobalDesignSpace. Separated from parsing for single responsibility.
"""

from typing import Dict, Any, List

from brainsmith.dse.segment import DSESegment
from brainsmith.dse.tree import DSETree
from brainsmith.dse.design_space import GlobalDesignSpace
from brainsmith.dse.config import DSEConfig
from brainsmith.dse._constants import SKIP_INDICATOR
from brainsmith.dse.types import OutputType


class DSETreeBuilder:
    """Builds DSE trees from design spaces."""
    
    def build_tree(self, space: GlobalDesignSpace, blueprint_config: DSEConfig) -> DSETree:
        """Build DSE tree with unified branching.

        Steps can now be direct strings or lists for variations.

        Args:
            space: GlobalDesignSpace containing steps and configuration
            blueprint_config: DSEConfig with FINN parameters
            
        Returns:
            DSETree containing the built tree
            
        Raises:
            ValueError: If tree exceeds max_combinations
        """
        root = DSESegment(
            steps=[],
            finn_config=self._extract_finn_config(space, blueprint_config)
        )
        
        current_segments = [root]
        pending_steps = []
        
        for step_i, step_spec in enumerate(space.steps):
            if isinstance(step_spec, list):
                # Branch point - append accumulated steps and split
                self._append_steps_to_segments(current_segments, pending_steps)
                current_segments = self._create_branches(current_segments, step_i, step_spec)
                pending_steps = []
            else:
                # Linear step - accumulate (skip SKIP_INDICATOR placeholders)
                if step_spec != SKIP_INDICATOR:
                    pending_steps.append(self._create_step_dict(step_spec, space))

        # Append final accumulated steps
        self._append_steps_to_segments(current_segments, pending_steps)
        
        # Validate tree size
        tree = DSETree(root)
        self._validate_tree_size(tree, space.max_combinations)
        
        return tree
    
    def _create_step_dict(self, step_spec: str, space: GlobalDesignSpace) -> Dict[str, Any]:
        """Create a standardized step dictionary.

        Args:
            step_spec: Step specification string
            space: GlobalDesignSpace containing configuration

        Returns:
            Dictionary with step configuration
        """
        return {"name": step_spec}
    
    def _extract_finn_config(self, space: GlobalDesignSpace, blueprint_config: DSEConfig) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from DSEConfig.

        Args:
            space: GlobalDesignSpace containing kernel backends
            blueprint_config: DSEConfig containing FINN parameters

        Returns:
            Dictionary of FINN configuration values

        Note:
            save_intermediate_models is not included here - it's always set to True
            by FINNAdapter as a workaround for FINN not returning output paths.
        """
        finn_config = {
            'output_products': blueprint_config.output.to_finn_products(),
            'board': blueprint_config.board,
            'synth_clk_period_ns': blueprint_config.clock_ns,
        }

        # Add start_step/stop_step if specified
        if blueprint_config.start_step:
            finn_config['start_step'] = blueprint_config.start_step
        if blueprint_config.stop_step:
            finn_config['stop_step'] = blueprint_config.stop_step

        # Convert kernel_backends to kernel_selections format
        # Format: [("source:kernel", ["source:backend1", "source:backend2", ...])]
        # Fully qualify all names to source:name format for registry lookups
        kernel_selections = []
        for kernel_name, backend_classes in space.kernel_backends:
            if not backend_classes:
                continue

            # Get fully qualified kernel name from first backend's target_kernel
            # All backends for a kernel should target the same kernel
            first_backend = backend_classes[0]
            if hasattr(first_backend, '__registry_name__'):
                from brainsmith.registry import get_component_metadata
                backend_meta = get_component_metadata(first_backend.__registry_name__, 'backend')
                qualified_kernel_name = backend_meta.backend_target
            else:
                # Fallback: assume kernel_name is already qualified or use as-is
                qualified_kernel_name = kernel_name if ':' in kernel_name else f"brainsmith:{kernel_name}"

            # Convert backend classes to fully qualified names
            backend_names = []
            for backend_class in backend_classes:
                if hasattr(backend_class, '__registry_name__'):
                    backend_names.append(backend_class.__registry_name__)
                else:
                    # Fallback: use class name (not ideal but better than failing)
                    backend_names.append(backend_class.__name__)

            kernel_selections.append((qualified_kernel_name, backend_names))

        if kernel_selections:
            finn_config['kernel_selections'] = kernel_selections

        # Apply any finn_config overrides from blueprint
        finn_config.update(blueprint_config.finn_overrides)

        return finn_config
    
    def _append_steps_to_segments(self, segments: List[DSESegment], steps: List[Dict]) -> None:
        """Append accumulated steps to all segments.

        Args:
            segments: List of segments to update
            steps: List of step dictionaries to append
        """
        for segment in segments:
            segment.steps.extend(steps)
    
    def _create_branches(self, segments: List[DSESegment],
                        branch_index: int,
                        branch_options: List[str]) -> List[DSESegment]:
        new_segments = []

        for segment in segments:
            for option in branch_options:
                if option == SKIP_INDICATOR:
                    branch_id = f"step_{branch_index}_skip"
                    child = segment.add_child(branch_id, [])
                else:
                    branch_id = option
                    child = segment.add_child(branch_id, [{"name": option}])
                new_segments.append(child)

        return new_segments
    
    def _validate_tree_size(self, tree: DSETree, max_combinations: int) -> None:
        def count_leaves(node: DSESegment) -> int:
            return 1 if not node.children else sum(count_leaves(c) for c in node.children.values())

        leaf_count = count_leaves(tree.root)
        if leaf_count > max_combinations:
            raise ValueError(
                f"Tree has {leaf_count} paths, exceeds limit {max_combinations}. "
                "Reduce design space or increase limit."
            )
