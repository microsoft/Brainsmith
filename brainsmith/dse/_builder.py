# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSE Tree Builder - Constructs DSESegment tree from DesignSpace

This module is responsible for building the segment-based DSE tree
from a parsed DesignSpace. Separated from parsing for single responsibility.
"""

from typing import Dict, Any, List

from brainsmith.dse._segment import DSESegment
from brainsmith.dse._tree import DSETree
from brainsmith.dse.design_space import DesignSpace
from brainsmith.dse.config import DSEConfig
from brainsmith.core.constants import SKIP_INDICATOR
from brainsmith.core.types import OutputType
from brainsmith.core.plugins.registry import has_step, list_all_steps


class DSETreeBuilder:
    """Builds DSE trees from design spaces."""
    
    def build_tree(self, space: DesignSpace, blueprint_config: DSEConfig) -> DSETree:
        """Build DSE tree with unified branching.

        Steps can now be direct strings or lists for variations.

        Args:
            space: DesignSpace containing steps and configuration
            blueprint_config: DSEConfig with FINN parameters
            
        Returns:
            DSETree containing the built tree
            
        Raises:
            ValueError: If tree exceeds max_combinations
        """
        # Root node starts empty, will accumulate initial steps
        root = DSESegment(
            steps=[],
            finn_config=self._extract_finn_config(blueprint_config)
        )
        
        current_segments = [root]
        pending_steps = []
        
        for step_i, step_spec in enumerate(space.steps):
            if isinstance(step_spec, list):
                # Branch point - flush and split
                self._flush_steps(current_segments, pending_steps)
                current_segments = self._create_branches(current_segments, step_i, step_spec)
                pending_steps = []
            else:
                # Linear step - accumulate
                pending_steps.append(self._create_step_dict(step_spec, space))
        
        # Flush final steps
        self._flush_steps(current_segments, pending_steps)
        
        # Validate tree size
        tree = DSETree(root)
        self._validate_tree_size(tree, space.max_combinations)
        
        return tree
    
    def _create_step_dict(self, step_spec: str, space: DesignSpace) -> Dict[str, Any]:
        """Create a standardized step dictionary.
        
        Args:
            step_spec: Step specification string
            space: DesignSpace containing configuration
            
        Returns:
            Dictionary with step configuration
            
        Raises:
            ValueError: If step is not found in registry
        """
        # Validate step exists (defensive check - should already be validated in DesignSpace)
        if not has_step(step_spec):
            available_steps = list_all_steps()
            # Find similar steps for helpful error message
            similar = [s for s in available_steps if step_spec.lower() in s.lower() or s.lower() in step_spec.lower()]
            
            error_msg = f"Step '{step_spec}' not found in registry."
            if similar:
                error_msg += f" Did you mean one of: {', '.join(similar[:3])}?"
            error_msg += f"\n\nAvailable steps: {', '.join(available_steps)}"
            raise ValueError(error_msg)
        
        if step_spec == "infer_kernels" and space.kernel_backends:
            return {
                "name": step_spec,
                "kernel_backends": space.kernel_backends
            }
        return {"name": step_spec}
    
    def _extract_finn_config(self, blueprint_config: DSEConfig) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from DSEConfig.

        Args:
            blueprint_config: DSEConfig containing FINN parameters

        Returns:
            Dictionary of FINN configuration values
        """
        # Map DSEConfig to FINN's expected format
        output_products = []
        if blueprint_config.output == OutputType.ESTIMATES:
            output_products = ["estimates"]
        elif blueprint_config.output == OutputType.RTL:
            output_products = ["rtl_sim", "ip_gen"]
        elif blueprint_config.output == OutputType.BITFILE:
            output_products = ["bitfile"]

        finn_config = {
            'output_products': output_products,
            'board': blueprint_config.board,
            'synth_clk_period_ns': blueprint_config.clock_ns,
            'save_intermediate_models': blueprint_config.save_intermediate_models
        }

        # Apply any finn_config overrides from blueprint
        finn_config.update(blueprint_config.finn_overrides)
        
        return finn_config
    
    def _flush_steps(self, segments: List[DSESegment], steps: List[Dict]) -> None:
        """Add accumulated steps to segments.

        Args:
            segments: List of DSESegment segments to update
            steps: List of step dictionaries to add
        """
        if steps:
            for segment in segments:
                segment.steps.extend(steps)
    
    def _create_branches(self, segments: List[DSESegment],
                        branch_index: int,
                        branch_options: List[str]) -> List[DSESegment]:
        """Create child segments for branch options.
        
        Unified handling for all branches - no special transform stage logic.
        
        Args:
            segments: Parent segments to branch from
            branch_index: Index of the branch in the step sequence
            branch_options: List of branch options (steps or skip indicators)
            
        Returns:
            List of newly created child segments
        """
        new_segments = []
        
        for segment in segments:
            for i, option in enumerate(branch_options):
                if option == SKIP_INDICATOR:
                    # Skip branch
                    branch_id = f"step_{branch_index}_skip"
                    child = segment.add_child(branch_id, [])
                else:
                    # Regular branch with step
                    branch_id = option  # Use step name as branch ID
                    child = segment.add_child(branch_id, [{"name": option}])
                new_segments.append(child)
        
        return new_segments
    
    def _validate_tree_size(self, tree: DSETree, max_combinations: int) -> None:
        """Validate tree doesn't exceed maximum combinations.
        
        Args:
            tree: DSE tree to validate
            max_combinations: Maximum allowed leaf nodes
            
        Raises:
            ValueError: If tree has too many paths
        """
        leaf_count = tree.count_leaves()
        if leaf_count > max_combinations:
            raise ValueError(
                f"Execution tree has {leaf_count} paths, exceeds limit of "
                f"{max_combinations}. Reduce design space or increase limit."
            )