# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tree Builder - Constructs ExecutionNode tree from DesignSpace

This module is responsible for building the segment-based execution tree
from a parsed DesignSpace. Separated from parsing for single responsibility.
"""

from typing import Dict, Any, List
from pathlib import Path

from .execution_tree import ExecutionNode, count_leaves
from .design_space_v2 import DesignSpace


class TreeBuilder:
    """Builds execution trees from design spaces."""
    
    def build_tree(self, space: DesignSpace) -> ExecutionNode:
        """Build execution tree with unified branching.
        
        Steps can now be direct strings or lists for variations.
        
        Args:
            space: DesignSpace containing steps and configuration
            
        Returns:
            Root ExecutionNode of the built tree
            
        Raises:
            ValueError: If tree exceeds max_combinations
        """
        # Root node starts empty, will accumulate initial steps
        root = ExecutionNode(
            segment_steps=[],
            finn_config=self._extract_finn_config(space)
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
        self._validate_tree_size(root, space.global_config.max_combinations)
        
        return root
    
    def _extract_finn_config(self, space: DesignSpace) -> Dict[str, Any]:
        """Extract FINN-relevant configuration from design space.
        
        Args:
            space: DesignSpace containing finn_config and global_config
            
        Returns:
            Dictionary of FINN configuration values
        """
        # Start with explicit finn_config
        config = space.finn_config.copy()
        
        # Add relevant fields from global config
        if hasattr(space.global_config, '__dict__'):
            for key, value in space.global_config.__dict__.items():
                # Skip internal fields and non-FINN config
                if not key.startswith('_') and key not in ['max_combinations']:
                    config[key] = value
        
        return config
    
    def _flush_steps(self, segments: List[ExecutionNode], steps: List[Dict]) -> None:
        """Add accumulated steps to segments.
        
        Args:
            segments: List of ExecutionNode segments to update
            steps: List of step dictionaries to add
        """
        if steps:
            for segment in segments:
                segment.segment_steps.extend(steps)
    
    def _create_branches(self, segments: List[ExecutionNode], 
                        branch_options: List[str]) -> List[ExecutionNode]:
        """Create child segments for branch options.
        
        Unified handling for all branches - no special transform stage logic.
        
        Args:
            segments: Parent segments to branch from
            branch_options: List of branch options (steps or skip indicators)
            
        Returns:
            List of newly created child segments
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
        """Validate tree doesn't exceed maximum combinations.
        
        Args:
            root: Root node of the tree
            max_combinations: Maximum allowed leaf nodes
            
        Raises:
            ValueError: If tree has too many paths
        """
        leaf_count = count_leaves(root)
        if leaf_count > max_combinations:
            raise ValueError(
                f"Execution tree has {leaf_count} paths, exceeds limit of "
                f"{max_combinations}. Reduce design space or increase limit."
            )