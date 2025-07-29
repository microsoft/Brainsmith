# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Tree Builder - Constructs ExecutionSegment tree from DesignSpace

This module is responsible for building the segment-based execution tree
from a parsed DesignSpace. Separated from parsing for single responsibility.
"""

from typing import Dict, Any, List

from .execution_tree import ExecutionSegment, count_leaves
from .design_space import DesignSpace
from .config import ForgeConfig


class TreeBuilder:
    """Builds execution trees from design spaces."""
    
    def build_tree(self, space: DesignSpace, forge_config: ForgeConfig) -> ExecutionSegment:
        """Build execution tree with unified branching.
        
        Steps can now be direct strings or lists for variations.
        
        Args:
            space: DesignSpace containing steps and configuration
            forge_config: ForgeConfig with FINN parameters
            
        Returns:
            Root ExecutionSegment of the built tree
            
        Raises:
            ValueError: If tree exceeds max_combinations
        """
        # Root node starts empty, will accumulate initial steps
        root = ExecutionSegment(
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
        """Extract FINN-relevant configuration from ForgeConfig.
        
        Args:
            forge_config: ForgeConfig containing FINN parameters
            
        Returns:
            Dictionary of FINN configuration values
        """
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
    
    def _flush_steps(self, segments: List[ExecutionSegment], steps: List[Dict]) -> None:
        """Add accumulated steps to segments.
        
        Args:
            segments: List of ExecutionSegment segments to update
            steps: List of step dictionaries to add
        """
        if steps:
            for segment in segments:
                segment.segment_steps.extend(steps)
    
    def _create_branches(self, segments: List[ExecutionSegment], 
                        branch_options: List[str]) -> List[ExecutionSegment]:
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
    
    def _validate_tree_size(self, root: ExecutionSegment, max_combinations: int) -> None:
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