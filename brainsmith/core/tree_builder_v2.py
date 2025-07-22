"""
Segment-based Tree Builder - DesignSpace to ExecutionTree

This module builds execution trees with segments at branch points,
consolidating linear sequences of steps into single nodes.
"""

from typing import List, Dict, Any, Tuple, Type

from .design_space import DesignSpace
from .execution_tree_v2 import ExecutionNode


def build_execution_tree(space: DesignSpace) -> ExecutionNode:
    """Build execution tree with segments at branch points."""
    # Root node starts empty, will accumulate initial steps
    root = ExecutionNode(
        segment_steps=[],
        finn_config=_extract_finn_config(space.global_config)
    )
    
    current_segments = [root]
    pending_steps = []
    
    for pipeline_step in space.build_pipeline:
        step_name = _extract_step_name(pipeline_step)
        
        if step_name in space.transform_stages:
            stage = space.transform_stages[step_name]
            combinations = stage.get_combinations()
            
            if len(combinations) <= 1:
                # Linear - accumulate
                pending_steps.append({
                    "transforms": combinations[0] if combinations else [],
                    "stage_name": step_name
                })
            else:
                # Branch point - flush pending and split
                _flush_steps(current_segments, pending_steps)
                current_segments = _create_branches(
                    current_segments, stage, step_name, combinations
                )
                pending_steps = []
                
        elif step_name == "infer_kernels":
            if _has_kernel_choices(space.kernel_backends):
                # Branch for kernel choices
                _flush_steps(current_segments, pending_steps)
                current_segments = _create_kernel_branches(
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
    _flush_steps(current_segments, pending_steps)
    
    return root


def _extract_step_name(pipeline_step) -> str:
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


def _extract_finn_config(global_config) -> Dict[str, Any]:
    """Extract FINN-relevant configuration from global config."""
    # Convert GlobalConfig to dict, excluding non-FINN fields
    config_dict = {}
    
    if hasattr(global_config, '__dict__'):
        for key, value in global_config.__dict__.items():
            # Skip internal fields and non-FINN config
            if not key.startswith('_') and key not in ['max_combinations']:
                config_dict[key] = value
    
    return config_dict


def _flush_steps(segments: List[ExecutionNode], steps: List[Dict]) -> None:
    """Add accumulated steps to segments."""
    if steps:
        for segment in segments:
            segment.segment_steps.extend(steps)


def _create_branches(segments: List[ExecutionNode], stage, 
                    step_name: str, combinations: List) -> List[ExecutionNode]:
    """Create child segments for each combination."""
    new_segments = []
    
    for segment in segments:
        for i, transforms in enumerate(combinations):
            branch_id = f"{step_name}_{_format_branch_name(transforms, i)}"
            child = segment.add_child(branch_id, [{
                "transforms": transforms,
                "stage_name": step_name
            }])
            new_segments.append(child)
    
    return new_segments


def _format_branch_name(transforms: List, index: int) -> str:
    """Create readable branch name."""
    if not transforms:
        return "skip"
    elif len(transforms) == 1:
        return transforms[0].__name__
    else:
        return f"opt{index}"


def _has_kernel_choices(kernel_backends: List[Tuple[str, List[Type]]]) -> bool:
    """Check if kernel configuration has multiple choices."""
    # For now, we don't branch on kernel backends
    # This could be extended in the future if kernels have independent choices
    return False


def _create_kernel_branches(segments: List[ExecutionNode], 
                           kernel_backends: List[Tuple[str, List[Type]]]) -> List[ExecutionNode]:
    """Create branches for kernel choices (if any)."""
    # Currently not implemented as kernels don't branch in current design
    # This is a placeholder for future extension
    raise NotImplementedError("Kernel branching not yet implemented")


def validate_tree_size(root: ExecutionNode, max_combinations: int) -> None:
    """
    Validate tree doesn't exceed maximum combinations.
    
    Args:
        root: Root node of execution tree
        max_combinations: Maximum allowed leaf nodes
        
    Raises:
        ValueError: If tree exceeds size limit
    """
    from .execution_tree_v2 import count_leaves
    
    leaf_count = count_leaves(root)
    if leaf_count > max_combinations:
        raise ValueError(
            f"Execution tree has {leaf_count} paths, exceeds limit of "
            f"{max_combinations}. Reduce design space or increase limit."
        )


def find_common_ancestors(nodes: List[ExecutionNode]) -> List[ExecutionNode]:
    """
    Find common ancestor nodes that could be cached/reused.
    
    This is useful for identifying shared computation that multiple
    paths depend on.
    
    Args:
        nodes: List of nodes to analyze
        
    Returns:
        List of common ancestor nodes
    """
    if not nodes:
        return []
    
    # Get all ancestors for each node
    ancestor_sets = []
    for node in nodes:
        ancestors = set()
        current = node
        while current.parent is not None:
            ancestors.add(current.parent)
            current = current.parent
        ancestor_sets.append(ancestors)
    
    # Find intersection
    common = ancestor_sets[0]
    for ancestor_set in ancestor_sets[1:]:
        common = common.intersection(ancestor_set)
    
    return list(common)


def get_execution_order(root: ExecutionNode) -> List[ExecutionNode]:
    """
    Get breadth-first execution order for the tree.
    
    This ensures parent nodes are executed before children,
    enabling proper result sharing.
    
    Args:
        root: Root node of execution tree
        
    Returns:
        List of nodes in execution order
    """
    if root.segment_id == "root" and not root.segment_steps:
        # Skip empty root node in execution
        queue = list(root.children.values())
    else:
        queue = [root]
    
    order = []
    seen = set()
    
    while queue:
        node = queue.pop(0)
        if id(node) in seen:
            continue
            
        seen.add(id(node))
        order.append(node)
        queue.extend(node.children.values())
    
    return order