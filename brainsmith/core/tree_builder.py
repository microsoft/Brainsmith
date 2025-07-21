"""
Direct Tree Builder - DesignSpace to ExecutionTree

This module builds execution trees directly from design spaces,
automatically sharing common prefixes for optimal execution.
"""

from typing import List

from .design_space import DesignSpace
from .execution_tree import ExecutionNode


def build_execution_tree(space: DesignSpace) -> ExecutionNode:
    """
    Build execution tree directly from design space.
    
    This is the core algorithm that creates an optimal execution tree
    with automatic prefix sharing. The tree structure emerges naturally
    from traversing the pipeline and branching at variation points.
    
    Args:
        space: DesignSpace with resolved plugins
        
    Returns:
        Root node of execution tree
    """
    root = ExecutionNode("root", {"model": space.model_path})
    active_nodes = [root]
    
    for step in space.build_pipeline:
        if step.startswith("{") and step.endswith("}"):
            # Transform stage - may branch
            stage_name = step[1:-1]
            stage = space.transform_stages.get(stage_name)
            
            if not stage:
                # Stage not in design space, skip
                continue
            
            # Get all combinations for this stage
            stage_combinations = stage.get_combinations()
            
            if len(stage_combinations) == 1:
                # No branching needed
                transforms = stage_combinations[0]
                if transforms:  # Only create node if there are transforms
                    next_nodes = []
                    for node in active_nodes:
                        child = node.find_or_create_child(
                            f"stage_{stage_name}",
                            {"transforms": transforms}
                        )
                        next_nodes.append(child)
                    active_nodes = next_nodes
            else:
                # Multiple combinations - branch the tree
                next_nodes = []
                for node in active_nodes:
                    for transforms in stage_combinations:
                        if not transforms:
                            # Empty combination - continue with current node
                            next_nodes.append(node)
                        else:
                            child = node.find_or_create_child(
                                f"stage_{stage_name}",
                                {"transforms": transforms}
                            )
                            next_nodes.append(child)
                active_nodes = next_nodes
                
        elif step == "infer_kernels":
            # Kernel inference step - no branching in new design
            next_nodes = []
            for node in active_nodes:
                child = node.find_or_create_child(
                    "infer_kernels",
                    {"kernel_backends": space.kernel_backends}
                )
                next_nodes.append(child)
            active_nodes = next_nodes
            
        else:
            # Regular pipeline step
            next_nodes = []
            for node in active_nodes:
                child = node.find_or_create_child(step, {})
                next_nodes.append(child)
            active_nodes = next_nodes
    
    return root


def validate_tree_size(root: ExecutionNode, max_combinations: int) -> None:
    """
    Validate tree doesn't exceed maximum combinations.
    
    Args:
        root: Root node of execution tree
        max_combinations: Maximum allowed leaf nodes
        
    Raises:
        ValueError: If tree exceeds size limit
    """
    from .execution_tree import count_leaves
    
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
    if root.step_name == "root":
        # Skip root node in execution
        queue = list(root.children)
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
        queue.extend(node.children)
    
    return order