# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Segment-based Execution Tree Implementation

This module implements the new segment-based execution tree architecture where
each node represents a segment of execution between branch points.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Type
from pathlib import Path


@dataclass
class ArtifactState:
    """Track artifact locations and sharing."""
    source_dir: Optional[Path] = None
    size_bytes: Optional[int] = None
    copied_to: List[Path] = field(default_factory=list)


@dataclass
class ExecutionSegment:
    """
    A segment of execution between branch points.
    
    Each segment is executed as a single FINN build, containing all
    steps from the last branch point (or root) to the next branch point
    (or leaf).
    """
    # Core identity
    segment_steps: List[Dict[str, Any]]
    branch_decision: Optional[str] = None  # How we got here from parent
    
    # Tree structure
    parent: Optional['ExecutionSegment'] = None
    children: Dict[str, 'ExecutionSegment'] = field(default_factory=dict)
    
    # Execution state
    status: str = "pending"  # pending, running, completed, failed
    output_dir: Optional[Path] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    
    # Artifact management
    artifacts: ArtifactState = field(default_factory=ArtifactState)
    
    # FINN configuration
    finn_config: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def segment_id(self) -> str:
        """Deterministic ID from content."""
        # Build ID from branch decisions in path
        path_parts = []
        node = self
        while node and node.branch_decision:
            path_parts.append(node.branch_decision)
            node = node.parent
        path_parts.reverse()
        return "/".join(path_parts) if path_parts else "root"
    
    @property
    def is_branch_point(self) -> bool:
        """Check if this segment branches."""
        return len(self.children) > 1
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a complete path endpoint."""
        return len(self.children) == 0
    
    def add_child(self, branch_id: str, steps: List[Dict[str, Any]]) -> 'ExecutionSegment':
        """Create a child segment for a branch."""
        child = ExecutionSegment(
            segment_steps=steps,
            branch_decision=branch_id,
            parent=self,
            finn_config=self.finn_config.copy()
        )
        self.children[branch_id] = child
        return child
    
    def get_path(self) -> List['ExecutionSegment']:
        """Get all segments from root to here."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path
    
    def get_all_steps(self) -> List[Dict[str, Any]]:
        """Get all steps from root to end of this segment."""
        steps = []
        for segment in self.get_path():
            steps.extend(segment.segment_steps)
        return steps
    
    def get_cache_key(self) -> str:
        """Simple, deterministic cache key."""
        return self.segment_id
    
    def count_descendants(self) -> int:
        """Count total number of descendant nodes."""
        count = len(self.children)
        for child in self.children.values():
            count += child.count_descendants()
        return count


def get_leaf_segments(root: ExecutionSegment) -> List[ExecutionSegment]:
    """Get all complete execution paths (leaf segments)."""
    leaves = []
    
    def collect_leaves(node: ExecutionSegment):
        if node.is_leaf:
            leaves.append(node)
        else:
            for child in node.children.values():
                collect_leaves(child)
    
    collect_leaves(root)
    return leaves


def count_leaves(node: ExecutionSegment) -> int:
    """Count leaf nodes in tree."""
    if not node.children:
        return 1
    return sum(count_leaves(child) for child in node.children.values())


def count_nodes(node: ExecutionSegment) -> int:
    """Count all nodes in tree."""
    count = 0 if node.segment_id == "root" else 1
    for child in node.children.values():
        count += count_nodes(child)
    return count


def print_tree(node: ExecutionSegment, indent: str = "", last: bool = True):
    """Pretty print the execution tree with segment information."""
    if node.segment_id != "root":
        prefix = "└── " if last else "├── "
        
        # Format segment info
        segment_info = f"{node.branch_decision or 'root'}"
        if node.segment_steps:
            step_count = len(node.segment_steps)
            segment_info += f" ({step_count} steps)"
        
        # Add status if not pending
        if node.status != "pending":
            segment_info += f" [{node.status}]"
        
        print(f"{indent}{prefix}{segment_info}")
    
    extension = "    " if last else "│   "
    child_items = list(node.children.items())
    
    for i, (branch_id, child) in enumerate(child_items):
        is_last = i == len(child_items) - 1
        new_indent = indent + extension if node.segment_id != "root" else indent
        print_tree(child, new_indent, is_last)


def get_execution_order(root: ExecutionSegment) -> List[ExecutionSegment]:
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


def get_tree_stats(root: ExecutionSegment) -> Dict[str, Any]:
    """Get statistics about the execution tree."""
    leaf_count = count_leaves(root)
    node_count = count_nodes(root)
    
    # Calculate depth
    max_depth = 0
    
    def calculate_depth(node: ExecutionSegment, depth: int = 0):
        nonlocal max_depth
        if node.segment_id != "root":
            max_depth = max(max_depth, depth)
        for child in node.children.values():
            calculate_depth(child, depth + 1)
    
    calculate_depth(root)
    
    # Count total steps
    total_steps = 0
    
    def count_steps(node: ExecutionSegment):
        nonlocal total_steps
        total_steps += len(node.segment_steps)
        for child in node.children.values():
            count_steps(child)
    
    count_steps(root)
    
    # Calculate segment efficiency
    # Without segments, we'd execute all steps for each path
    steps_without_segments = 0
    for leaf in get_leaf_segments(root):
        steps_without_segments += len(leaf.get_all_steps())
    
    segment_efficiency = 1 - (total_steps / steps_without_segments) if steps_without_segments > 0 else 0
    
    return {
        'total_paths': leaf_count,
        'total_segments': node_count,
        'max_depth': max_depth,
        'total_steps': total_steps,
        'steps_without_segments': steps_without_segments,
        'segment_efficiency': round(segment_efficiency * 100, 1),  # As percentage
        'avg_steps_per_segment': round(total_steps / node_count, 1) if node_count > 0 else 0
    }