# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration Tree structure and operations.
"""

from typing import Dict, List, Any
from ._segment import DSESegment
from brainsmith.dse._types import SegmentStatus


class DSETree:
    """Design space exploration tree structure and operations."""
    
    def __init__(self, root: DSESegment):
        self.root = root
    
    def get_leaf_segments(self) -> List[DSESegment]:
        """Get all complete exploration paths (leaf segments)."""
        leaves = []
        
        def collect_leaves(node: DSESegment):
            if node.is_leaf:
                leaves.append(node)
            else:
                for child in node.children.values():
                    collect_leaves(child)
        
        collect_leaves(self.root)
        return leaves
    
    def count_leaves(self) -> int:
        """Count leaf nodes in tree."""
        return self._count_leaves(self.root)
    
    def _count_leaves(self, node: DSESegment) -> int:
        """Count leaf nodes from given node."""
        if not node.children:
            return 1
        return sum(self._count_leaves(child) for child in node.children.values())
    
    def count_nodes(self) -> int:
        """Count all nodes in tree."""
        return self._count_nodes(self.root)
    
    def _count_nodes(self, node: DSESegment) -> int:
        """Count all nodes from given node."""
        count = 1  # All nodes should be counted, including root
        for child in node.children.values():
            count += self._count_nodes(child)
        return count
    
    def print_tree(self) -> None:
        """Pretty print the DSE tree."""
        self._print_node(self.root, "", True)
    
    def _print_node(self, node: DSESegment, indent: str, last: bool) -> None:
        """Pretty print a node and its children."""
        if node.segment_id != "root":
            prefix = "└── " if last else "├── "
            
            # Format segment info
            segment_info = f"{node.branch_choice or 'root'}"
            if node.steps:
                step_count = len(node.steps)
                segment_info += f" ({step_count} steps)"
            
            # Add status if not pending
            if node.status != SegmentStatus.PENDING:
                segment_info += f" [{node.status.value}]"
            
            print(f"{indent}{prefix}{segment_info}")
        
        extension = "    " if last else "│   "
        child_items = list(node.children.items())
        
        for i, (branch_id, child) in enumerate(child_items):
            is_last = i == len(child_items) - 1
            new_indent = indent + extension if node.segment_id != "root" else indent
            self._print_node(child, new_indent, is_last)
    
    def get_all_segments(self) -> List[DSESegment]:
        """Get all segments in the tree."""
        all_segments = []
        
        def collect_segments(node: DSESegment):
            all_segments.append(node)
            for child in node.children.values():
                collect_segments(child)
        
        collect_segments(self.root)
        return all_segments
    
    def get_execution_order(self) -> List[DSESegment]:
        """
        Get breadth-first execution order for the tree.
        
        This ensures parent nodes are executed before children,
        enabling proper result sharing.
        
        Returns:
            List of nodes in execution order
        """
        if self.root.segment_id == "root" and not self.root.steps:
            # Skip empty root node in execution
            queue = list(self.root.children.values())
        else:
            queue = [self.root]
        
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
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the DSE tree."""
        leaf_count = self.count_leaves()
        node_count = self.count_nodes()
        
        # Calculate depth
        max_depth = 0
        
        def calculate_depth(node: DSESegment, depth: int = 0):
            nonlocal max_depth
            max_depth = max(max_depth, depth)  # Count depth from root
            for child in node.children.values():
                calculate_depth(child, depth + 1)
        
        calculate_depth(self.root)
        
        # Count total steps
        total_steps = 0

        def count_steps(node: DSESegment):
            nonlocal total_steps
            total_steps += len(node.steps)
            for child in node.children.values():
                count_steps(child)

        count_steps(self.root)

        # Calculate segment efficiency
        # Without segments, we'd execute all steps for each path
        steps_without_segments = 0
        for leaf in self.get_leaf_segments():
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