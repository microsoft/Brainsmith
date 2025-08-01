# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration Tree structure and operations.
"""

from typing import Dict, List, Any
from .segment import DSESegment


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
        count = 0 if node.segment_id == "root" else 1
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
            if node.transforms:
                transform_count = len(node.transforms)
                segment_info += f" ({transform_count} transforms)"
            
            # Add status if not pending
            if node.status != "pending":
                segment_info += f" [{node.status}]"
            
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
        if self.root.segment_id == "root" and not self.root.transforms:
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
            if node.segment_id != "root":
                max_depth = max(max_depth, depth)
            for child in node.children.values():
                calculate_depth(child, depth + 1)
        
        calculate_depth(self.root)
        
        # Count total transforms
        total_transforms = 0
        
        def count_transforms(node: DSESegment):
            nonlocal total_transforms
            total_transforms += len(node.transforms)
            for child in node.children.values():
                count_transforms(child)
        
        count_transforms(self.root)
        
        # Calculate segment efficiency
        # Without segments, we'd execute all transforms for each path
        transforms_without_segments = 0
        for leaf in self.get_leaf_segments():
            transforms_without_segments += len(leaf.get_all_transforms())
        
        segment_efficiency = 1 - (total_transforms / transforms_without_segments) if transforms_without_segments > 0 else 0
        
        return {
            'total_paths': leaf_count,
            'total_segments': node_count,
            'max_depth': max_depth,
            'total_transforms': total_transforms,
            'transforms_without_segments': transforms_without_segments,
            'segment_efficiency': round(segment_efficiency * 100, 1),  # As percentage
            'avg_transforms_per_segment': round(total_transforms / node_count, 1) if node_count > 0 else 0
        }