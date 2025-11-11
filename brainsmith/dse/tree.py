# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration Tree structure and operations.
"""

from collections import deque
from typing import Any

from brainsmith.dse.types import SegmentStatus

from .segment import DSESegment


class DSETree:
    """Design space exploration tree structure.

    Represents the complete exploration space as a tree of segments,
    where each segment contains steps to execute between branch points.

    Attributes:
        root: Root segment of the tree
    """

    def __init__(self, root: DSESegment):
        self.root = root

    def format_tree(self) -> str:
        """Format tree as a string representation.

        Returns:
            Multi-line string with ASCII tree visualization

        Example:
            >>> tree = build_tree(design_space, config)
            >>> print(tree.format_tree())
            └── transform_step_1 (3 steps)
                ├── kernel_backend_A (2 steps)
                └── kernel_backend_B (2 steps)
        """
        lines = []
        self._format_node(self.root, "", True, lines)
        return "\n".join(lines)

    def _format_node(self, node: DSESegment, indent: str, last: bool, lines: list[str]) -> None:
        """Format a node and its children into line list."""
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

            lines.append(f"{indent}{prefix}{segment_info}")

        extension = "    " if last else "│   "
        child_items = list(node.children.items())

        for i, (branch_id, child) in enumerate(child_items):
            is_last = i == len(child_items) - 1
            new_indent = indent + extension if node.segment_id != "root" else indent
            self._format_node(child, new_indent, is_last, lines)
    
    def get_all_segments(self) -> list[DSESegment]:
        """Get all segments in the tree."""
        all_segments = []
        
        def collect_segments(node: DSESegment):
            all_segments.append(node)
            for child in node.children.values():
                collect_segments(child)
        
        collect_segments(self.root)
        return all_segments
    
    def get_execution_order(self) -> list[DSESegment]:
        """Get breadth-first execution order for the tree.

        Skips root if it has no steps (purely structural).
        """
        # Start from root's children if root is structural-only
        if not self.root.steps and self.root.children:
            queue = deque(self.root.children.values())
        else:
            queue = deque([self.root])

        order = []

        while queue:
            node = queue.popleft()  # O(1)
            order.append(node)
            queue.extend(node.children.values())

        return order
    
    def get_statistics(self) -> dict[str, Any]:
        """Get statistics about the DSE tree."""
        stats = {
            'nodes': 0,
            'leaves': 0,
            'max_depth': 0,
            'total_steps': 0,
            'leaf_steps': []
        }

        def traverse(node: DSESegment, depth: int = 0):
            stats['nodes'] += 1
            stats['total_steps'] += len(node.steps)
            stats['max_depth'] = max(stats['max_depth'], depth)

            if not node.children:
                # Leaf node
                stats['leaves'] += 1
                stats['leaf_steps'].append(len(node.get_all_steps()))
            else:
                for child in node.children.values():
                    traverse(child, depth + 1)

        traverse(self.root)

        # Calculate efficiency
        steps_without_segments = sum(stats['leaf_steps'])
        segment_efficiency = (
            1 - stats['total_steps'] / steps_without_segments
            if steps_without_segments
            else 0
        )

        return {
            'total_paths': stats['leaves'],
            'total_segments': stats['nodes'],
            'max_depth': stats['max_depth'],
            'total_steps': stats['total_steps'],
            'steps_without_segments': steps_without_segments,
            'segment_efficiency': round(segment_efficiency * 100, 1),
            'avg_steps_per_segment': (
                round(stats['total_steps'] / stats['nodes'], 1)
                if stats['nodes'] > 0 else 0
            )
        }
