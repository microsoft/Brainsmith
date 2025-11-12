# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Segment-based Design Space Exploration Tree Implementation

This module implements the segment-based DSE tree architecture where
each node represents a segment of execution between branch points.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

from brainsmith.dse.types import SegmentStatus


@dataclass
class DSESegment:
    """
    A segment in the design space exploration tree.

    Each segment is executed as a single FINN build, containing all
    steps from the last branch point (or root) to the next branch point
    (or leaf).
    """

    # Core identity
    steps: list[dict[str, Any]]  # Execution steps for this segment
    branch_choice: str | None = None

    # Tree structure
    parent: DSESegment | None = None
    children: dict[str, DSESegment] = field(default_factory=dict)

    # Execution state
    status: SegmentStatus = SegmentStatus.PENDING
    output_dir: Path | None = None
    error: str | None = None
    execution_time: float | None = None

    # FINN configuration
    finn_config: dict[str, Any] = field(default_factory=dict)

    @cached_property
    def segment_id(self) -> str:
        """Deterministic ID from branch path (cached for O(1) access)."""
        # Build ID from branch decisions in path
        path_parts = []
        node = self
        while node and node.branch_choice:
            path_parts.append(node.branch_choice)
            node = node.parent
        path_parts.reverse()
        return "/".join(path_parts) if path_parts else "root"

    @property
    def is_branch_point(self) -> bool:
        return len(self.children) > 1

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    def add_child(self, branch_id: str, steps: list[dict[str, Any]]) -> DSESegment:
        """Create a child segment for a branch."""
        child = DSESegment(
            steps=steps, branch_choice=branch_id, parent=self, finn_config=self.finn_config.copy()
        )
        self.children[branch_id] = child
        return child

    def get_path(self) -> list[DSESegment]:
        """Get all segments from root to here."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path

    def get_all_steps(self) -> list[dict[str, Any]]:
        """Get all steps from root to end of this segment."""
        steps = []
        for segment in self.get_path():
            steps.extend(segment.steps)
        return steps

    def count_descendants(self) -> int:
        """Count total number of descendant nodes."""
        count = len(self.children)
        for child in self.children.values():
            count += child.count_descendants()
        return count
