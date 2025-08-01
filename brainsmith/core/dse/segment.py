# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Segment-based Design Space Exploration Tree Implementation

This module implements the segment-based DSE tree architecture where
each node represents a segment of execution between branch points.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class ArtifactState:
    """Track artifact locations and sharing."""
    source_dir: Optional[Path] = None
    size_bytes: Optional[int] = None
    copied_to: List[Path] = field(default_factory=list)


@dataclass
class DSESegment:
    """
    A segment in the design space exploration tree.
    
    Each segment is executed as a single FINN build, containing all
    transforms from the last branch point (or root) to the next branch point
    (or leaf).
    """
    # Core identity
    transforms: List[Dict[str, Any]]  # was: segment_steps
    branch_choice: Optional[str] = None  # was: branch_decision
    
    # Tree structure
    parent: Optional['DSESegment'] = None
    children: Dict[str, 'DSESegment'] = field(default_factory=dict)
    
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
        while node and node.branch_choice:
            path_parts.append(node.branch_choice)
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
    
    def add_child(self, branch_id: str, transforms: List[Dict[str, Any]]) -> 'DSESegment':
        """Create a child segment for a branch."""
        child = DSESegment(
            transforms=transforms,
            branch_choice=branch_id,
            parent=self,
            finn_config=self.finn_config.copy()
        )
        self.children[branch_id] = child
        return child
    
    def get_path(self) -> List['DSESegment']:
        """Get all segments from root to here."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        path.reverse()
        return path
    
    def get_all_transforms(self) -> List[Dict[str, Any]]:
        """Get all transforms from root to end of this segment."""
        transforms = []
        for segment in self.get_path():
            transforms.extend(segment.transforms)
        return transforms
    
    def get_cache_key(self) -> str:
        """Simple, deterministic cache key."""
        return self.segment_id
    
    def count_descendants(self) -> int:
        """Count total number of descendant nodes."""
        count = len(self.children)
        for child in self.children.values():
            count += child.count_descendants()
        return count