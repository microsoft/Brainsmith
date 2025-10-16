# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Artifact sharing utilities for DSE tree execution.

Provides functionality to share build artifacts between segments at branch points.
"""

import shutil
from pathlib import Path
from typing import List

from brainsmith.dse._segment import DSESegment
from brainsmith.dse._types import SegmentResult


def share_artifacts_at_branch(
    parent_result: SegmentResult,
    child_segments: List[DSESegment],
    base_output_dir: Path
) -> None:
    """
    Copy build artifacts to child segments.
    Uses full directory copies for compatibility.

    Args:
        parent_result: Result from parent segment execution
        child_segments: List of child segments to share artifacts with
        base_output_dir: Base directory for DSE outputs
    """
    if not parent_result.success:
        return

    print(f"\n  Sharing artifacts to {len(child_segments)} children...")

    for child in child_segments:
        child_dir = base_output_dir / child.segment_id
        # Full copy required for compatibility
        if child_dir.exists():
            shutil.rmtree(child_dir)
        shutil.copytree(parent_result.output_dir, child_dir)
