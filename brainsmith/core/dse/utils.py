# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for segment execution.

Provides serialization and artifact sharing utilities for the explorer.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from .segment import DSESegment
from .types import SegmentResult, TreeExecutionResult


def serialize_tree(root: DSESegment) -> str:
    """Serialize execution tree to JSON."""
    def node_to_dict(node: DSESegment) -> Dict[str, Any]:
        # Serialize steps, handling kernel_backends specially
        serialized_steps = []
        for step in node.steps:
            if isinstance(step, dict) and "kernel_backends" in step:
                # Convert backend classes to string names
                step_copy = step.copy()
                kernel_backends_str = []
                for kernel_name, backend_classes in step["kernel_backends"]:
                    backend_names = [cls.__name__ for cls in backend_classes]
                    kernel_backends_str.append((kernel_name, backend_names))
                step_copy["kernel_backends"] = kernel_backends_str
                serialized_steps.append(step_copy)
            else:
                serialized_steps.append(step)

        return {
            "segment_id": node.segment_id,
            "steps": serialized_steps,
            "branch_choice": node.branch_choice,
            "is_branch_point": node.is_branch_point,
            "children": {
                name: node_to_dict(child)
                for name, child in node.children.items()
            }
        }

    return json.dumps(node_to_dict(root), indent=2)


def serialize_results(result: TreeExecutionResult) -> str:
    """Serialize execution results to JSON."""
    return json.dumps({
        "stats": result.stats,
        "total_time": result.total_time,
        "segments": {
            segment_id: {
                "success": r.success,
                "cached": r.cached,
                "error": r.error,
                "execution_time": r.execution_time,
                "output_model": str(r.output_model) if r.output_model else None,
                "output_dir": str(r.output_dir) if r.output_dir else None
            }
            for segment_id, r in result.segment_results.items()
        }
    }, indent=2)



def share_artifacts_at_branch(
    parent_result: SegmentResult,
    child_segments: List[DSESegment],
    base_output_dir: Path
) -> None:
    """
    Copy build artifacts to child segments.
    Uses full directory copies for compatibility.
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