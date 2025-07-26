# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Utility functions for segment execution.

Provides serialization and artifact sharing utilities for the explorer.
"""

import json
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from brainsmith.core.execution_tree import ExecutionNode
from .types import SegmentResult, TreeExecutionResult


def serialize_tree(root: ExecutionNode) -> str:
    """Serialize execution tree to JSON."""
    def node_to_dict(node: ExecutionNode) -> Dict[str, Any]:
        return {
            "segment_id": node.segment_id,
            "segment_steps": node.segment_steps,
            "branch_decision": node.branch_decision,
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
    child_segments: List[ExecutionNode],
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