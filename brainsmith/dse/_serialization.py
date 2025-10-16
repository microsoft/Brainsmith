# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
DSE tree and results serialization.

Provides JSON serialization utilities for execution trees and results.
"""

import json
from typing import Dict, Any

from brainsmith.dse._segment import DSESegment
from brainsmith.dse._types import TreeExecutionResult


def serialize_tree(root: DSESegment) -> str:
    """Serialize execution tree to JSON.

    Args:
        root: Root segment of the execution tree

    Returns:
        JSON string representation of the tree
    """
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
    """Serialize execution results to JSON.

    Args:
        result: Tree execution result containing stats and segment results

    Returns:
        JSON string representation of the results
    """
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
