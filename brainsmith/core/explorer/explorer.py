# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main entry point for exploring an execution tree with FINN."""

from pathlib import Path
from typing import Dict, Any, Union
from brainsmith.core.execution_tree import ExecutionNode
from .types import TreeExecutionResult
from .executor import Executor
from .finn_adapter import FINNAdapter
from .utils import serialize_tree, serialize_results


def explore_execution_tree(
    tree: ExecutionNode,
    model_path: Union[str, Path],
    output_dir: Union[str, Path],
    blueprint_config: Dict[str, Any]
) -> TreeExecutionResult:
    """Execute exploration of the tree.
    
    Args:
        tree: Execution tree to explore
        model_path: Path to input model
        output_dir: Output directory
        blueprint_config: Blueprint configuration
    """
    
    # Ensure Path objects
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    # Extract configs
    global_config = blueprint_config.get("global_config", {})
    finn_config = blueprint_config.get("finn_config", {})
    
    # Save tree structure
    tree_json = output_dir / "tree.json"
    tree_json.parent.mkdir(parents=True, exist_ok=True)
    tree_json.write_text(serialize_tree(tree))
    
    # Create adapter and executor
    finn_adapter = FINNAdapter()
    executor = Executor(finn_adapter, finn_config, global_config)
    result = executor.execute(tree, Path(model_path), output_dir)
    
    # Save summary
    summary_json = output_dir / "summary.json"
    summary_json.write_text(serialize_results(result))
    
    return result