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
    forge_config,
    design_space=None
) -> TreeExecutionResult:
    """Execute exploration of the tree.
    
    Args:
        tree: Execution tree to explore
        model_path: Path to input model
        output_dir: Output directory
        forge_config: ForgeConfig with build settings
        design_space: Optional design space with kernel selections
    """
    
    # Ensure Path objects
    model_path = Path(model_path)
    output_dir = Path(output_dir)
    
    # Extract configs from ForgeConfig
    global_config = {
        'output_stage': forge_config.output_stage,
        'working_directory': forge_config.working_directory,
        'save_intermediate_models': forge_config.save_intermediate_models,
        'fail_fast': forge_config.fail_fast,
        'timeout_minutes': forge_config.timeout_minutes
    }
    finn_config = forge_config.finn_params.copy()
    
    # Add kernel selections from design space if available
    if design_space and hasattr(design_space, 'kernel_backends'):
        kernel_selections = []
        for kernel_name, backend_classes in design_space.kernel_backends:
            # For now, use the first backend (FINN's default)
            if backend_classes:
                backend_name = backend_classes[0].__name__.replace('_hls', '').replace('_rtl', '')
                kernel_selections.append((kernel_name, backend_name))
        finn_config["kernel_selections"] = kernel_selections
    
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