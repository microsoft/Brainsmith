# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Main entry point for exploring an execution tree with FINN."""

from pathlib import Path
from typing import Dict, Any, Union
from brainsmith.core.execution_tree import ExecutionSegment
from .types import TreeExecutionResult
from .executor import Executor
from .finn_adapter import FINNAdapter
from .utils import serialize_tree, serialize_results


def explore_execution_tree(
    tree: ExecutionSegment,
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
    
    # Map ForgeConfig to FINN's expected format
    output_products = []
    if forge_config.output == "estimates":
        output_products = ["estimates"]
    elif forge_config.output == "rtl":
        output_products = ["rtl_sim", "ip_gen"]  
    elif forge_config.output == "bitfile":
        output_products = ["bitfile"]
    
    finn_config = {
        'output_products': output_products,
        'board': forge_config.board,
        'synth_clk_period_ns': forge_config.clock_ns,
        'save_intermediate_models': forge_config.save_intermediate_models
    }
    
    # Add verification config if enabled
    if forge_config.verify and forge_config.verify_data:
        finn_config['verify_steps'] = ['initial_python']
        finn_config['verify_input_npy'] = str(forge_config.verify_data / 'input.npy')
        finn_config['verify_expected_output_npy'] = str(forge_config.verify_data / 'expected_output.npy')
    
    # Add kernel selections from design space if available
    if design_space and hasattr(design_space, 'kernel_backends'):
        kernel_selections = []
        for kernel_name, backend_classes in design_space.kernel_backends:
            # For now, use the first backend (FINN's default)
            if backend_classes:
                backend_name = backend_classes[0].__name__.replace('_hls', '').replace('_rtl', '')
                kernel_selections.append((kernel_name, backend_name))
        finn_config["kernel_selections"] = kernel_selections
    
    # Apply any finn_overrides from forge config
    if hasattr(forge_config, 'finn_overrides') and forge_config.finn_overrides:
        finn_config.update(forge_config.finn_overrides)
    
    # Save tree structure
    tree_json = output_dir / "tree.json"
    tree_json.parent.mkdir(parents=True, exist_ok=True)
    tree_json.write_text(serialize_tree(tree))
    
    # Create adapter and executor
    finn_adapter = FINNAdapter()
    executor = Executor(finn_adapter, finn_config)
    result = executor.execute(tree, Path(model_path), output_dir)
    
    # Save summary
    summary_json = output_dir / "summary.json"
    summary_json.write_text(serialize_results(result))
    
    return result