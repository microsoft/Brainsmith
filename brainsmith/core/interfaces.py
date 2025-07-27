# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Clean interfaces to avoid circular imports.

This module provides deferred import wrappers to break circular dependencies
between forge and explorer modules.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .execution_tree import ExecutionNode
    from .design_space import DesignSpace, ForgeConfig
    from .explorer.types import TreeExecutionResult


def run_exploration(
    tree: 'ExecutionNode',
    model_path: str,
    output_dir: str,
    forge_config: 'ForgeConfig',
    design_space: 'DesignSpace'
) -> 'TreeExecutionResult':
    """
    Deferred import wrapper for explore_execution_tree.
    
    This breaks the circular dependency:
    forge.py → explorer/__init__.py → explorer.py → executor.py → plugins → steps → core → forge.py
    
    Args:
        tree: Root node of execution tree
        model_path: Path to the ONNX model
        output_dir: Output directory for results
        forge_config: Configuration for the build
        design_space: Design space with resolved plugins
        
    Returns:
        TreeExecutionResult with execution statistics
    """
    from .explorer import explore_execution_tree
    return explore_execution_tree(tree, model_path, output_dir, forge_config, design_space)