# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith: Neural Network Hardware Acceleration Toolkit

A framework for automating FPGA accelerator design and optimization.

Main Features:
    - Design Space Exploration (DSE)
    - Component-based architecture (Kernels via HWCustomOp, Steps)
    - Dataflow modeling and tiling
    - Kernel integration tools

Quick Start:
    >>> from brainsmith import explore_design_space, TreeExecutionResult
    >>> results: TreeExecutionResult = explore_design_space(model, blueprint)
    >>> print(f"Successful builds: {results.stats['successful']}")

Advanced DSE (Power Users):
    >>> from brainsmith import build_tree, execute_tree, SegmentRunner
    >>> # Separate build and execution for custom strategies
    >>> tree = build_tree(design_space, config)
    >>> results = execute_tree(tree, model, config, output_dir, runner=custom_runner)

For component lookup:
    >>> from brainsmith import get_kernel, get_step
    >>> LayerNorm = get_kernel('LayerNorm')
    >>> step_fn = get_step('streamline')

NOTE: Uses PEP 562 lazy imports to improve CLI startup performance.
Modules are loaded on-demand when attributes are accessed.
"""

__version__ = "0.1.0"

# Eager imports for decorators (needed at import time for deferred registration)
from .registry import step, kernel, backend, registry

# Lazy import mappings - using shared LazyModuleLoader for performance
from ._internal.lazy_imports import LazyModuleLoader

_LAZY_MODULES = {
    # DSE module
    'explore_design_space': 'dse',
    'build_tree': 'dse',
    'execute_tree': 'dse',
    'DesignSpace': 'dse',
    'DSEConfig': 'dse',
    'DSETree': 'dse',
    'DSESegment': 'dse',
    'SegmentRunner': 'dse',
    'TreeExecutionResult': 'dse',
    'SegmentResult': 'dse',
    'SegmentStatus': 'dse',
    'OutputType': 'dse',
    'ExecutionError': 'dse',

    # Loader module
    'get_kernel': 'loader',
    'get_step': 'loader',
    'get_kernel_infer': 'loader',
    'get_backend': 'loader',
    'get_backend_metadata': 'loader',
    'list_kernels': 'loader',
    'list_steps': 'loader',
    'list_backends_for_kernel': 'loader',
    'list_all_backends': 'loader',
    'has_step': 'loader',

    # Transforms module
    'import_transform': 'transforms',

    # Dataflow module
    'KernelDefinition': 'dataflow',
    'KernelModel': 'dataflow',
    'InputInterface': 'dataflow',
    'OutputInterface': 'dataflow',
    'TilingStrategy': 'dataflow',
    'TilingSpec': 'dataflow',

    # Settings module
    'SystemConfig': 'settings',
    'get_config': 'settings',
}

_lazy_loader = LazyModuleLoader(_LAZY_MODULES, package=__name__)
_config_exported = False


def __getattr__(name):
    """Lazy import attributes on first access (PEP 562)."""
    global _config_exported

    # Get the attribute using the shared lazy loader
    attr = _lazy_loader.get_attribute(name)

    # Export config to environment on first settings access
    # This is a special case for brainsmith's configuration system
    if name == 'get_config' and not _config_exported:
        try:
            config = attr()
            config.export_to_environment()
            _config_exported = True
        except Exception:
            pass  # Config might not be available during initial setup

    return attr


def __dir__():
    """Support for dir() and tab completion."""
    return _lazy_loader.dir() + ['__version__']

# __all__ is generated from lazy modules to avoid duplication
__all__ = list(_LAZY_MODULES.keys()) + ['__version__', 'step', 'kernel', 'backend', 'registry']
