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
    >>> print(f"Successful builds: {results.compute_stats()['successful']}")

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
# Registry decorators must be imported first
# Lazy import mappings - using shared LazyModuleLoader for performance
from ._internal.lazy_imports import LazyModuleLoader
from .registry import backend, kernel, source_context, step

# Eager import of core components to trigger decorator registration
# These imports must happen AFTER decorator imports but BEFORE any registry lookups
# Use source_context to ensure they're registered under 'brainsmith' namespace
with source_context("brainsmith"):
    import brainsmith.kernels  # noqa: F401 - Registers built-in kernels via @kernel decorator
    import brainsmith.steps  # noqa: F401 - Registers built-in steps via @step decorator

_LAZY_MODULES = {
    # DSE module
    "explore_design_space": "dse",
    "build_tree": "dse",
    "execute_tree": "dse",
    "GlobalDesignSpace": "dse",
    "DSEConfig": "dse",
    "DSETree": "dse",
    "DSESegment": "dse",
    "SegmentRunner": "dse",
    "TreeExecutionResult": "dse",
    "SegmentResult": "dse",
    "SegmentStatus": "dse",
    "OutputType": "dse",
    "ExecutionError": "dse",
    # Registry module (component lookup)
    "get_kernel": "registry",
    "get_step": "registry",
    "get_kernel_infer": "registry",
    "get_backend": "registry",
    "get_backend_metadata": "registry",
    "list_kernels": "registry",
    "list_steps": "registry",
    "list_backends_for_kernel": "registry",
    "list_backends": "registry",
    "has_step": "registry",
    # Dataflow module
    "KernelDefinition": "dataflow",
    "KernelModel": "dataflow",
    "InputInterface": "dataflow",
    "OutputInterface": "dataflow",
    "TilingStrategy": "dataflow",
    "TilingSpec": "dataflow",
    # Settings module
    "SystemConfig": "settings",
    "get_config": "settings",
}

_lazy_loader = LazyModuleLoader(_LAZY_MODULES, package=__name__)


def __getattr__(name):
    """Lazy import attributes on first access (PEP 562).

    Note: Config export to environment (FINN_ROOT, etc.) happens automatically
    in get_config() on first load, so no special handling needed here.
    """
    return _lazy_loader.get_attribute(name)


def __dir__():
    """Support for dir() and tab completion."""
    return _lazy_loader.dir() + ["__version__"]


# __all__ is generated from lazy modules to avoid duplication
__all__ = list(_LAZY_MODULES.keys()) + ["__version__", "step", "kernel", "backend"]
