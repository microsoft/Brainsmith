# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Brainsmith: Neural Network Hardware Acceleration Toolkit

A framework for automating FPGA accelerator design and optimization.

Main Features:
    - Design Space Exploration (DSE)
    - Component-based architecture (Kernels, Steps)
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
    >>> from brainsmith import get_kernel, get_step, import_transform
    >>> LayerNorm = get_kernel('LayerNorm')
    >>> step_fn = get_step('streamline')
    >>> FoldConstants = import_transform('FoldConstants')
"""

# Export configuration to environment for FINN and other tools
# This needs to happen early before any FINN imports
try:
    from .settings import get_config
    config = get_config()
    config.export_to_environment()
except Exception:
    # Config might not be available during initial setup
    pass

# DSE - Main API
from .dse import (
    # High-level API
    explore_design_space,
    # Advanced API
    build_tree,
    execute_tree,
    # Configuration and structures
    DesignSpace,
    DSEConfig,
    DSETree,
    DSESegment,
    SegmentRunner,
    # Result types
    TreeExecutionResult,
    SegmentResult,
    SegmentStatus,
    OutputType,
    ExecutionError,
)

# Component System - Namespace-based registry
from .loader import (
    # Lookup
    get_kernel,
    get_step,
    get_kernel_infer,
    get_backend,
    get_backend_metadata,
    # Listing
    list_kernels,
    list_steps,
    list_backends_for_kernel,
    list_all_backends,
    has_step,
)

# Transform imports
from .transforms import (
    import_transform,
    apply_transforms,
)

# Dataflow SDK
from .dataflow import (
    KernelDefinition,
    KernelModel,
    InputInterface,
    OutputInterface,
    TilingStrategy,
    TilingSpec,
)

# Configuration
from .settings import SystemConfig

__version__ = "0.1.0"

__all__ = [
    # DSE - High-level API
    'explore_design_space',
    # DSE - Advanced API
    'build_tree',
    'execute_tree',
    # DSE - Configuration and structures
    'DesignSpace',
    'DSEConfig',
    'DSETree',
    'DSESegment',
    'SegmentRunner',
    # DSE - Result types
    'TreeExecutionResult',
    'SegmentResult',
    'SegmentStatus',
    'OutputType',
    'ExecutionError',

    # Component System - Lazy Loader
    'get_kernel',
    'get_step',
    'get_kernel_infer',
    'get_backend',
    'list_kernels',
    'list_steps',
    'list_backends',
    'has_step',
    # Transforms
    'import_transform',
    'apply_transforms',

    # Dataflow
    'KernelDefinition',
    'KernelModel',
    'InputInterface',
    'OutputInterface',
    'TilingStrategy',
    'TilingSpec',

    # Configuration
    'SystemConfig',
]