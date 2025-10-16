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

For component development:
    >>> from brainsmith import kernel, step
    >>> @kernel(name='MyKernel')
    >>> class MyKernel: ...
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

# Component System
from .registry import (
    # Decorators
    kernel,
    step,
    # Lookup
    get_kernel,
    get_step,
    # Listing
    list_kernels,
    list_steps,
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

    # Component System
    'kernel',
    'step',
    'get_kernel',
    'get_step',
    'list_kernels',
    'list_steps',

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