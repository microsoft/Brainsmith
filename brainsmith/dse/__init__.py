# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration (DSE) Package.

Unified package for FPGA accelerator design space exploration.
Combines blueprint parsing, tree building, and segment execution.

Public API:
    - explore_design_space(): Main entry point for DSE
    - DSEConfig: Blueprint configuration class
    - DesignSpace: Design space representation
    - parse_blueprint(): Parse blueprint YAML to DesignSpace

Internal modules (prefixed with _) are implementation details.
"""

from .api import explore_design_space, build_tree, execute_tree
from .config import DSEConfig
from .design_space import DesignSpace
from ._parser import parse_blueprint
from .tree import DSETree
from .segment import DSESegment
from .runner import SegmentRunner

# Result types
from .types import (
    TreeExecutionResult,
    SegmentResult,
    SegmentStatus,
    OutputType,
    ExecutionError,
)


__all__ = [
    # High-level API (recommended for most users)
    'explore_design_space',
    'parse_blueprint',
    # Advanced API (power users)
    'build_tree',
    'execute_tree',
    'SegmentRunner',
    # Configuration
    'DSEConfig',
    'DesignSpace',
    # Tree structures
    'DSETree',
    'DSESegment',
    # Result types
    'TreeExecutionResult',
    'SegmentResult',
    'SegmentStatus',
    'OutputType',
    'ExecutionError',
]

