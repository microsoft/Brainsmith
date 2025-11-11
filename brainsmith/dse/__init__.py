# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration (DSE) Package.

Unified package for FPGA accelerator design space exploration.
Combines blueprint parsing, tree building, and segment execution.

Public API:
    - explore_design_space(): Main entry point for DSE
    - DSEConfig: Blueprint configuration class
    - GlobalDesignSpace: Design space representation
    - parse_blueprint(): Parse blueprint YAML to GlobalDesignSpace

Internal modules (prefixed with _) are implementation details.
"""

from ._parser import parse_blueprint
from .api import build_tree, execute_tree, explore_design_space
from .config import DSEConfig
from .design_space import GlobalDesignSpace
from .runner import SegmentRunner
from .segment import DSESegment
from .tree import DSETree

# Result types
from .types import (
    ExecutionError,
    OutputType,
    SegmentResult,
    SegmentStatus,
    TreeExecutionResult,
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
    'GlobalDesignSpace',
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

