# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
Design Space Exploration (DSE) module for Brainsmith.

This module implements the segment-based DSE tree architecture for
exploring different design points in FPGA accelerator synthesis.
"""

from .segment import DSESegment, ArtifactState
from .tree import DSETree
from .runner import SegmentRunner
from .finn_runner import FINNRunner
from .types import SegmentResult, TreeExecutionResult, ExecutionError

__all__ = [
    'DSESegment',
    'ArtifactState', 
    'DSETree',
    'SegmentRunner',
    'FINNRunner',
    'SegmentResult',
    'TreeExecutionResult',
    'ExecutionError'
]