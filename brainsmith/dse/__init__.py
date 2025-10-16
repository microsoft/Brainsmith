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

from .api import explore_design_space
from .config import DSEConfig
from .design_space import DesignSpace
from ._parser import parse_blueprint
from ._tree import DSETree
from ._segment import DSESegment

# Backwards compatibility alias (same object, different name)
from .config import BlueprintConfig

__all__ = [
    'explore_design_space',
    'DSEConfig',
    'DesignSpace',
    'DSETree',
    'DSESegment',
    'parse_blueprint',
    'BlueprintConfig',  # Alias for DSEConfig
]
