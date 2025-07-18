# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""BrainSmith: FPGA accelerator design space exploration"""

# Register custom operators with QONNX
try:
    from . import kernels
    from .kernels.layernorm import layernorm
    from .kernels.softmax import hwsoftmax
    from .kernels.shuffle import shuffle
    from .kernels.crop import crop
    from .operators import norms
except ImportError:
    pass

# Re-export the main API functions
from .core.phase1 import forge
from .core.phase2 import explore
from .core.phase3 import create_build_runner_factory

# Re-export key data structures users need
from .core.phase1 import DesignSpace
from .core.phase2 import BuildConfig, BuildResult, BuildStatus, ExplorationResults

__all__ = [
    # Main API
    'forge',
    'explore', 
    'create_build_runner_factory',
    # Data structures
    'DesignSpace',
    'BuildConfig',
    'BuildResult',
    'BuildStatus',
    'ExplorationResults',
]

__version__ = "1.0.0"
__author__ = "Microsoft Research"