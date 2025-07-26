# Copyright (c) Microsoft Corporation.
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
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    logger.warning(f"Failed to import kernel operators: {e}")
    logger.warning("Some kernel functionality may be unavailable. Install missing dependencies to enable full functionality.")

# Re-export the main API functions
from .core.forge import forge

# Re-export key data structures users need
from .core.design_space import DesignSpace

__all__ = [
    # Main API
    'forge',
    # Data structures
    'DesignSpace',
]

__version__ = "1.0.0"
__author__ = "Microsoft Research"