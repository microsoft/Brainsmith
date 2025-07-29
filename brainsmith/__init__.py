# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""BrainSmith: FPGA accelerator design space exploration"""

# Re-export the main API
from .core.forge import forge

__all__ = [
    # Main API
    'forge',
]

__version__ = "0.1.0"