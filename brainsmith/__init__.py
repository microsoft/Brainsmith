# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith: FPGA accelerator design space exploration"""

# Re-export the main API
from .core.dse_api import explore_design_space

__all__ = [
    # Main API
    'explore_design_space',
]

__version__ = "0.1.0"