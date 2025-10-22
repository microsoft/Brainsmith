# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith: FPGA accelerator design space exploration"""

# Export configuration to environment for FINN and other tools
# This needs to happen early before any FINN imports
try:
    from .config import get_config
    config = get_config()
    config.export_to_environment()
except Exception:
    # Config might not be available during initial setup
    pass

# Re-export the main API
from .core.dse_api import explore_design_space

__all__ = [
    # Main API
    'explore_design_space',
]

__version__ = "0.1.0"
