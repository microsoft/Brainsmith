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

# Install FINN import hook for automatic environment configuration
# NOTE: This is a TEMPORARY solution until FINN adopts better configuration
# management (e.g., Pydantic Settings). Import hooks are not ideal for
# PyPI packages due to global side effects.
try:
    from .hooks.finn_import_hook import install_finn_hook
    install_finn_hook()
except Exception:
    # Hook installation is optional - if it fails, users can still
    # import brainsmith before finn as a fallback
    pass

# Re-export the main API
from .core.dse_api import explore_design_space

__all__ = [
    # Main API
    'explore_design_space',
]

__version__ = "0.1.0"