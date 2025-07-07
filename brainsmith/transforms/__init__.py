"""
BrainSmith Transforms

Plugin-based transforms organized by compilation stage.
"""

# Import all stage modules to trigger transform registration
from . import (
    pre_proc,
    cleanup,
    topology_opt,
    kernel_opt,
    dataflow_opt,
    metadata
)

# Force import all submodules to ensure transforms are registered
import pkgutil
import importlib

# Import all submodules recursively
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    try:
        importlib.import_module(module_name)
    except ImportError as e:
        import logging
        logging.getLogger(__name__).debug(f"Failed to import {module_name}: {e}")

__all__ = [
    'pre_proc',
    'cleanup',
    'topology_opt',
    'kernel_opt', 
    'dataflow_opt',
    'metadata'
]
