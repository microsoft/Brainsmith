# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""FINN import hook for automatic environment configuration.

TEMPORARY SOLUTION: This import hook automatically configures the environment
when FINN or related modules are imported. This is a stopgap measure until
FINN adopts a more modern configuration system (e.g., Pydantic Settings)
that doesn't rely on environment variables.

Known limitations:
- Not suitable for PyPI distribution (global side effects)
- Can make debugging harder due to implicit behavior
- May interfere with other packages or testing frameworks

This hook will be removed once FINN's configuration system is refactored.
"""

import sys
import os
import threading
from importlib.abc import MetaPathFinder
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class FinnEnvironmentHook(MetaPathFinder):
    """Thread-safe import hook for FINN environment setup.
    
    This hook intercepts imports of FINN and related modules to ensure
    the Brainsmith environment is properly configured before FINN loads.
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._setup_done = False
        self._setup_in_progress = False
    
    def find_spec(self, fullname, path, target=None):
        """Intercept FINN imports to ensure environment is configured.
        
        This method is called by Python's import system for every import.
        We check if it's a FINN-related import and set up the environment
        if needed.
        """
        
        # Check if this is a FINN import
        if self._is_finn_import(fullname):
            with self._lock:
                if not self._setup_done and not self._setup_in_progress:
                    self._setup_in_progress = True
                    try:
                        self._setup_environment()
                        self._setup_done = True
                    finally:
                        self._setup_in_progress = False
        
        return None  # Let normal import continue
    
    def _is_finn_import(self, fullname: str) -> bool:
        """Check if this is a FINN-related import."""
        finn_modules = ['finn', 'qonnx', 'finn_experimental', 'finn_xsi']
        return any(fullname == m or fullname.startswith(f"{m}.") for m in finn_modules)
    
    def _setup_environment(self):
        """Configure environment for FINN.
        
        This sets up all necessary environment variables for FINN
        operation, including paths to Xilinx tools and build directories.
        """
        try:
            # Check if already setup by another method
            if os.environ.get('_BSMITH_FINN_ENV_SETUP') == '1':
                return
            
            # Delayed imports to avoid circular dependencies
            from brainsmith.config import get_config
            
            config = get_config()
            config.export_to_environment()
            
            # Mark as setup
            os.environ['_BSMITH_FINN_ENV_SETUP'] = '1'
            
            logger.debug("FINN environment configured via import hook")
            
        except ImportError:
            # Config not available yet (during initial setup?)
            logger.debug("Brainsmith config not available, skipping FINN setup")
        except Exception as e:
            logger.warning(f"Failed to setup FINN environment: {e}")


# Global hook instance
_finn_hook: Optional[FinnEnvironmentHook] = None


def install_finn_hook():
    """Install the FINN import hook globally.
    
    This function installs the import hook into Python's import system.
    The hook will then automatically configure the environment whenever
    FINN or related modules are imported.
    
    Note: This is a temporary solution and will be removed once FINN
    adopts better configuration management.
    """
    global _finn_hook
    
    if _finn_hook is None:
        _finn_hook = FinnEnvironmentHook()
        sys.meta_path.insert(0, _finn_hook)
        logger.debug("FINN import hook installed")


def uninstall_finn_hook():
    """Remove the FINN import hook.
    
    This function removes the import hook from Python's import system.
    Useful for testing or when you want to disable automatic configuration.
    """
    global _finn_hook
    
    if _finn_hook is not None:
        try:
            sys.meta_path.remove(_finn_hook)
            _finn_hook = None
            logger.debug("FINN import hook uninstalled")
        except ValueError:
            pass  # Hook wasn't in meta_path