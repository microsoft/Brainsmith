# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith CLI commands.

This module provides a registry of all CLI commands, organized by their
availability in different entry points.
"""

from .config import config
from .dfc import dfc
from .kernel import kernel
from .plugins import plugins
from .setup import setup

# Commands available in both brainsmith and smith CLIs
OPERATIONAL_COMMANDS = {
    'dfc': dfc,
    'kernel': kernel
}

# Administrative commands only available in brainsmith CLI
ADMIN_COMMANDS = {
    'config': config,
    'plugins': plugins,
    'setup': setup
}

# All commands (for brainsmith CLI)
ALL_COMMANDS = {**OPERATIONAL_COMMANDS, **ADMIN_COMMANDS}

__all__ = [
    'OPERATIONAL_COMMANDS',
    'ADMIN_COMMANDS',
    'ALL_COMMANDS',
    'config',
    'dfc',
    'kernel',
    'plugins',
    'setup'
]
