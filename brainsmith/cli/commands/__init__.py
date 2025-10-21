# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith CLI commands.

This module provides a registry of all CLI commands, organized by their
availability in different entry points.

Uses PEP 562 lazy imports to avoid loading settings during --help.
"""

from brainsmith._internal.lazy_imports import LazyModuleLoader

# Lazy import mapping for individual commands
_LAZY_COMMANDS = {
    'cache': '.cache',
    'config': '.config',
    'dfc': '.dfc',
    'kernel': '.kernel',
    'plugins': '.plugins',
    'setup': '.setup',
}

_lazy_loader = LazyModuleLoader(_LAZY_COMMANDS, package=__name__)


def __getattr__(name):
    """Lazy import commands on first access."""
    # Handle individual command imports
    if name in _LAZY_COMMANDS:
        return _lazy_loader.get_attribute(name)

    # Handle command group dictionaries
    if name == 'OPERATIONAL_COMMANDS':
        return {
            'dfc': __getattr__('dfc'),
            'kernel': __getattr__('kernel')
        }

    if name == 'ADMIN_COMMANDS':
        return {
            'cache': __getattr__('cache'),
            'config': __getattr__('config'),
            'plugins': __getattr__('plugins'),
            'setup': __getattr__('setup')
        }

    if name == 'ALL_COMMANDS':
        return {**__getattr__('OPERATIONAL_COMMANDS'), **__getattr__('ADMIN_COMMANDS')}

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Support for dir() and tab completion."""
    return _lazy_loader.dir() + ['OPERATIONAL_COMMANDS', 'ADMIN_COMMANDS', 'ALL_COMMANDS']


__all__ = [
    'OPERATIONAL_COMMANDS',
    'ADMIN_COMMANDS',
    'ALL_COMMANDS',
    'cache',
    'config',
    'dfc',
    'kernel',
    'plugins',
    'setup'
]
