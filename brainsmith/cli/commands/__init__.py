# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith CLI commands.

This module provides the single source of truth for all CLI command registration.
Command mappings are used by cli.py's LazyGroup for lazy loading.

Uses PEP 562 lazy imports to avoid loading settings during --help.
"""

from brainsmith._internal.lazy_imports import LazyModuleLoader

# Single source of truth for command registration
# Format: 'command_name': (relative_module, attribute_name, category)
_COMMAND_REGISTRY = {
    "components": (".components", "components", "admin"),
    "config": (".config", "config", "admin"),
    "setup": (".setup", "setup", "admin"),
    "dfc": (".dfc", "dfc", "operational"),
    "kernel": (".kernel", "kernel", "operational"),
}

# Lazy loader for individual command imports
_lazy_loader = LazyModuleLoader(
    {name: module for name, (module, _, _) in _COMMAND_REGISTRY.items()},
    package=__name__
)


def get_command_map(category: str | None = None) -> dict[str, tuple[str, str]]:
    """Get command mappings for LazyGroup, optionally filtered by category.

    Args:
        category: Filter by 'operational', 'admin', or None for all commands

    Returns:
        Dict mapping command names to (absolute_module_path, attribute_name) tuples
    """
    commands = {
        name: (module, attr)
        for name, (module, attr, cat) in _COMMAND_REGISTRY.items()
        if category is None or cat == category
    }

    # Convert relative imports to absolute for LazyGroup
    return {
        name: (f"brainsmith.cli.commands{module}", attr)
        for name, (module, attr) in commands.items()
    }


def __getattr__(name):
    """Lazy import commands on first access."""
    # Handle individual command imports
    if name in {cmd for cmd, _, _ in _COMMAND_REGISTRY.values()}:
        return _lazy_loader.get_attribute(name)

    # Handle command map exports
    if name == "OPERATIONAL_COMMAND_MAP":
        return get_command_map("operational")

    if name == "ADMIN_COMMAND_MAP":
        return get_command_map("admin")

    if name == "ALL_COMMAND_MAP":
        return get_command_map()

    # Legacy support for old names (deprecated)
    if name == "OPERATIONAL_COMMANDS":
        return {cmd: __getattr__(cmd) for cmd in ["dfc", "kernel"]}

    if name == "ADMIN_COMMANDS":
        return {cmd: __getattr__(cmd) for cmd in ["components", "config", "setup"]}

    if name == "ALL_COMMANDS":
        return {**__getattr__("OPERATIONAL_COMMANDS"), **__getattr__("ADMIN_COMMANDS")}

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    """Support for dir() and tab completion."""
    return _lazy_loader.dir() + [
        "OPERATIONAL_COMMAND_MAP",
        "ADMIN_COMMAND_MAP",
        "ALL_COMMAND_MAP",
        "get_command_map"
    ]


__all__ = [
    "OPERATIONAL_COMMAND_MAP",
    "ADMIN_COMMAND_MAP",
    "ALL_COMMAND_MAP",
    "get_command_map",
    "components",
    "config",
    "dfc",
    "kernel",
    "setup"
]
