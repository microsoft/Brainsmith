# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith CLI commands.

This module provides the single source of truth for all CLI command registration.
Command mappings are used by cli.py's LazyGroup for lazy loading.
"""

# Single source of truth for command registration
# Format: 'command_name': (relative_module, attribute_name, category)
_COMMAND_REGISTRY = {
    "registry": (".registry", "registry", "admin"),
    "config": (".config", "config", "admin"),
    "setup": (".setup", "setup", "admin"),
    "dfc": (".dfc", "dfc", "operational"),
    "kernel": (".kernel", "kernel", "operational"),
}


def _build_command_map(category: str | None = None) -> dict[str, tuple[str, str]]:
    """Build command map for LazyGroup.

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


# Eagerly create command maps (cheap - just dicts)
OPERATIONAL_COMMAND_MAP = _build_command_map("operational")
ADMIN_COMMAND_MAP = _build_command_map("admin")
ALL_COMMAND_MAP = _build_command_map()

__all__ = [
    "OPERATIONAL_COMMAND_MAP",
    "ADMIN_COMMAND_MAP",
    "ALL_COMMAND_MAP",
]
