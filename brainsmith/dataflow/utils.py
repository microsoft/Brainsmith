############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Shared utilities for the dataflow system."""

from typing import Any, Dict


def get_interface(
    interfaces: Dict[str, Any],
    name: str,
    context: str = ""
) -> Any:
    """Retrieve interface with helpful error on missing key.

    Args:
        interfaces: Dict mapping interface names to InterfaceModel instances
        name: Interface name to retrieve
        context: Optional context string for error messages

    Returns:
        Interface object

    Raises:
        ValueError: If interface not found, with list of available interfaces

    Example:
        >>> source = get_interface(interfaces, "input0", "DerivedDim")
    """
    if name not in interfaces:
        available = ', '.join(sorted(interfaces.keys()))
        raise ValueError(
            f"Source '{name}' not found. "
            f"Available interfaces/internals: {available}"
        )
    return interfaces[name]


__all__ = ['get_interface']
