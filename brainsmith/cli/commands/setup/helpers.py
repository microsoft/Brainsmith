# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helper functions for setup commands."""

from pathlib import Path

from ...utils import console, warning


def confirm_removal(items: list[str], description: str, skip_confirm: bool = False) -> bool:
    """Ask user to confirm removal of items.

    Args:
        items: List of items that will be removed
        description: Description of what's being removed
        skip_confirm: Skip confirmation prompt if True

    Returns:
        True if user confirms, False otherwise
    """
    warning(f"The following {description} will be removed:")
    for item in sorted(items):
        console.print(f"      • {item}")

    if skip_confirm:
        return True

    console.print("\n[yellow]Are you sure you want to remove these items?[/yellow]")
    response = console.input("[dim](y/N)[/dim] ").strip().lower()
    return response == 'y'


def _is_cnpy_installed(deps_mgr) -> bool:
    cnpy_dir = deps_mgr.deps_dir / "cnpy"
    return (cnpy_dir / "cnpy.h").exists()


def _are_hlslib_headers_installed(deps_mgr) -> bool:
    hlslib_dir = deps_mgr.deps_dir / "finn-hlslib"
    return (hlslib_dir / "tb").exists()


def _is_finnxsim_built() -> bool:
    from finn import xsi
    return xsi.is_available()
