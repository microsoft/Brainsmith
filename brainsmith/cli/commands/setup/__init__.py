# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Setup and installation commands for Brainsmith.

This package provides commands for installing and managing optional dependencies:
- cppsim: C++ simulation dependencies (cnpy, finn-hlslib)
- xsim: Xilinx simulation support (finn-xsim)
- boards: FPGA board definition files
- all: Install all dependencies at once
- check: Verify installation status
"""

import logging
import os

import click
from rich.table import Table

from brainsmith._internal.io.dependencies import DependencyManager, BoardManager
from ...utils import (
    console, error_exit, success, tip, show_panel,
    format_status, format_warning_status
)

# Lazy import settings - only imported when check() command is run
from .cppsim import cppsim
from .xsim import xsim
from .boards import boards
from .helpers import _is_cnpy_installed, _are_hlslib_headers_installed, _is_finnxsim_built

logger = logging.getLogger(__name__)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def setup():
    """Install and configure optional Brainsmith dependencies."""
    pass


@setup.command(name="all", context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Force reinstallation even if already installed")
@click.option("--remove", "-r", is_flag=True, help="Remove all dependencies")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
def setup_all(force: bool, remove: bool, yes: bool) -> None:
    """Install all dependencies (cppsim, xsim, boards)."""
    if remove:
        show_panel(
            "Brainsmith Complete Removal",
            "This will remove all optional dependencies."
        )
    else:
        show_panel(
            "Brainsmith Complete Setup",
            "This will install all optional dependencies."
        )

    ctx = click.get_current_context()

    console.print(f"\n[bold cyan]1. {'Removing' if remove else 'Setting up'} C++ Simulation[/bold cyan]")
    ctx.invoke(cppsim, force=force, remove=remove, yes=yes)

    console.print(f"\n[bold cyan]2. {'Removing' if remove else 'Setting up'} Xilinx Simulation[/bold cyan]")
    ctx.invoke(xsim, force=force, remove=remove, yes=yes)

    console.print(f"\n[bold cyan]3. {'Removing' if remove else 'Downloading'} Board Files[/bold cyan]")
    ctx.invoke(boards, force=force, remove=remove, repo=(), verbose=False, yes=yes)

    success(f"All {'removal' if remove else 'setup'} tasks completed!")


@setup.command(context_settings={"help_option_names": ["-h", "--help"]})
def check() -> None:
    """Check the status of all setup components.

    Displays a table showing the installation status of all
    Brainsmith dependencies and tools.
    """
    from brainsmith.settings import get_config  # Lazy import
    config = get_config()
    deps_mgr = DependencyManager()

    table = Table(title="Brainsmith Setup Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Location", style="dim")

    cnpy_installed = _is_cnpy_installed(deps_mgr)
    cnpy_status = format_status("Installed" if cnpy_installed else "Not installed", cnpy_installed)
    table.add_row("cnpy", cnpy_status, "deps/cnpy")

    hlslib_installed = _are_hlslib_headers_installed(deps_mgr)
    hlslib_status = format_status("Installed" if hlslib_installed else "Not installed", hlslib_installed)
    table.add_row("finn-hlslib headers", hlslib_status, "deps/finn-hlslib")

    finnxsim_built = _is_finnxsim_built()
    finnxsim_status = format_status("Built" if finnxsim_built else "Not built", finnxsim_built)
    table.add_row("finn-xsim", finnxsim_status, "deps/finn/finn_xsi")

    if config.vivado_path:
        vivado_details = []

        # Check if settings64.sh has been sourced
        is_sourced = "XILINX_VIVADO" in os.environ
        if not is_sourced:
            vivado_details.append("⚠️  Not sourced")
        else:
            vivado_details.append("✓ Sourced")

        # Show version
        if config.xilinx_version:
            vivado_details.append(f"v{config.xilinx_version}")

        vivado_path = str(config.vivado_path)
        status_text = f"Found ({', '.join(vivado_details)})" if vivado_details else "Found"
        vivado_status = format_warning_status(status_text) if not is_sourced else format_status(status_text, True)
    else:
        vivado_status = format_status("Not found", False)
        vivado_path = "Not configured"

    table.add_row("Vivado", vivado_status, vivado_path)

    # Check Vitis HLS
    if config.vitis_hls_path:
        hls_details = []

        # Check if sourced
        is_sourced = "XILINX_HLS" in os.environ or "XILINX_VITIS_HLS" in os.environ
        if not is_sourced:
            hls_details.append("⚠️  Not sourced")
        else:
            hls_details.append("✓ Sourced")

        # Show version
        if config.xilinx_version:
            hls_details.append(f"v{config.xilinx_version}")

        hls_path = str(config.vitis_hls_path)
        status_text = f"Found ({', '.join(hls_details)})" if hls_details else "Found"
        hls_status = format_warning_status(status_text) if not is_sourced else format_status(status_text, True)
    else:
        hls_status = format_warning_status("Not found")
        hls_path = "Not configured"

    table.add_row("Vitis HLS", hls_status, hls_path)

    board_mgr = BoardManager(deps_mgr.deps_dir / "board-files")
    board_count = len(board_mgr.list_downloaded_repositories())
    board_status_text = f"{board_count} repositories" if board_count > 0 else "None"
    board_status = format_status(board_status_text, True) if board_count > 0 else format_warning_status(board_status_text)
    table.add_row("Board files", board_status, "deps/board-files")

    console.print(table)

    if not cnpy_installed or not hlslib_installed:
        tip("Run 'brainsmith setup cppsim' to install C++ simulation dependencies")

    if not finnxsim_built and config.vivado_path:
        tip("Run 'brainsmith setup xsim' to build Xilinx simulation support")

    if board_count == 0:
        tip("Run 'brainsmith setup boards' to download board definition files")


# Register subcommands
setup.add_command(cppsim)
setup.add_command(xsim)
setup.add_command(boards)

__all__ = ["setup"]
