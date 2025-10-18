# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Xilinx simulation dependency management command."""

import click

from brainsmith.settings import get_config
from brainsmith._internal.io.dependencies import DependencyManager
from ...utils import (
    console, error_exit, success, warning,
    progress_spinner
)
from .helpers import confirm_removal, _is_finnxsim_built


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if already built')
@click.option('--remove', '-r', is_flag=True, help='Remove Xilinx simulation dependencies')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def xsim(force: bool, remove: bool, yes: bool) -> None:
    """Setup Xilinx simulation (build finn-xsim with Vivado)."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")

    config = get_config()
    deps_mgr = DependencyManager()

    if remove:
        xsim_deps = []
        if (deps_mgr.deps_dir / 'oh-my-xilinx').exists():
            xsim_deps.append('oh-my-xilinx')
        if _is_finnxsim_built():
            xsim_deps.append('finn-xsim')

        if not xsim_deps:
            warning("No Xilinx simulation dependencies are installed")
            return

        if not confirm_removal(xsim_deps, "Xilinx simulation dependencies", skip_confirm=yes):
            console.print("Removal cancelled")
            return

        with progress_spinner("Removing Xilinx simulation dependencies...") as task:
            results = deps_mgr.remove_group('xsim')
            # remove_group returns Dict[str, Optional[Exception]] where None = success
            failed = [k for k, v in results.items() if v is not None]
            if failed:
                error_exit(f"Failed to remove some dependencies: {', '.join(failed)}")
            else:
                success("Xilinx simulation dependencies removed successfully")
        return

    if not config.effective_vivado_path:
        error_exit(
            "Vivado not found in configuration.",
            details=[
                "Please set up Vivado and update your configuration.",
                "Set Vivado path using:",
                "  - Environment variable: export BSMITH_XILINX__VIVADO_PATH=/path/to/vivado",
                "  - Config file: Add xilinx.vivado_path to brainsmith_settings.yaml"
            ]
        )

    if not force and _is_finnxsim_built():
        warning("finn-xsim already built (use --force to rebuild)")
        return

    with progress_spinner("Setting up Xilinx simulation dependencies...") as task:
        try:
            # First install oh-my-xilinx (raises exception on failure)
            deps_mgr.install('oh-my-xilinx', force=force)

            # Then build finn-xsim (raises exception on failure)
            deps_mgr.install('finn-xsim', force=force)
        except Exception as e:
            error_exit(
                f"Failed to setup Xilinx simulation: {e}",
                details=[
                    "Vivado is properly installed",
                    "You have the required Vivado license",
                    "The Vivado path in configuration is correct"
                ]
            )

    success("Xilinx simulation dependencies installed successfully")
