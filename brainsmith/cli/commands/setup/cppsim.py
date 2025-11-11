# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""C++ simulation dependency management command."""

import shutil

import click

from brainsmith._internal.io.dependencies import DEPENDENCIES, DependencyManager

from ...utils import confirm_or_abort, console, error_exit, progress_spinner, success, warning
from .helpers import _are_hlslib_headers_installed, _is_cnpy_installed


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Force reinstallation even if already installed")
@click.option("--remove", "-r", is_flag=True, help="Remove C++ simulation dependencies")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.pass_obj
def cppsim(ctx, force: bool, remove: bool, yes: bool) -> None:
    """Setup C++ simulation dependencies (cnpy, finn-hlslib)."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")

    deps_mgr = DependencyManager()

    if remove:
        cppsim_deps = [k for k, v in DEPENDENCIES.items()
                       if v.get("group") == "cppsim" and (deps_mgr.deps_dir / k).exists()]

        if not cppsim_deps:
            warning("No C++ simulation dependencies are installed")
            return

        warning("The following C++ simulation dependencies will be removed:")
        for dep in sorted(cppsim_deps):
            console.print(f"      • {dep}")

        confirm_or_abort("\nAre you sure you want to remove these dependencies?", skip=yes)

        with progress_spinner("Removing C++ simulation dependencies...", no_progress=ctx.no_progress) as task:
            results = deps_mgr.remove_group("cppsim")
            # remove_group returns Dict[str, Optional[Exception]] where None = success
            failed = [k for k, v in results.items() if v is not None]
            if failed:
                error_exit(f"Failed to remove some dependencies: {', '.join(failed)}")
            else:
                success("C++ simulation dependencies removed successfully")
        return

    cnpy_installed = _is_cnpy_installed(deps_mgr)
    hlslib_installed = _are_hlslib_headers_installed(deps_mgr)

    if not force and cnpy_installed and hlslib_installed:
        warning("C++ simulation dependencies already installed (use --force to reinstall)")
        return

    with progress_spinner("Setting up C++ simulation dependencies...", no_progress=ctx.no_progress) as task:
        try:
            results = deps_mgr.install_group("cppsim", force=force)
            # install_group returns Dict[str, Optional[Exception]] where None = success
            failed = [k for k, v in results.items() if v is not None]
            if failed:
                error_exit(f"Failed to setup dependencies: {', '.join(failed)}")

        except Exception as e:
            # Check if it's likely a missing g++ issue
            details = []
            if not shutil.which("g++"):
                details = [
                    "C++ compiler (g++) is required for C++ simulation.",
                    "Install it with: sudo apt install g++"
                ]
            error_exit(f"Failed to setup C++ simulation: {e}", details=details)

    success("C++ simulation dependencies installed successfully")
