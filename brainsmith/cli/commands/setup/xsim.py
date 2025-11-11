# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Xilinx simulation dependency management command."""

import click

from brainsmith._internal.io.dependencies import DependencyManager

from ...utils import confirm_or_abort, console, error_exit, progress_spinner, success, warning
from .helpers import _is_finnxsim_built


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Force rebuild even if already built")
@click.option("--remove", "-r", is_flag=True, help="Remove Xilinx simulation dependencies")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.pass_obj
def xsim(ctx, force: bool, remove: bool, yes: bool) -> None:
    """Setup Xilinx simulation (build finn-xsim with Vivado)."""
    from brainsmith.settings import get_config  # Lazy import

    if force and remove:
        error_exit("Cannot use --force and --remove together")

    config = get_config()
    deps_mgr = DependencyManager()

    if remove:
        xsim_deps = []
        if (deps_mgr.deps_dir / "oh-my-xilinx").exists():
            xsim_deps.append("oh-my-xilinx")
        if _is_finnxsim_built():
            xsim_deps.append("finn-xsim")

        if not xsim_deps:
            warning("No Xilinx simulation dependencies are installed")
            return

        warning("The following Xilinx simulation dependencies will be removed:")
        for dep in sorted(xsim_deps):
            console.print(f"      • {dep}")

        confirm_or_abort("\nAre you sure you want to remove these dependencies?", skip=yes)

        with progress_spinner("Removing Xilinx simulation dependencies...", no_progress=ctx.no_progress):
            results = deps_mgr.remove_group("xsim")
            # remove_group returns Dict[str, Optional[Exception]] where None = success
            failed = [k for k, v in results.items() if v is not None]
            if failed:
                error_exit(f"Failed to remove some dependencies: {', '.join(failed)}")
            else:
                success("Xilinx simulation dependencies removed successfully")
        return

    if not config.vivado_path:
        error_exit(
            "Vivado not found in configuration.",
            details=[
                "Please set up Vivado and update your configuration.",
                "Set Vivado path using:",
                "  - Environment variable: export BSMITH_XILINX__VIVADO_PATH=/path/to/vivado",
                "  - Config file: Add xilinx_path to brainsmith.yaml"
            ]
        )

    if not force and _is_finnxsim_built():
        warning("finn-xsim already built (use --force to rebuild)")
        return

    with progress_spinner("Setting up Xilinx simulation dependencies...", no_progress=ctx.no_progress):
        try:
            # First install oh-my-xilinx (raises exception on failure)
            deps_mgr.install("oh-my-xilinx", force=force)

            # Then build finn-xsim (raises exception on failure)
            deps_mgr.install("finn-xsim", force=force)
        except Exception as e:
            error_exit(
                f"Failed to setup Xilinx simulation: {e}",
                details=[
                    "Verify Vivado is properly installed",
                    "Ensure you have the required Vivado license",
                    "Check that the Vivado path in your configuration is correct"
                ]
            )

    success("Xilinx simulation dependencies installed successfully")
