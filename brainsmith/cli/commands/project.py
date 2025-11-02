# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from __future__ import annotations  # PEP 563: Postponed evaluation of annotations

import logging
import os
from pathlib import Path
from textwrap import dedent
from typing import TYPE_CHECKING

import click
import yaml
from rich.table import Table
from rich.panel import Panel
from rich.console import Console as RichConsole

logger = logging.getLogger(__name__)

from ..context import ApplicationContext
from ..utils import console, success
from ..formatters import ConfigFormatter
from ..exceptions import ConfigurationError
from ..messages import CONFIG_EDIT_HINT

# Lazy import settings - deferred until command actually runs
if TYPE_CHECKING:
    from brainsmith.settings import SystemConfig


def _generate_config_template(defaults) -> str:
    return dedent("""\
        # Brainsmith Configuration
        # Relative paths resolve to the directory containing the .brainsmith folder

        # Build output directory
        build_dir: build

        # Xilinx tools (adjust paths for your installation)
        xilinx_path: /tools/Xilinx
        xilinx_version: "2024.2"

        # Advanced options
        default_workers: 4
        netron_port: 8080
        components_strict: true
        vendor_platform_paths: /opt/xilinx/platforms
    """)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
def project():
    """\b
    Manage Brainsmith projects and configuration.

    Each project has a .brainsmith/config.yaml file that configures:
    - Build directories and Xilinx tool paths
    - Component sources and priorities
    - Environment variables for tools
    """
    pass


@project.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--finn", is_flag=True, help="Include FINN-specific configuration settings")
@click.pass_obj
def show(ctx: ApplicationContext, finn: bool) -> None:
    """Display current configuration with source information."""
    try:
        logger.debug(f"Showing config with finn={finn}")
        config = ctx.get_effective_config()
        formatter = ConfigFormatter(console)

        # Show metadata panel
        metadata_panel = formatter._format_metadata_section(config)
        console.print(metadata_panel)
        console.print()  # Add spacing between metadata and table

        # Show configuration table with sources
        table = formatter.format_table(config, include_finn=finn)
        console.print(table)
        formatter.show_validation_warnings(config)

    except Exception as e:
        raise ConfigurationError(f"Failed to load configuration: {e}") from e


@project.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.argument('path', type=click.Path(), default='.')
@click.option("--force", "-f", is_flag=True, help="Overwrite existing config.yaml")
def init(path: str, force: bool) -> None:
    """Initialize a Brainsmith project with configuration and environment scripts.

    Creates .brainsmith/config.yaml and generates environment activation scripts
    (env.sh, .envrc).

    PATH defaults to current directory. If PATH doesn't exist, it will be created.

    Examples:

      \b
      brainsmith project init                    # Current directory
      brainsmith project init ./my-project       # Specific directory
      brainsmith project init new-project        # Creates new directory

    After init, enable environment with:

      \b
      brainsmith project allow-direnv   # Recommended
      source .brainsmith/env.sh         # Alternative
    """
    try:
        # 1. Resolve and create project path if needed
        project_path = Path(path).expanduser().resolve()
        if not project_path.exists():
            project_path.mkdir(parents=True, exist_ok=True)
            console.print(f"[dim]Created directory: {project_path}[/dim]")

        # 2. Ensure .brainsmith/ exists
        brainsmith_dir = project_path / ".brainsmith"
        brainsmith_dir.mkdir(parents=True, exist_ok=True)

        # 3. Create config.yaml (unless exists and not force)
        config_path = brainsmith_dir / "config.yaml"
        if config_path.exists() and not force:
            console.print(f"[yellow]Config already exists: {config_path}[/yellow]")
            console.print("[yellow]Skipping config.yaml (use --force to overwrite)[/yellow]")
            console.print()
        else:
            # Generate and write config
            from brainsmith.settings import get_default_config
            defaults = get_default_config()
            yaml_content = _generate_config_template(defaults)

            config_path.write_text(yaml_content)
            success(f"Created configuration: {config_path}")
            console.print(f"[dim]{CONFIG_EDIT_HINT}[/dim]")
            console.print()

        # 4. Always regenerate environment scripts
        console.print("Generating environment activation scripts...")
        from brainsmith.settings import SystemConfig

        # Load config from the project we just created/updated
        # Use explicit project file to avoid upward walk
        config = SystemConfig(_project_file=config_path)

        # Generate all scripts in the target project's .brainsmith/ directory
        config.generate_activation_script(brainsmith_dir / "env.sh")
        config.generate_direnv_file(project_path / ".envrc")

        success("Environment scripts generated")
        console.print()
        console.print("[bold]Next steps:[/bold]")
        console.print(f"  1. Edit {config_path} to match your environment")
        console.print("  2. Enable environment:")
        console.print("     [green]brainsmith project allow-direnv[/green]  (recommended)")
        console.print(f"     source {brainsmith_dir / 'env.sh'}        (alternative)")

    except Exception as e:
        raise ConfigurationError(f"Failed to initialize project: {e}") from e


@project.command(name="allow-direnv", context_settings={"help_option_names": ["-h", "--help"]})
@click.pass_obj
def allow_direnv(ctx: ApplicationContext) -> None:
    """Enable direnv integration for automatic environment switching.

    This will:
    1. Check if direnv is installed
    2. Generate .envrc file (if needed)
    3. Run 'direnv allow' to trust the file

    direnv provides automatic environment loading when you cd into the project directory,
    and automatically switches between different project configurations.
    """
    import subprocess

    # Check if direnv is installed
    try:
        result = subprocess.run(
            ['direnv', 'version'],
            capture_output=True,
            text=True,
            check=True
        )
        console.print(f"[dim]Found direnv {result.stdout.strip()}[/dim]")
    except (subprocess.CalledProcessError, FileNotFoundError):
        raise ConfigurationError(
            "direnv is not installed.\n\n"
            "Install with:\n"
            "  Ubuntu/Debian: sudo apt install direnv\n"
            "  macOS:         brew install direnv\n\n"
            "Then add to your shell config:\n"
            "  ~/.bashrc: eval \"$(direnv hook bash)\"\n"
            "  ~/.zshrc:  eval \"$(direnv hook zsh)\""
        )

    # Check if direnv hook is configured in shell
    shell = os.environ.get('SHELL', '')
    if 'bash' in shell:
        rcfile = Path.home() / '.bashrc'
        hook_line = 'eval "$(direnv hook bash)"'
    elif 'zsh' in shell:
        rcfile = Path.home() / '.zshrc'
        hook_line = 'eval "$(direnv hook zsh)"'
    else:
        rcfile = None
        hook_line = None

    if rcfile and rcfile.exists():
        try:
            rcfile_content = rcfile.read_text()
            if hook_line not in rcfile_content:
                console.print(f"[yellow]⚠️  direnv hook not found in {rcfile}[/yellow]")
                console.print(f"[yellow]   Add this line and restart your shell:[/yellow]")
                console.print(f"[yellow]   {hook_line}[/yellow]")
                console.print()
        except (OSError, PermissionError):
            # Can't read file, skip hook check
            pass

    # Ensure project is initialized
    config_path = Path.cwd() / ".brainsmith" / "config.yaml"
    if not config_path.exists():
        raise ConfigurationError(
            f"No project found in {Path.cwd()}\n\n"
            "Initialize a project first:\n"
            "  brainsmith project init"
        )

    # Generate .envrc if needed
    envrc_path = Path.cwd() / ".envrc"
    if not envrc_path.exists():
        console.print("Generating .envrc file...")
        from brainsmith.settings.schema import generate_activation_scripts
        generate_activation_scripts()
        console.print()

    # Run direnv allow
    try:
        subprocess.run(['direnv', 'allow'], check=True, cwd=Path.cwd())
        success("✅ direnv enabled for this project")
        console.print()
        console.print("[bold green]Environment will auto-load when you cd into this directory[/bold green]")

        # Suggest quiet direnv if not configured
        if 'DIRENV_LOG_FORMAT' not in os.environ:
            console.print()
            console.print("[dim]To reduce direnv output, add to your shell config:[/dim]")
            console.print('[dim]  export DIRENV_LOG_FORMAT=""[/dim]')

        console.print()
        console.print("[dim]Test it now:[/dim]")
        console.print("  cd .")
        console.print("  echo $VIVADO_PATH  # Should be set")
    except subprocess.CalledProcessError as e:
        raise ConfigurationError(f"Failed to run 'direnv allow': {e}")
