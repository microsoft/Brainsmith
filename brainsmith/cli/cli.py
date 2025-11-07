# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import sys
from pathlib import Path

import click

from .context import ApplicationContext
from .utils import console
from .constants import (
    CLI_NAME_BRAINSMITH,
    CLI_NAME_SMITH,
    ExitCode,
)

logger = logging.getLogger(__name__)


def _version_callback(ctx, param, value):
    if not value:
        return
    import importlib.metadata
    from .messages import PACKAGE_NAME
    version = importlib.metadata.version(PACKAGE_NAME)
    console.print(f"[bold]{CLI_NAME_BRAINSMITH}[/bold], version {version}")
    ctx.exit()


def _should_skip_environment_validation() -> bool:
    """Detect if current command should skip environment validation.

    Commands that create or bootstrap the environment don't need it sourced yet.

    Returns:
        True if validation should be skipped, False otherwise
    """
    args = sys.argv[1:]  # Skip program name

    # project init - creates environment files
    if len(args) >= 2 and args[0] == 'project' and args[1] == 'init':
        return True

    # Future: add other bootstrap commands here if needed
    # e.g., setup commands, environment generation, etc.

    return False


class LazyGroup(click.Group):
    def __init__(self, *args, lazy_commands=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lazy_commands = lazy_commands or {}

    def list_commands(self, ctx):
        lazy_names = set(self.lazy_commands.keys())
        manual_names = set(super().list_commands(ctx))
        return sorted(lazy_names | manual_names)

    def get_command(self, ctx, name):
        if name in self.lazy_commands:
            from importlib import import_module
            module_path, attr_name = self.lazy_commands[name]
            module = import_module(module_path)
            return getattr(module, attr_name)

        return super().get_command(ctx, name)


def _create_smith_subcommand() -> click.Command:
    @click.command(context_settings={"help_option_names": ["-h", "--help"]})
    @click.argument("args", nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def smith(ctx: click.Context, args: tuple[str, ...]) -> None:
        """Create hardware designs and components.

        Provides access to dataflow accelerator and kernel generation tools.
        Inherits configuration from parent brainsmith command.

        Example: brainsmith --log-level debug smith dfc model.onnx blueprint.yaml
        """
        smith_cli = create_cli(CLI_NAME_SMITH, include_admin=False)

        with smith_cli.make_context(
            CLI_NAME_SMITH, list(args), parent=ctx, obj=ctx.obj
        ) as smith_ctx:
            smith_cli.invoke(smith_ctx)

    return smith


def create_cli(name: str, include_admin: bool = True) -> click.Group:
    # Import command maps from single source of truth
    from brainsmith.cli.commands import OPERATIONAL_COMMAND_MAP, ADMIN_COMMAND_MAP

    lazy_commands = {}
    if name == CLI_NAME_SMITH:
        lazy_commands.update(OPERATIONAL_COMMAND_MAP)

    if include_admin:
        lazy_commands.update(ADMIN_COMMAND_MAP)

    @click.pass_context
    def callback(
        ctx: click.Context,
        build_dir: Path | None,
        config: Path | None,
        log_level: str,
        no_progress: bool
    ) -> None:
        # Bootstrap commands (like project init) don't need environment or config
        skip_bootstrap = _should_skip_environment_validation()

        if not skip_bootstrap:
            # Validate environment is sourced
            from brainsmith.settings.validation import ensure_environment_sourced
            ensure_environment_sourced()

            # Create ApplicationContext (loads config)
            ctx.obj = ApplicationContext.from_cli_args(
                config_file=config,
                build_dir_override=build_dir,
                log_level=log_level,
                no_progress=no_progress,
                cli_name=name
            )

    # Create the lazy group
    cli = LazyGroup(
        name=name,
        callback=callback,
        context_settings={"help_option_names": ["-h", "--help"]},
        lazy_commands=lazy_commands
    )

    cli.params.append(click.Option(
        ["-b", "--build-dir"],
        type=click.Path(path_type=Path),
        help="Override build directory"
    ))
    cli.params.append(click.Option(
        ["-c", "--config"],
        type=click.Path(exists=True, path_type=Path),
        help="Override configuration file"
    ))
    cli.params.append(click.Option(
        ["-l", "--log-level"],
        type=click.Choice(["quiet", "normal", "verbose", "debug"]),
        default="normal",
        metavar="LEVEL",
        help="Set log verbosity (quiet|normal|verbose|debug)"
    ))
    cli.params.append(click.Option(
        ["--no-progress"],
        is_flag=True,
        help="Disable progress spinners and animations"
    ))

    cli.params.append(click.Option(
        ["--version"],
        is_flag=True,
        expose_value=False,
        is_eager=True,
        callback=_version_callback,
        help="Show the version and exit."
    ))

    if name == CLI_NAME_SMITH:
        cli.help = """Smith - Create hardware designs and components.

\b
COMMANDS:
  smith dfc MODEL BLUEPRINT           Create dataflow core
  smith kernel RTL_FILE               Generate hardware kernel

\b
Use --help with any command for detailed options."""
    else:
        cli.help = """Brainsmith - Neural network hardware acceleration toolkit.

\b
Use 'smith' subcommand or standalone 'smith' CLI for hardware design creation."""

    if include_admin and name == CLI_NAME_BRAINSMITH:
        cli.add_command(_create_smith_subcommand(), name="smith")

    return cli


def _run_cli(name: str, include_admin: bool) -> None:
    """Run CLI with consistent error handling."""
    from .exceptions import CLIError

    try:
        cli = create_cli(name, include_admin=include_admin)
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(ExitCode.INTERRUPTED)
    except CLIError as e:
        # Structured CLI errors - format nicely
        console.print(e.format_for_console())
        sys.exit(e.exit_code)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception(f"Unexpected error in {name} CLI")
        sys.exit(ExitCode.SOFTWARE)


def brainsmith_main() -> None:
    _run_cli(CLI_NAME_BRAINSMITH, include_admin=True)


def smith_main() -> None:
    _run_cli(CLI_NAME_SMITH, include_admin=False)


if __name__ == "__main__":
    prog_name = os.path.basename(sys.argv[0])

    if CLI_NAME_BRAINSMITH in prog_name:
        brainsmith_main()
    else:
        smith_main()
