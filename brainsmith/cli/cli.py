# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import sys
from pathlib import Path

import click

from .commands import OPERATIONAL_COMMANDS, ADMIN_COMMANDS
from .context import ApplicationContext
from .utils import console, setup_logging
from .constants import (
    CLI_NAME_BRAINSMITH,
    CLI_NAME_SMITH,
    ENV_QUIET,
    EXIT_INTERRUPTED,
)

logger = logging.getLogger(__name__)


def _register_commands(cli: click.Group, name: str, include_admin: bool) -> None:
    """Register commands based on CLI type and admin privileges.

    name='smith': Registers operational commands (dfc, kernel)
    include_admin=True: Registers admin commands (config, setup) + smith subcommand

    Valid combinations:
    - name='smith', include_admin=False → operational commands only
    - name='brainsmith', include_admin=True → admin + smith subcommand + operational (via subcommand)
    """
    if name == CLI_NAME_SMITH:
        for cmd_name, cmd in OPERATIONAL_COMMANDS.items():
            cli.add_command(cmd, name=cmd_name)

    if include_admin:
        for cmd_name, cmd in ADMIN_COMMANDS.items():
            cli.add_command(cmd, name=cmd_name)

        cli.add_command(_create_smith_subcommand(), name=CLI_NAME_SMITH)


def _create_smith_subcommand() -> click.Command:
    @click.command(context_settings={'help_option_names': ['-h', '--help']})
    @click.argument('args', nargs=-1, type=click.UNPROCESSED)
    @click.pass_context
    def smith(ctx: click.Context, args: tuple[str, ...]) -> None:
        """Create hardware designs and components.

        Provides access to dataflow accelerator and kernel generation tools.
        Inherits configuration from parent brainsmith command.

        Example: brainsmith --debug smith dfc model.onnx blueprint.yaml
        """
        smith_cli = create_cli(CLI_NAME_SMITH, include_admin=False)

        with smith_cli.make_context(CLI_NAME_SMITH, list(args), parent=ctx, obj=ctx.obj) as smith_ctx:
            smith_cli.invoke(smith_ctx)

    return smith


def create_cli(name: str, include_admin: bool = True) -> click.Group:
    @click.group(
        invoke_without_command=True,
        context_settings={'help_option_names': ['-h', '--help']}
    )
    @click.option('-b', '--build-dir', type=click.Path(path_type=Path),
                  help='Override build directory')
    @click.option('-c', '--config', type=click.Path(exists=True, path_type=Path),
                  help='Override configuration file')
    @click.option('-l', '--logs',
                  type=click.Choice(['error', 'warning', 'info', 'debug']),
                  default='warning',
                  metavar='LEVEL',
                  help='Set log level (error|warning|info|debug)')
    @click.option('--no-progress', is_flag=True,
                  help='Disable progress spinners and animations')
    @click.version_option(package_name='brainsmith', prog_name=name)
    @click.pass_context
    def cli(
        ctx: click.Context,
        build_dir: Path | None,
        config: Path | None,
        logs: str,
        no_progress: bool
    ) -> None:
        """Brainsmith - Hardware acceleration for neural networks."""
        context = ApplicationContext(
            config_file=config,
            no_progress=no_progress
        )

        if build_dir:
            context.overrides['build_dir'] = str(build_dir)

        context.load_configuration()

        ctx.obj = context

        setup_logging(level=logs)
        logger.debug(f"{name} CLI initialized with logs={logs}, no_progress={no_progress}")

        # Set ENV_QUIET for progress spinners when --no-progress is used
        if no_progress:
            os.environ[ENV_QUIET] = '1'

        # Legacy compatibility: export config to environment variables
        effective_config = context.get_effective_config()
        effective_config.export_to_environment(verbose=False)

        if ctx.invoked_subcommand is None:
            click.echo(ctx.get_help())

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

    _register_commands(cli, name, include_admin)

    return cli


def _run_cli(name: str, include_admin: bool) -> None:
    """Run CLI with consistent error handling."""
    try:
        cli = create_cli(name, include_admin=include_admin)
        cli()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted by user[/yellow]")
        sys.exit(EXIT_INTERRUPTED)
    except Exception as e:
        console.print(f"[red]Unexpected error:[/red] {e}")
        logging.exception(f"Unexpected error in {name} CLI")
        sys.exit(1)


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
