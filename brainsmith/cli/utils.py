# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""CLI utility functions for output formatting and user interaction.

Provides standardized functions for:
- Error reporting and exit handling
- Progress indicators
- Status formatting
- User messaging (success, warning, tips)
"""

import logging
import os
import sys
from contextlib import contextmanager
from typing import NoReturn, Iterator

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn, TaskID
from rich.panel import Panel

from .constants import ENV_QUIET, ExitCode

console = Console()


def error_exit(message: str, details: list[str] | None = None, code: int = ExitCode.USAGE) -> NoReturn:
    """Print error message and exit (defaults to EX_USAGE per BSD sysexits.h)."""
    console.print(f"[red]Error:[/red] {message}")

    if details:
        console.print("")
        for detail in details:
            console.print(f"  • {detail}")

    sys.exit(code)


@contextmanager
def progress_spinner(description: str, transient: bool = True, no_progress: bool = False) -> Iterator[TaskID | None]:
    """Display a progress spinner during long-running operations.

    Args:
        description: Text to display next to the spinner
        transient: If True, spinner disappears after completion
        no_progress: If True, disable spinner and yield None

    Yields:
        TaskID for progress updates, or None in quiet mode

    Example:
        >>> with progress_spinner("Installing dependencies...") as task:
        ...     if task:
        ...         task.update(description="Still working...")
    """
    if no_progress:
        yield None
        return

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=transient
    ) as progress:
        task = progress.add_task(description, total=None)
        try:
            yield task
        finally:
            progress.update(task, completed=True)


def show_panel(title: str, content: str, border_style: str = "blue") -> None:
    """Display a bordered panel with title and content."""
    console.print(Panel.fit(
        f"[bold]{title}[/bold]\n{content}",
        border_style=border_style
    ))


def format_status(status: str, is_success: bool) -> str:
    """Format status with color (green=success, red=failure)."""
    color = "green" if is_success else "red"
    return f"[{color}]{status}[/{color}]"


def format_warning_status(status: str) -> str:
    """Format status in yellow."""
    return f"[yellow]{status}[/yellow]"


def success(message: str) -> None:
    console.print(f"[green]✓[/green] {message}")


def warning(message: str, details: list[str] | None = None) -> None:
    """Print a warning message with optional detail bullets."""
    console.print(f"[yellow]Warning:[/yellow] {message}")
    if details:
        for detail in details:
            console.print(f"  • {detail}")


def tip(message: str) -> None:
    console.print(f"\n[yellow]Tip:[/yellow] {message}")


def confirm_or_abort(message: str, skip: bool = False) -> None:
    """Ask for confirmation or abort operation.

    Raises:
        SystemExit: If user declines confirmation
    """
    import click

    if not skip and not click.confirm(message):
        console.print("[yellow]Cancelled[/yellow]")
        sys.exit(0)
