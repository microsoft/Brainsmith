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

from .constants import ENV_QUIET, EX_INTERRUPTED, EX_USAGE

console = Console()


def error_exit(message: str, details: list[str] | None = None, code: int = EX_USAGE) -> NoReturn:
    """Print error message and exit. Shows bullet list if details provided.

    Defaults to EX_USAGE (64) per BSD sysexits.h standard.
    """
    console.print(f"[red]Error:[/red] {message}")

    if details:
        console.print("")
        for detail in details:
            console.print(f"  • {detail}")

    sys.exit(code)


class NoOpTask:
    """Placeholder for progress tasks in quiet mode. Accepts all method calls."""
    def __getattr__(self, name):
        return lambda *args, **kwargs: None


@contextmanager
def progress_spinner(description: str, transient: bool = True) -> Iterator[TaskID | NoOpTask]:
    """Display a progress spinner during long-running operations.

    In quiet mode (ENV_QUIET=1), returns NoOpTask that accepts all method calls.
    Otherwise, yields a Rich TaskID for progress updates.

    Args:
        description: Text to display next to the spinner
        transient: If True, spinner disappears after completion

    Yields:
        TaskID for progress updates, or NoOpTask in quiet mode

    Example:
        >>> with progress_spinner("Installing dependencies...") as task:
        ...     task.update(description="Still working...")  # Works in both modes
    """
    if os.environ.get(ENV_QUIET) == '1':
        yield NoOpTask()
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
    """Display a bordered panel with title and content.

    Args:
        title: Panel title
        content: Panel content
        border_style: Rich color/style for border (default: "blue")
    """
    console.print(Panel.fit(
        f"[bold]{title}[/bold]\n{content}",
        border_style=border_style
    ))


def format_status(status: str, is_success: bool) -> str:
    """Format status text with color based on success/failure.

    Args:
        status: Status text to format
        is_success: If True, use green; if False, use red

    Returns:
        Rich-formatted status string
    """
    color = "green" if is_success else "red"
    return f"[{color}]{status}[/{color}]"


def format_warning_status(status: str) -> str:
    """Format status text with warning color (yellow).

    Args:
        status: Status text to format

    Returns:
        Rich-formatted status string in yellow
    """
    return f"[yellow]{status}[/yellow]"


def success(message: str) -> None:
    """Print success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def warning(message: str, details: list[str] | None = None) -> None:
    """Print a warning message with optional detail bullets.

    Args:
        message: Main warning message
        details: Optional list of additional detail lines
    """
    console.print(f"[yellow]Warning:[/yellow] {message}")
    if details:
        for detail in details:
            console.print(f"  • {detail}")


def tip(message: str) -> None:
    """Print a tip message for user guidance.

    Args:
        message: Tip message to display
    """
    console.print(f"\n[yellow]Tip:[/yellow] {message}")


def confirm_or_abort(message: str, skip: bool = False) -> None:
    """Ask for confirmation or abort operation.

    Args:
        message: Confirmation prompt
        skip: If True, skip confirmation and proceed

    Raises:
        SystemExit: If user declines confirmation
    """
    import click

    if not skip and not click.confirm(message):
        console.print("[yellow]Cancelled[/yellow]")
        sys.exit(0)
