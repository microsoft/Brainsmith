# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import logging
import os
import sys
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

from .constants import ENV_QUIET, EXIT_INTERRUPTED

console = Console()


def setup_logging(level: str = "warning") -> None:
    from brainsmith._internal.logging import setup_logging as core_setup_logging
    core_setup_logging(level=level)


def error_exit(message: str, details: list[str] | None = None, code: int = 1) -> None:
    console.print(f"[red]Error:[/red] {message}")

    if details:
        console.print("")
        for detail in details:
            console.print(f"  • {detail}")

    sys.exit(code)


@contextmanager
def progress_spinner(description: str, transient: bool = True):
    if os.environ.get(ENV_QUIET) == '1':
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
    console.print(Panel.fit(
        f"[bold]{title}[/bold]\n{content}",
        border_style=border_style
    ))


def format_status(status: str, is_success: bool) -> str:
    """Format status with color: green for success, red for error."""
    color = "green" if is_success else "red"
    return f"[{color}]{status}[/{color}]"


def format_warning_status(status: str) -> str:
    return f"[yellow]{status}[/yellow]"


def success(message: str) -> None:
    """Print a success message with green checkmark."""
    console.print(f"[green]✓[/green] {message}")


def warning(message: str, details: list[str] | None = None) -> None:
    """Print a warning message with optional detail bullets."""
    console.print(f"[yellow]Warning:[/yellow] {message}")
    if details:
        for detail in details:
            console.print(f"  • {detail}")


def tip(message: str) -> None:
    """Print a tip message for user guidance."""
    console.print(f"\n[yellow]Tip:[/yellow] {message}")


def info(message: str, style: str = "cyan") -> None:
    """Print an informational message with custom styling."""
    console.print(f"[{style}]{message}[/{style}]")
