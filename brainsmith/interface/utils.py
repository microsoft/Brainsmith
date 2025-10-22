# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Essential utilities for the Brainsmith CLI interface."""

import logging
import sys
from typing import Optional, List
from contextlib import contextmanager

from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.panel import Panel

console = Console()


def setup_logging(debug: bool = False) -> None:
    """Configure logging with Rich handler."""
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(console=console, rich_tracebacks=True)]
    )


def error_exit(message: str, details: Optional[List[str]] = None, code: int = 1) -> None:
    """Print an error message and exit with the given code.
    
    Args:
        message: The main error message
        details: Optional list of detail points to show
        code: Exit code (default: 1)
    """
    console.print(f"[red]Error:[/red] {message}")
    
    if details:
        console.print("")
        for detail in details:
            console.print(f"  • {detail}")
    
    sys.exit(code)


@contextmanager
def progress_spinner(description: str, transient: bool = True):
    """Context manager for showing a progress spinner.

    Args:
        description: Description to show next to the spinner
        transient: Whether to remove the spinner after completion

    Yields:
        The progress task object
    """
    import os

    # Skip spinner in quiet mode
    if os.environ.get('BSMITH_QUIET') == '1':
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
    """Display a panel with title and content.
    
    Args:
        title: Panel title
        content: Panel content
        border_style: Rich style for the border
    """
    console.print(Panel.fit(
        f"[bold]{title}[/bold]\n{content}",
        border_style=border_style
    ))


def format_status(status: str, is_good: bool) -> str:
    """Format a status string with appropriate color.
    
    Args:
        status: The status text
        is_good: Whether this is a positive status
        
    Returns:
        Formatted status string
    """
    color = "green" if is_good else "red"
    return f"[{color}]{status}[/{color}]"


def format_warning_status(status: str) -> str:
    """Format a warning status string.
    
    Args:
        status: The status text
        
    Returns:
        Formatted status string
    """
    return f"[yellow]{status}[/yellow]"


def success(message: str) -> None:
    """Print a success message."""
    console.print(f"[green]✓[/green] {message}")


def warning(message: str, details: Optional[List[str]] = None) -> None:
    """Print a warning message with optional details."""
    console.print(f"[yellow]Warning:[/yellow] {message}")
    if details:
        for detail in details:
            console.print(f"  • {detail}")


def tip(message: str) -> None:
    """Print a tip message."""
    console.print(f"\n[yellow]Tip:[/yellow] {message}")


def info(message: str, style: str = "cyan") -> None:
    """Print an info message."""
    console.print(f"[{style}]{message}[/{style}]")