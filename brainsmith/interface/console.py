# Adapted from FINN-plus (https://github.com/eki-project/finn-plus)
# Copyright (c) 2020-2025, AMD/Xilinx and Paderborn University
# Licensed under BSD License - see FINN-plus repository for full license text

"""Console output utilities."""
from rich.console import Console

console = Console()


def error(msg: str) -> None:
    """Print an error"""
    console.print(f"[bold red]ERROR: [/bold red][red]{msg}[/red]")


def warning(msg: str) -> None:
    """Print a warning"""
    console.print(f"[bold orange1]WARNING: [/bold orange1][orange3]{msg}[/orange3]")


def status(msg: str) -> None:
    """Print a status message"""
    console.print(f"[bold cyan]STATUS: [/bold cyan][cyan]{msg}[/cyan]")


def success(msg: str) -> None:
    """Print a success message"""
    console.print(f"[bold green]SUCCESS: [/bold green][green]{msg}[/green]")


def debug(msg: str) -> None:
    """Print debug message if enabled."""
    from brainsmith.config import get_config
    if get_config().debug.enabled:
        console.print(f"[bold blue]DEBUG: [/bold blue][blue]{msg}[/blue]")