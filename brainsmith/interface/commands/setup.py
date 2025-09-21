"""Setup and installation commands for the smith CLI."""

import os
import sys
import shutil
from typing import Optional, List, Dict, Tuple
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.panel import Panel

from brainsmith.config import get_config
from brainsmith.core.plugins.dependencies import DependencyManager

console = Console()

# Constants
EXCLUDED_BOARD_NAMES = {'deprecated', 'boards', 'board_files', 'Xilinx'}


@click.group()
def setup():
    """Install and configure Brainsmith dependencies."""
    pass


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
def all(force: bool):
    """Install all dependencies (cppsim, xsim, boards)."""
    console.print(Panel.fit(
        "[bold]Brainsmith Complete Setup[/bold]\n"
        "This will install all optional dependencies.",
        border_style="blue"
    ))
    
    # Run all setup tasks
    ctx = click.get_current_context()
    console.print("\n[bold cyan]1. Setting up C++ Simulation[/bold cyan]")
    ctx.invoke(cppsim, force=force)
    
    console.print("\n[bold cyan]2. Setting up Xilinx Simulation[/bold cyan]")
    ctx.invoke(xsim, force=force)
    
    console.print("\n[bold cyan]3. Downloading Board Files[/bold cyan]")
    ctx.invoke(boards, force=force)
    
    console.print("\n[green]✓ All setup tasks completed![/green]")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
def cppsim(force: bool):
    """Setup C++ simulation dependencies (cnpy, finn-hlslib)."""
    config = get_config()
    deps_mgr = DependencyManager(deps_dir=str(config.bsmith_deps_dir))
    
    # Check if both are already installed
    cnpy_installed = _is_cnpy_installed(deps_mgr)
    hlslib_installed = _are_hlslib_headers_installed(deps_mgr)
    
    if not force and cnpy_installed and hlslib_installed:
        console.print("  [yellow]![/yellow] C++ simulation dependencies already installed (use --force to reinstall)")
        return
    
    # Install what's needed
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        task = progress.add_task("Setting up C++ simulation dependencies...", total=None)
        
        try:
            # Use the group install method with quiet mode
            success = deps_mgr.setup_cppsim(force=force)
            if not success:
                progress.stop()
                console.print("  [red]✗[/red] Failed to setup C++ simulation dependencies")
                sys.exit(1)
                    
        except Exception as e:
            progress.stop()
            console.print(f"  [red]✗[/red] Failed to setup C++ simulation: {e}")
            
            # Check if it's likely a missing g++ issue
            if not shutil.which('g++'):
                console.print("\n[yellow]Note:[/yellow] C++ compiler (g++) is required for C++ simulation.")
                console.print("Install it with: [cyan]sudo apt install g++[/cyan]")
            sys.exit(1)
    
    # Show result
    console.print("  [green]✓[/green] C++ simulation dependencies installed successfully")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if already built')
def xsim(force: bool):
    """Setup Xilinx simulation (build finn-xsim with Vivado)."""
    config = get_config()
    
    # Check Vivado availability
    if not config.vivado_path:
        console.print("[red]Error:[/red] Vivado not found in configuration.")
        console.print("Please set up Vivado and update your configuration.")
        console.print("\nYou can set Vivado path using:")
        console.print("  - Environment variable: export BSMITH_XILINX__VIVADO_PATH=/path/to/vivado")
        console.print("  - Config file: Add xilinx.vivado_path to brainsmith_settings.yaml")
        sys.exit(1)
    
    deps_mgr = DependencyManager(deps_dir=str(config.bsmith_deps_dir))
    
    # Check if already built (xsi.so exists)
    xsi_so_path = config.bsmith_deps_dir / "finn" / "finn_xsi" / "xsi.so"
    
    if not force and xsi_so_path.exists():
        console.print("  [yellow]![/yellow] finn-xsim already built (use --force to rebuild)")
        return
    
    # Build it
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True  # This removes the progress bar after completion
    ) as progress:
        task = progress.add_task("Building finn-xsim module...", total=None)
        try:
            success = deps_mgr.build_finnxsim(force=force)
        except Exception as e:
            progress.stop()
            console.print(f"  [red]✗[/red] Failed to build finn-xsim: {e}")
            console.print("\nPlease ensure:")
            console.print("  • Vivado is properly installed")
            console.print("  • You have the required Vivado license")
            console.print("  • The Vivado path in configuration is correct")
            sys.exit(1)
    
    # Show result after progress completes
    if success:
        console.print("  [green]✓[/green] finn-xsim built successfully")
    else:
        console.print("  [red]✗[/red] Failed to build finn-xsim")
        sys.exit(1)


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force redownload even if already present')
@click.option('--repo', '-r', multiple=True, help='Specific repository to download (e.g., xilinx, avnet). Downloads all if not specified.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed list of all board files')
def boards(force: bool, repo: tuple, verbose: bool):
    """Download FPGA board definition files."""
    config = get_config()
    deps_mgr = DependencyManager(deps_dir=str(config.bsmith_deps_dir))
    
    # Check what's already downloaded
    board_files_dir = deps_mgr.deps_path / "board-files"
    existing_boards = []
    if board_files_dir.exists():
        existing_boards = [d.name for d in board_files_dir.iterdir() if d.is_dir() and any(d.iterdir())]
    
    # If specific repos requested, validate and check if they're already downloaded
    if repo:
        # Validate repo names
        valid_repos = ['avnet', 'xilinx', 'realdigital']
        invalid_repos = [r for r in repo if r not in valid_repos]
        
        if invalid_repos:
            console.print(f"  [red]✗[/red] Unknown board repositories: {', '.join(invalid_repos)}")
            console.print("\n  [yellow]Available repositories:[/yellow]")
            for r in valid_repos:
                console.print(f"      • {r}")
            sys.exit(1)
        
        repos_to_download = list(repo)
        already_have = [r for r in repos_to_download if r in existing_boards]
        
        if not force and already_have == repos_to_download:
            console.print(f"  [yellow]![/yellow] Requested repositories already downloaded:")
            
            # Show repos and optionally board details
            total_boards = 0
            boards_by_repo = {}
            
            for r in sorted(already_have):
                console.print(f"      • {r}")
                
                if verbose:
                    repo_path = board_files_dir / r
                    board_files = _find_board_files(repo_path)
                    total_boards += len(board_files)
                    boards_by_repo[r] = _extract_board_names(board_files, r)
            
            # In verbose mode, list all boards
            if verbose and boards_by_repo:
                console.print("\n  [dim]Board definitions:[/dim]")
                for r in sorted(boards_by_repo.keys()):
                    if boards_by_repo[r]:
                        console.print(f"\n      [yellow]{r}:[/yellow]")
                        for board in sorted(set(boards_by_repo[r])):
                            console.print(f"        • {board}")
                            
            console.print("\n  [dim]Use --force to redownload[/dim]")
            return
    else:
        # Check if all boards are already downloaded
        if not force and existing_boards:
            # Count total boards and collect board names
            total_boards = 0
            boards_by_repo = {}
            
            for r in existing_boards:
                repo_path = board_files_dir / r
                board_files = _find_board_files(repo_path)
                total_boards += len(board_files)
                
                # Collect board names for verbose mode
                if verbose:
                    boards_by_repo[r] = _extract_board_names(board_files, r)
                
            console.print(f"  [yellow]![/yellow] Board files already downloaded:")
            for board in sorted(existing_boards):
                console.print(f"      • {board}")
            if total_boards > 0:
                console.print(f"  [dim]{total_boards} board definitions available[/dim]")
                
            # In verbose mode, list all boards
            if verbose and boards_by_repo:
                console.print("\n  [dim]Board definitions by repository:[/dim]")
                for r in sorted(boards_by_repo.keys()):
                    if boards_by_repo[r]:
                        console.print(f"\n      [yellow]{r}:[/yellow]")
                        for board in sorted(set(boards_by_repo[r])):
                            console.print(f"        • {board}")
                            
            console.print("\n  [dim]Use --force to redownload[/dim]")
            return
    
    # Download boards
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True
    ) as progress:
        if repo:
            task = progress.add_task(f"Downloading {len(repo)} board repositories...", total=None)
        else:
            task = progress.add_task("Downloading board definition files...", total=None)
        
        try:
            success = deps_mgr.download_board_files(boards=list(repo) if repo else None, quiet=True)
            if not success:
                progress.stop()
                console.print("  [red]✗[/red] Failed to download board files")
                sys.exit(1)
                
        except Exception as e:
            progress.stop()
            console.print(f"  [red]✗[/red] Failed to download board files: {e}")
            sys.exit(1)
    
    # Show what was downloaded
    console.print("  [green]✓[/green] Board definition files downloaded:")
    
    # List repositories - if specific repos were requested, only show those
    if repo:
        # Check what actually got downloaded
        new_existing = [d.name for d in board_files_dir.iterdir() if d.is_dir() and any(d.iterdir())]
        downloaded_repos = [r for r in repo if r in new_existing]
    else:
        downloaded_repos = [d.name for d in board_files_dir.iterdir() if d.is_dir() and any(d.iterdir())]
    
    # Collect board information
    total_boards = 0
    boards_by_repo = {}
    
    for r in sorted(downloaded_repos):
        console.print(f"      • {r}")
        repo_path = board_files_dir / r
        # Look for board.xml files in board directories (handle various depths)
        board_files = _find_board_files(repo_path)
        total_boards += len(board_files)
        
        # Collect board names for verbose mode
        if verbose:
            boards_by_repo[r] = _extract_board_names(board_files, r)
    
    # Show board count
    if total_boards > 0:
        console.print(f"  [dim]{total_boards} board definitions downloaded[/dim]")
    
    # In verbose mode, list all boards
    if verbose and boards_by_repo:
        console.print("\n  [dim]Board definitions by repository:[/dim]")
        for r in sorted(boards_by_repo.keys()):
            if boards_by_repo[r]:
                console.print(f"\n      [yellow]{r}:[/yellow]")
                for board in sorted(set(boards_by_repo[r])):  # Use set to remove duplicates
                    console.print(f"        • {board}")


@setup.command()
def check():
    """Check the status of all setup components."""
    config = get_config()
    deps_mgr = DependencyManager(deps_dir=str(config.bsmith_deps_dir))
    
    table = Table(title="Brainsmith Setup Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Location", style="dim")
    
    # Check C++ Simulation
    cnpy_status = "Installed" if _is_cnpy_installed(deps_mgr) else "Not installed"
    cnpy_color = "green" if cnpy_status == "Installed" else "red"
    table.add_row("cnpy", f"[{cnpy_color}]{cnpy_status}[/{cnpy_color}]", "deps/cnpy")
    
    hlslib_status = "Installed" if _are_hlslib_headers_installed(deps_mgr) else "Not installed"
    hlslib_color = "green" if hlslib_status == "Installed" else "red"
    table.add_row("finn-hlslib headers", f"[{hlslib_color}]{hlslib_status}[/{hlslib_color}]", 
                  "deps/finn-hlslib")
    
    # Check RTL Simulation
    finnxsim_status = "Built" if _is_finnxsim_built(deps_mgr) else "Not built"
    finnxsim_color = "green" if finnxsim_status == "Built" else "red"
    table.add_row("finn-xsim", f"[{finnxsim_color}]{finnxsim_status}[/{finnxsim_color}]", 
                  "deps/finn/finn_xsi")
    
    # Check Vivado
    if config.vivado_path:
        vivado_status = "Found"
        vivado_color = "green"
        vivado_details = []
        
        # Check if settings64.sh has been sourced
        if "XILINX_VIVADO" not in os.environ:
            vivado_details.append("⚠️  Not sourced")
            vivado_color = "yellow"
        else:
            vivado_details.append("✓ Sourced")
        
        # Show version
        if config.xilinx_version:
            vivado_details.append(f"v{config.xilinx_version}")
            
        vivado_path = f"{config.vivado_path}"
        if vivado_details:
            vivado_status = f"Found ({', '.join(vivado_details)})"
    else:
        vivado_status = "Not found"
        vivado_color = "red"
        vivado_path = "Not configured"
        
    table.add_row("Vivado", f"[{vivado_color}]{vivado_status}[/{vivado_color}]", vivado_path)
    
    # Check Vitis HLS
    if config.vitis_hls_path:
        hls_status = "Found"
        hls_color = "green"
        hls_details = []
        
        # Check if sourced
        if "XILINX_HLS" not in os.environ and "XILINX_VITIS_HLS" not in os.environ:
            hls_details.append("⚠️  Not sourced")
            hls_color = "yellow"
        else:
            hls_details.append("✓ Sourced")
        
        # Show version
        if config.xilinx_version:
            hls_details.append(f"v{config.xilinx_version}")
            
        hls_path = str(config.vitis_hls_path)
        if hls_details:
            hls_status = f"Found ({', '.join(hls_details)})"
    else:
        hls_status = "Not found"
        hls_color = "yellow"
        hls_path = "Not configured"
        
    table.add_row("Vitis HLS", f"[{hls_color}]{hls_status}[/{hls_color}]", hls_path)
    
    # Check boards
    board_count = _count_downloaded_boards(deps_mgr)
    board_status = f"{board_count} repositories" if board_count > 0 else "None"
    board_color = "green" if board_count > 0 else "yellow"
    table.add_row("Board files", f"[{board_color}]{board_status}[/{board_color}]", "deps/board-files")
    
    console.print(table)
    
    # Print recommendations
    if cnpy_status == "Not installed" or hlslib_status == "Not installed":
        console.print("\n[yellow]Tip:[/yellow] Run 'smith setup cppsim' to install C++ simulation dependencies")
    
    if finnxsim_status == "Not built" and vivado_status == "Found":
        console.print("\n[yellow]Tip:[/yellow] Run 'smith setup xsim' to build Xilinx simulation support")
    
    if board_count == 0:
        console.print("\n[yellow]Tip:[/yellow] Run 'smith setup boards' to download board definition files")


# Helper functions to check installation status
def _is_cnpy_installed(deps_mgr) -> bool:
    """Check if cnpy is installed."""
    cnpy_dir = deps_mgr.deps_path / "cnpy"
    return (cnpy_dir / "cnpy.h").exists()


def _are_hlslib_headers_installed(deps_mgr) -> bool:
    """Check if finn-hlslib headers are installed."""
    hlslib_dir = deps_mgr.deps_path / "finn-hlslib"
    return (hlslib_dir / "tb").exists()


def _is_finnxsim_built(deps_mgr) -> bool:
    """Check if finn-xsim module is built."""
    # Check for the actual build output: xsi.so
    xsi_so = deps_mgr.deps_path / "finn" / "finn_xsi" / "xsi.so"
    return xsi_so.exists()


def _count_downloaded_boards(deps_mgr) -> int:
    """Count number of downloaded board repositories."""
    board_files_dir = deps_mgr.deps_path / "board-files"
    if not board_files_dir.exists():
        return 0
    return len([d for d in board_files_dir.iterdir() if d.is_dir() and any(d.iterdir())])


def _find_board_files(repo_path: Path) -> List[Path]:
    """Find all board.xml files in a repository at various depths."""
    return list(repo_path.glob("*/*/board.xml")) + \
           list(repo_path.glob("*/*/*/board.xml")) + \
           list(repo_path.glob("*/*/*/*/board.xml"))


def _extract_board_names(board_files: List[Path], repo_name: str) -> List[str]:
    """Extract board names from board.xml file paths."""
    board_names = []
    for board_file in board_files:
        parts = board_file.parts
        if 'board.xml' in parts:
            idx = parts.index('board.xml')
            if idx >= 2:
                board_name = parts[idx-2]
                if board_name not in EXCLUDED_BOARD_NAMES and board_name != repo_name:
                    board_names.append(board_name)
    return board_names