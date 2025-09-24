"""Setup and installation commands for the smith CLI."""

# Standard library imports
import os
import shutil
from pathlib import Path
from typing import Optional, List

# Third-party imports
import click
from rich.table import Table

# Local imports
from brainsmith.config import get_config
from brainsmith.core.plugins.dependencies import DependencyManager
from ..utils import (
    console, error_exit, success, warning, tip, 
    progress_spinner, show_panel, format_status, format_warning_status
)

# Constants
EXCLUDED_BOARD_NAMES = {'deprecated', 'boards', 'board_files', 'Xilinx'}


@click.group()
def setup():
    """Install and configure Brainsmith dependencies."""
    pass


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
def all(force: bool) -> None:
    """Install all dependencies (cppsim, xsim, boards).
    
    Args:
        force: Whether to force reinstallation
    """
    show_panel(
        "Brainsmith Complete Setup",
        "This will install all optional dependencies."
    )
    
    # Run all setup tasks
    ctx = click.get_current_context()
    console.print("\n[bold cyan]1. Setting up C++ Simulation[/bold cyan]")
    ctx.invoke(cppsim, force=force)
    
    console.print("\n[bold cyan]2. Setting up Xilinx Simulation[/bold cyan]")
    ctx.invoke(xsim, force=force)
    
    console.print("\n[bold cyan]3. Downloading Board Files[/bold cyan]")
    ctx.invoke(boards, force=force)
    
    success("All setup tasks completed!")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
def cppsim(force: bool) -> None:
    """Setup C++ simulation dependencies (cnpy, finn-hlslib).
    
    Args:
        force: Whether to force reinstallation
    """
    config = get_config()
    
    deps_mgr = DependencyManager(deps_dir=config.deps_dir)
    
    # Check if both are already installed
    cnpy_installed = _is_cnpy_installed(deps_mgr)
    hlslib_installed = _are_hlslib_headers_installed(deps_mgr)
    
    if not force and cnpy_installed and hlslib_installed:
        warning("C++ simulation dependencies already installed (use --force to reinstall)")
        return
    
    # Install what's needed
    with progress_spinner("Setting up C++ simulation dependencies...") as task:
        try:
            # Use the group install method with quiet mode
            result = deps_mgr.setup_cppsim(force=force)
            if not result:
                error_exit("Failed to setup C++ simulation dependencies")
                    
        except Exception as e:
            # Check if it's likely a missing g++ issue
            details = []
            if not shutil.which('g++'):
                details = [
                    "C++ compiler (g++) is required for C++ simulation.",
                    "Install it with: sudo apt install g++"
                ]
            error_exit(f"Failed to setup C++ simulation: {e}", details=details)
    
    # Show result
    success("C++ simulation dependencies installed successfully")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force rebuild even if already built')
def xsim(force: bool) -> None:
    """Setup Xilinx simulation (build finn-xsim with Vivado).
    
    Args:
        force: Whether to force rebuild
    """
    from brainsmith.config import export_to_environment
    
    config = get_config()
    
    # Check Vivado availability
    if not config.effective_vivado_path:
        error_exit(
            "Vivado not found in configuration.",
            details=[
                "Please set up Vivado and update your configuration.",
                "Set Vivado path using:",
                "  - Environment variable: export BSMITH_XILINX__VIVADO_PATH=/path/to/vivado",
                "  - Config file: Add xilinx.vivado_path to brainsmith_settings.yaml"
            ]
        )
    
    # Export environment variables before building
    export_to_environment(config)
    
    deps_mgr = DependencyManager(deps_dir=config.deps_dir)
    
    # Check if already built (xsi.so exists)
    xsi_so_path = config.deps_dir / "finn" / "finn_xsi" / "xsi.so"
    
    if not force and xsi_so_path.exists():
        warning("finn-xsim already built (use --force to rebuild)")
        return
    
    # Build it
    with progress_spinner("Building finn-xsim module...") as task:
        try:
            result = deps_mgr.build_finnxsim(force=force, quiet=False)
            if not result:
                error_exit("Failed to build finn-xsim")
        except Exception as e:
            error_exit(
                f"Failed to build finn-xsim: {e}",
                details=[
                    "Vivado is properly installed",
                    "You have the required Vivado license",
                    "The Vivado path in configuration is correct"
                ]
            )
    
    # Show result after progress completes
    success("finn-xsim built successfully")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force redownload even if already present')
@click.option('--repo', '-r', multiple=True, help='Specific repository to download (e.g., xilinx, avnet). Downloads all if not specified.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed list of all board files')
def boards(force: bool, repo: tuple, verbose: bool) -> None:
    """Download FPGA board definition files.
    
    Args:
        force: Whether to force redownload
        repo: Specific repositories to download
        verbose: Whether to show detailed board list
    """
    config = get_config()
    
    deps_mgr = DependencyManager(deps_dir=config.deps_dir)
    
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
            error_exit(
                f"Unknown board repositories: {', '.join(invalid_repos)}",
                details=[f"Available repositories: {', '.join(valid_repos)}"]
            )
        
        repos_to_download = list(repo)
        already_have = [r for r in repos_to_download if r in existing_boards]
        
        if not force and already_have == repos_to_download:
            warning("Requested repositories already downloaded:")
            
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
                            
            console.print("\n[dim]Use --force to redownload[/dim]")
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
                
            warning("Board files already downloaded:")
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
                            
            console.print("\n[dim]Use --force to redownload[/dim]")
            return
    
    # Download boards
    description = (f"Downloading {len(repo)} board repositories..." if repo 
                   else "Downloading board definition files...")
    
    with progress_spinner(description) as task:
        try:
            result = deps_mgr.download_board_files(boards=list(repo) if repo else None, quiet=True)
            if not result:
                error_exit("Failed to download board files")
                
        except Exception as e:
            error_exit(f"Failed to download board files: {e}")
    
    # Show what was downloaded
    success("Board definition files downloaded:")
    
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
def check() -> None:
    """Check the status of all setup components.
    
    Displays a table showing the installation status of all
    Brainsmith dependencies and tools.
    """
    config = get_config()
    deps_mgr = DependencyManager(deps_dir=config.deps_dir)
    
    table = Table(title="Brainsmith Setup Status")
    table.add_column("Component", style="cyan")
    table.add_column("Status", style="green")
    table.add_column("Location", style="dim")
    
    # Check C++ Simulation
    cnpy_installed = _is_cnpy_installed(deps_mgr)
    cnpy_status = format_status("Installed" if cnpy_installed else "Not installed", cnpy_installed)
    table.add_row("cnpy", cnpy_status, "deps/cnpy")
    
    hlslib_installed = _are_hlslib_headers_installed(deps_mgr)
    hlslib_status = format_status("Installed" if hlslib_installed else "Not installed", hlslib_installed)
    table.add_row("finn-hlslib headers", hlslib_status, "deps/finn-hlslib")
    
    # Check RTL Simulation
    finnxsim_built = _is_finnxsim_built(deps_mgr)
    finnxsim_status = format_status("Built" if finnxsim_built else "Not built", finnxsim_built)
    table.add_row("finn-xsim", finnxsim_status, "deps/finn/finn_xsi")
    
    # Check Vivado
    if config.effective_vivado_path:
        vivado_details = []
        
        # Check if settings64.sh has been sourced
        is_sourced = "XILINX_VIVADO" in os.environ
        if not is_sourced:
            vivado_details.append("⚠️  Not sourced")
        else:
            vivado_details.append("✓ Sourced")
        
        # Show version
        if config.xilinx_version:
            vivado_details.append(f"v{config.xilinx_version}")
            
        vivado_path = str(config.effective_vivado_path)
        status_text = f"Found ({', '.join(vivado_details)})" if vivado_details else "Found"
        vivado_status = format_warning_status(status_text) if not is_sourced else format_status(status_text, True)
    else:
        vivado_status = format_status("Not found", False)
        vivado_path = "Not configured"
        
    table.add_row("Vivado", vivado_status, vivado_path)
    
    # Check Vitis HLS
    if config.effective_vitis_hls_path:
        hls_details = []
        
        # Check if sourced
        is_sourced = "XILINX_HLS" in os.environ or "XILINX_VITIS_HLS" in os.environ
        if not is_sourced:
            hls_details.append("⚠️  Not sourced")
        else:
            hls_details.append("✓ Sourced")
        
        # Show version
        if config.xilinx_version:
            hls_details.append(f"v{config.xilinx_version}")
            
        hls_path = str(config.effective_vitis_hls_path)
        status_text = f"Found ({', '.join(hls_details)})" if hls_details else "Found"
        hls_status = format_warning_status(status_text) if not is_sourced else format_status(status_text, True)
    else:
        hls_status = format_warning_status("Not found")
        hls_path = "Not configured"
        
    table.add_row("Vitis HLS", hls_status, hls_path)
    
    # Check boards
    board_count = _count_downloaded_boards(deps_mgr)
    board_status_text = f"{board_count} repositories" if board_count > 0 else "None"
    board_status = format_status(board_status_text, True) if board_count > 0 else format_warning_status(board_status_text)
    table.add_row("Board files", board_status, "deps/board-files")
    
    console.print(table)
    
    # Print recommendations
    if not cnpy_installed or not hlslib_installed:
        tip("Run 'smith setup cppsim' to install C++ simulation dependencies")
    
    if not finnxsim_built and config.effective_vivado_path:
        tip("Run 'smith setup xsim' to build Xilinx simulation support")
    
    if board_count == 0:
        tip("Run 'smith setup boards' to download board definition files")


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