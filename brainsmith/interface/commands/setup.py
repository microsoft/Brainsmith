"""Setup and installation commands for the smith CLI."""

# Standard library imports
import logging
import os
import shutil
from pathlib import Path
from typing import Optional, List, Dict, Tuple

# Third-party imports
import click
from rich.table import Table

logger = logging.getLogger(__name__)

# Local imports
from brainsmith.config import get_config
from brainsmith.core.plugins.dependencies import DependencyManager
from ..utils import (
    console, error_exit, success, warning, tip,
    progress_spinner, show_panel, format_status, format_warning_status
)
# Board file utilities inlined from board_utils.py
# These functions are only used by the setup command

# Board names to exclude from listings
EXCLUDED_BOARD_NAMES = {'deprecated', 'boards', 'board_files', 'Xilinx'}


@click.group()
def setup():
    """Install and configure Brainsmith dependencies."""
    pass


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
def all(force: bool) -> None:
    """Install all dependencies (cppsim, xsim, boards)."""
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
    """Setup C++ simulation dependencies (cnpy, finn-hlslib)."""
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
    """Setup Xilinx simulation (build finn-xsim with Vivado)."""
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
    
    # Build it (the build function handles force flag internally)
    with progress_spinner("Building finn-xsim module...") as task:
        try:
            result = deps_mgr.build_finnxsim(force=force, quiet=True)
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
    
    success("finn-xsim built successfully")


def _show_board_summary(boards_by_repo: Dict[str, List[str]], title: str) -> None:
    """Display a summary of boards organized by repository.
    
    Args:
        boards_by_repo: Dictionary mapping repo names to board lists
        title: Title to display above the board list
    """
    if boards_by_repo:
        console.print(f"\n  [dim]{title}:[/dim]")
        for repo_name in sorted(boards_by_repo.keys()):
            if boards_by_repo[repo_name]:
                console.print(f"\n      [yellow]{repo_name}:[/yellow]")
                for board in sorted(set(boards_by_repo[repo_name])):
                    console.print(f"        • {board}")


def _handle_existing_boards(board_files_dir: Path, requested_repos: List[str], 
                           verbose: bool, force: bool) -> bool:
    """Handle the case where boards are already downloaded.
    
    Args:
        board_files_dir: Path to board files directory
        requested_repos: List of specifically requested repos (empty for all)
        verbose: Whether to show detailed board list
        force: Whether to force redownload
        
    Returns:
        True if should continue with download, False otherwise
    """
    existing_repos = list_downloaded_repositories(board_files_dir)
    
    # Check if requested repos are already present
    if requested_repos:
        already_have = [r for r in requested_repos if r in existing_repos]
        if not force and already_have == requested_repos:
            warning("Requested repositories already downloaded:")
            for r in sorted(already_have):
                console.print(f"      • {r}")
            
            if verbose:
                boards_by_repo = get_board_summary(board_files_dir)
                filtered = {r: boards_by_repo[r] for r in already_have if r in boards_by_repo}
                _show_board_summary(filtered, "Board definitions")
            
            console.print("\n[dim]Use --force to redownload[/dim]")
            return False
    else:
        # Check if any boards exist
        if not force and existing_repos:
            boards_by_repo = get_board_summary(board_files_dir)
            total_boards = sum(len(boards) for boards in boards_by_repo.values())
            
            warning("Board files already downloaded:")
            for repo in sorted(existing_repos):
                console.print(f"      • {repo}")
            if total_boards > 0:
                console.print(f"  [dim]{total_boards} board definitions available[/dim]")
            
            if verbose:
                _show_board_summary(boards_by_repo, "Board definitions by repository")
            
            console.print("\n[dim]Use --force to redownload[/dim]")
            return False
    
    return True


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force redownload even if already present')
@click.option('--repo', '-r', multiple=True, help='Specific repository to download (e.g., xilinx, avnet). Downloads all if not specified.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed list of all board files')
def boards(force: bool, repo: tuple, verbose: bool) -> None:
    """Download FPGA board definition files."""
    config = get_config()
    deps_mgr = DependencyManager(deps_dir=config.deps_dir)
    board_files_dir = deps_mgr.deps_path / "board-files"
    
    # Validate repository names if specified
    if repo:
        valid_repos, invalid_repos = validate_repository_names(list(repo))
        if invalid_repos:
            error_exit(
                f"Unknown board repositories: {', '.join(invalid_repos)}",
                details=["Available repositories: avnet, xilinx, realdigital"]
            )
        repos_to_download = valid_repos
    else:
        repos_to_download = []
    
    # Check if boards already exist
    if not _handle_existing_boards(board_files_dir, repos_to_download, verbose, force):
        return
    
    # Download boards
    description = (f"Downloading {len(repos_to_download)} board repositories..." 
                  if repos_to_download
                  else "Downloading board definition files...")
    
    with progress_spinner(description) as task:
        try:
            boards_arg = repos_to_download if repos_to_download else None
            result = deps_mgr.download_board_files(boards=boards_arg, force=force, quiet=True)
            if not result:
                error_exit("Failed to download board files")
        except Exception as e:
            error_exit(f"Failed to download board files: {e}")
    
    # Show what was downloaded
    success("Board definition files downloaded:")
    
    # Get summary of downloaded boards
    downloaded_repos = list_downloaded_repositories(board_files_dir)
    if repos_to_download:
        # Filter to show only requested repos that were downloaded
        downloaded_repos = [r for r in downloaded_repos if r in repos_to_download]
    
    # Display downloaded repositories
    boards_by_repo = {}
    total_boards = 0
    
    for repo_name in sorted(downloaded_repos):
        console.print(f"      • {repo_name}")
        repo_path = board_files_dir / repo_name
        board_files = find_board_files(repo_path)
        board_count = len(board_files)
        total_boards += board_count
        
        if verbose:
            boards = extract_board_names(board_files, repo_name)
            if boards:
                boards_by_repo[repo_name] = boards
    
    # Show summary
    if total_boards > 0:
        console.print(f"  [dim]{total_boards} board definitions downloaded[/dim]")
    
    # Show detailed board list in verbose mode
    if verbose:
        _show_board_summary(boards_by_repo, "Board definitions by repository")


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
        tip("Run 'brainsmith setup cppsim' to install C++ simulation dependencies")
    
    if not finnxsim_built and config.effective_vivado_path:
        tip("Run 'brainsmith setup xsim' to build Xilinx simulation support")
    
    if board_count == 0:
        tip("Run 'brainsmith setup boards' to download board definition files")


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
    return len(list_downloaded_repositories(board_files_dir))


# Board file utilities (previously in board_utils.py)

def find_board_files(repo_path: Path) -> List[Path]:
    """Find all board.xml files in a repository at various depths.
    
    Args:
        repo_path: Path to the repository to search
        
    Returns:
        List of paths to board.xml files
    """
    board_files = []
    
    # Search at different depths where board files might be located
    patterns = [
        "*/*/board.xml",
        "*/*/*/board.xml",
        "*/*/*/*/board.xml"
    ]
    
    for pattern in patterns:
        board_files.extend(repo_path.glob(pattern))
    
    logger.debug(f"Found {len(board_files)} board files in {repo_path}")
    return board_files


def extract_board_names(board_files: List[Path], repo_name: str) -> List[str]:
    """Extract board names from board.xml file paths.
    
    Args:
        board_files: List of paths to board.xml files
        repo_name: Name of the repository (to exclude from board names)
        
    Returns:
        List of board names
    """
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


def get_board_summary(board_files_dir: Path) -> Dict[str, List[str]]:
    """Get a summary of all boards organized by repository.
    
    Args:
        board_files_dir: Path to the board-files directory
        
    Returns:
        Dictionary mapping repository names to lists of board names
    """
    boards_by_repo = {}
    
    if not board_files_dir.exists():
        return boards_by_repo
    
    for repo_dir in board_files_dir.iterdir():
        if repo_dir.is_dir() and any(repo_dir.iterdir()):
            board_files = find_board_files(repo_dir)
            if board_files:
                board_names = extract_board_names(board_files, repo_dir.name)
                if board_names:
                    boards_by_repo[repo_dir.name] = sorted(set(board_names))
    
    return boards_by_repo


def list_downloaded_repositories(board_files_dir: Path) -> List[str]:
    """List all downloaded board repositories.
    
    Args:
        board_files_dir: Path to the board-files directory
        
    Returns:
        List of repository names that contain boards
    """
    if not board_files_dir.exists():
        return []
    
    repos = []
    for repo_dir in board_files_dir.iterdir():
        if repo_dir.is_dir() and any(repo_dir.iterdir()):
            # Check if it actually contains board files
            if find_board_files(repo_dir):
                repos.append(repo_dir.name)
    
    return sorted(repos)


def validate_repository_names(repo_names: List[str]) -> Tuple[List[str], List[str]]:
    """Validate repository names against known repositories.
    
    Args:
        repo_names: List of repository names to validate
        
    Returns:
        Tuple of (valid_repos, invalid_repos)
    """
    valid_repos = ['avnet', 'xilinx', 'realdigital']
    
    valid = [r for r in repo_names if r in valid_repos]
    invalid = [r for r in repo_names if r not in valid_repos]
    
    return valid, invalid


