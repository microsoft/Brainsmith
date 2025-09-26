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
from brainsmith.utils.dependencies import DependencyManager, BoardManager
from ..utils import (
    console, error_exit, success, warning, tip,
    progress_spinner, show_panel, format_status, format_warning_status
)


def confirm_removal(items: List[str], description: str, skip_confirm: bool = False) -> bool:
    """Ask user to confirm removal of items.
    
    Args:
        items: List of items that will be removed
        description: Description of what's being removed
        skip_confirm: Skip confirmation prompt if True
        
    Returns:
        True if user confirms, False otherwise
    """
    warning(f"The following {description} will be removed:")
    for item in sorted(items):
        console.print(f"      • {item}")
    
    if skip_confirm:
        return True
        
    console.print("\n[yellow]Are you sure you want to remove these items?[/yellow]")
    response = console.input("[dim](y/N)[/dim] ").strip().lower()
    return response == 'y'


@click.group()
def setup():
    """Install and configure Brainsmith dependencies."""
    pass


@setup.command(name='all')
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
@click.option('--remove', '-r', is_flag=True, help='Remove all dependencies')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def setup_all(force: bool, remove: bool, yes: bool) -> None:
    """Install all dependencies (cppsim, xsim, boards)."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")
        
    if remove:
        show_panel(
            "Brainsmith Complete Removal",
            "This will remove all optional dependencies."
        )
    else:
        show_panel(
            "Brainsmith Complete Setup",
            "This will install all optional dependencies."
        )
    
    # Run all setup tasks
    ctx = click.get_current_context()
    console.print(f"\n[bold cyan]1. {'Removing' if remove else 'Setting up'} C++ Simulation[/bold cyan]")
    ctx.invoke(cppsim, force=force, remove=remove, yes=yes)
    
    console.print(f"\n[bold cyan]2. {'Removing' if remove else 'Setting up'} Xilinx Simulation[/bold cyan]")
    ctx.invoke(xsim, force=force, remove=remove, yes=yes)
    
    console.print(f"\n[bold cyan]3. {'Removing' if remove else 'Downloading'} Board Files[/bold cyan]")
    ctx.invoke(boards, force=force, remove=remove, repo=(), verbose=False, yes=yes)
    
    success(f"All {'removal' if remove else 'setup'} tasks completed!")


@setup.command()
@click.option('--force', '-f', is_flag=True, help='Force reinstallation even if already installed')
@click.option('--remove', '-r', is_flag=True, help='Remove C++ simulation dependencies')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def cppsim(force: bool, remove: bool, yes: bool) -> None:
    """Setup C++ simulation dependencies (cnpy, finn-hlslib)."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")
        
    deps_mgr = DependencyManager()
    
    if remove:
        # Get list of installed cppsim dependencies
        from brainsmith.utils.dependencies import DEPENDENCIES
        cppsim_deps = [k for k, v in DEPENDENCIES.items() 
                       if v.get('group') == 'cppsim' and (deps_mgr.deps_dir / k).exists()]
        
        if not cppsim_deps:
            warning("No C++ simulation dependencies are installed")
            return
            
        if not confirm_removal(cppsim_deps, "C++ simulation dependencies", skip_confirm=yes):
            console.print("Removal cancelled")
            return
            
        # Remove dependencies
        with progress_spinner("Removing C++ simulation dependencies...") as task:
            results = deps_mgr.remove_group('cppsim', quiet=True)
            if all(results.values()):
                success("C++ simulation dependencies removed successfully")
            else:
                failed = [k for k, v in results.items() if not v]
                error_exit(f"Failed to remove some dependencies: {', '.join(failed)}")
        return
    
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
            results = deps_mgr.install_group('cppsim', force=force, quiet=True)
            if not all(results.values()):
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
@click.option('--remove', '-r', is_flag=True, help='Remove Xilinx simulation dependencies')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def xsim(force: bool, remove: bool, yes: bool) -> None:
    """Setup Xilinx simulation (build finn-xsim with Vivado)."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")
        
    config = get_config()
    deps_mgr = DependencyManager()
    
    if remove:
        # Get list of installed xsim dependencies
        xsim_deps = []
        if (deps_mgr.deps_dir / 'oh-my-xilinx').exists():
            xsim_deps.append('oh-my-xilinx')
        if _is_finnxsim_built(deps_mgr):
            xsim_deps.append('finn-xsim')
            
        if not xsim_deps:
            warning("No Xilinx simulation dependencies are installed")
            return
            
        if not confirm_removal(xsim_deps, "Xilinx simulation dependencies", skip_confirm=yes):
            console.print("Removal cancelled")
            return
            
        # Remove dependencies
        with progress_spinner("Removing Xilinx simulation dependencies...") as task:
            results = deps_mgr.remove_group('xsim', quiet=True)
            if all(results.values()):
                success("Xilinx simulation dependencies removed successfully")
            else:
                failed = [k for k, v in results.items() if not v]
                error_exit(f"Failed to remove some dependencies: {', '.join(failed)}")
        return
    
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
    
    # Check if already built
    if not force and _is_finnxsim_built(deps_mgr):
        warning("finn-xsim already built (use --force to rebuild)")
        return
    
    # Install dependencies and build finn-xsim
    with progress_spinner("Setting up Xilinx simulation dependencies...") as task:
        try:
            # First install oh-my-xilinx
            oh_my_xilinx_result = deps_mgr.install('oh-my-xilinx', force=force, quiet=True)
            if not oh_my_xilinx_result:
                error_exit("Failed to install oh-my-xilinx")
            
            # Then build finn-xsim
            result = deps_mgr.install('finn-xsim', force=force, quiet=True)
            if not result:
                error_exit("Failed to build finn-xsim")
        except Exception as e:
            error_exit(
                f"Failed to setup Xilinx simulation: {e}",
                details=[
                    "Vivado is properly installed",
                    "You have the required Vivado license",
                    "The Vivado path in configuration is correct"
                ]
            )
    
    success("Xilinx simulation dependencies installed successfully")


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


def _handle_existing_boards(board_mgr: BoardManager, requested_repos: List[str], 
                           verbose: bool, force: bool) -> bool:
    """Handle the case where boards are already downloaded.
    
    Args:
        board_mgr: BoardManager instance
        requested_repos: List of specifically requested repos (empty for all)
        verbose: Whether to show detailed board list
        force: Whether to force redownload
        
    Returns:
        True if should continue with download, False otherwise
    """
    existing_repos = board_mgr.list_downloaded_repositories()
    
    # Check if requested repos are already present
    if requested_repos:
        already_have = [r for r in requested_repos if r in existing_repos]
        if not force and already_have == requested_repos:
            warning("Requested repositories already downloaded:")
            for r in sorted(already_have):
                console.print(f"      • {r}")
            
            if verbose:
                boards_by_repo = board_mgr.get_board_summary()
                filtered = {r: boards_by_repo[r] for r in already_have if r in boards_by_repo}
                _show_board_summary(filtered, "Board definitions")
            
            console.print("\n[dim]Use --force to redownload[/dim]")
            return False
    else:
        # Check if any boards exist
        if not force and existing_repos:
            boards_by_repo = board_mgr.get_board_summary()
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
@click.option('--remove', is_flag=True, help='Remove board definition files')
@click.option('--repo', '-r', multiple=True, help='Specific repository to download (e.g., xilinx, avnet). Downloads all if not specified.')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed list of all board files')
@click.option('--yes', '-y', is_flag=True, help='Skip confirmation prompts')
def boards(force: bool, remove: bool, repo: tuple, verbose: bool, yes: bool) -> None:
    """Download FPGA board definition files."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")
    config = get_config()
    deps_mgr = DependencyManager()
    board_files_dir = config.deps_dir / "board-files"
    board_mgr = BoardManager(board_files_dir)
    
    if remove:
        # Handle removal
        existing_repos = board_mgr.list_downloaded_repositories()
        
        if repo:
            # Validate and filter to installed repos
            valid_repos, invalid_repos = board_mgr.validate_repository_names(list(repo))
            if invalid_repos:
                error_exit(
                    f"Unknown board repositories: {', '.join(invalid_repos)}",
                    details=["Available repositories: avnet, xilinx, rfsoc4x2, kv260, aupzu3, pynq-z1, pynq-z2"]
                )
            repos_to_remove = [r for r in valid_repos if r in existing_repos]
            if not repos_to_remove:
                warning(f"None of the specified repositories are installed: {', '.join(valid_repos)}")
                return
        else:
            # Remove all
            repos_to_remove = existing_repos
            if not repos_to_remove:
                warning("No board repositories are installed")
                return
        
        if not confirm_removal(repos_to_remove, "board repositories", skip_confirm=yes):
            console.print("Removal cancelled")
            return
            
        # Remove repositories
        with progress_spinner("Removing board repositories...") as task:
            success_count = 0
            for repo_name in repos_to_remove:
                if deps_mgr.remove(repo_name, quiet=True):
                    success_count += 1
                else:
                    warning(f"Failed to remove {repo_name}")
                    
            if success_count == len(repos_to_remove):
                success(f"Removed {success_count} board repositories successfully")
            else:
                error_exit(f"Only {success_count} of {len(repos_to_remove)} repositories were removed")
        return
    
    # Validate repository names if specified
    if repo:
        valid_repos, invalid_repos = board_mgr.validate_repository_names(list(repo))
        if invalid_repos:
            error_exit(
                f"Unknown board repositories: {', '.join(invalid_repos)}",
                details=["Available repositories: avnet, xilinx, rfsoc4x2, kv260, aupzu3, pynq-z1, pynq-z2"]
            )
        repos_to_download = valid_repos
    else:
        repos_to_download = []
    
    # Check if boards already exist
    if not _handle_existing_boards(board_mgr, repos_to_download, verbose, force):
        return
    
    # Download boards  
    description = (f"Downloading {len(repos_to_download)} board repositories..." 
                  if repos_to_download
                  else "Downloading board definition files...")
    
    with progress_spinner(description) as task:
        try:
            if repos_to_download:
                # Download specific boards
                results = {board: deps_mgr.install(board, force=force, quiet=True) 
                          for board in repos_to_download}
                if not all(results.values()):
                    error_exit("Failed to download board files")
            else:
                # Download all board dependencies
                results = deps_mgr.install_group('boards', force=force, quiet=True)
                if not all(results.values()):
                    error_exit("Failed to download board files")
        except Exception as e:
            error_exit(f"Failed to download board files: {e}")
    
    # Show what was downloaded
    success("Board repositories downloaded:")
    
    # Get summary of downloaded boards
    downloaded_repos = board_mgr.list_downloaded_repositories()
    if repos_to_download:
        # Filter to show only requested repos that were downloaded
        downloaded_repos = [r for r in downloaded_repos if r in repos_to_download]
    
    # Display downloaded repositories
    boards_by_repo = {}
    total_boards = 0
    
    for repo_name in sorted(downloaded_repos):
        console.print(f"      • {repo_name}")
        repo_path = board_files_dir / repo_name
        board_files = board_mgr.find_board_files(repo_path)
        board_count = len(board_files)
        total_boards += board_count
        
        if verbose:
            boards = board_mgr.extract_board_names(board_files, repo_name)
            if boards:
                boards_by_repo[repo_name] = boards
    
    # Show summary
    if total_boards > 0:
        console.print(f"  [dim]{total_boards} total board definitions downloaded[/dim]")
    
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
    deps_mgr = DependencyManager()
    
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
    board_mgr = BoardManager(deps_mgr.deps_dir / "board-files")
    board_count = len(board_mgr.list_downloaded_repositories())
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
    cnpy_dir = deps_mgr.deps_dir / "cnpy"
    return (cnpy_dir / "cnpy.h").exists()


def _are_hlslib_headers_installed(deps_mgr) -> bool:
    """Check if finn-hlslib headers are installed."""
    hlslib_dir = deps_mgr.deps_dir / "finn-hlslib"
    return (hlslib_dir / "tb").exists()


def _is_finnxsim_built(deps_mgr) -> bool:
    """Check if finn-xsim module is built."""
    from finn import xsi
    return xsi.is_available()


