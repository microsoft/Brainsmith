# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Board file management command."""

from typing import TYPE_CHECKING

import click

# Import settings and dependencies lazily inside function to keep --help fast
from ...utils import (
    console, error_exit, success, warning,
    progress_spinner, confirm_or_abort
)

if TYPE_CHECKING:
    from brainsmith._internal.io.dependencies import BoardManager


def _show_board_summary(boards_by_repo: dict[str, list[str]], title: str) -> None:
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


def _handle_existing_boards(board_mgr: "BoardManager", requested_repos: list[str],
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


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--force", "-f", is_flag=True, help="Force redownload even if already present")
@click.option("--remove", is_flag=True, help="Remove board definition files")
@click.option("--repo", "-r", multiple=True, help="Specific repository to download (e.g., xilinx, avnet). Downloads all if not specified.")
@click.option("--verbose", "-v", is_flag=True, help="Show detailed list of all board definitions by repository")
@click.option("--yes", "-y", is_flag=True, help="Skip confirmation prompts")
@click.pass_obj
def boards(ctx, force: bool, remove: bool, repo: tuple, verbose: bool, yes: bool) -> None:
    """Download FPGA board definition files."""
    if force and remove:
        error_exit("Cannot use --force and --remove together")

    # Import settings and dependencies only when command executes (not for --help)
    from brainsmith.settings import get_config
    from brainsmith._internal.io.dependencies import DependencyManager, BoardManager

    config = get_config()
    deps_mgr = DependencyManager()
    board_files_dir = config.deps_dir / "board-files"
    board_mgr = BoardManager(board_files_dir)

    if remove:
        existing_repos = board_mgr.list_downloaded_repositories()

        if repo:
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
            repos_to_remove = existing_repos
            if not repos_to_remove:
                warning("No board repositories are installed")
                return

        warning("The following board repositories will be removed:")
        for repo in sorted(repos_to_remove):
            console.print(f"      • {repo}")

        confirm_or_abort("\nAre you sure you want to remove these repositories?", skip=yes)

        with progress_spinner("Removing board repositories...", no_progress=ctx.no_progress) as task:
            # remove() raises exception on failure, returns None on success
            failed = []
            for repo_name in repos_to_remove:
                try:
                    deps_mgr.remove(repo_name)
                except Exception as e:
                    warning(f"Failed to remove {repo_name}: {e}")
                    failed.append(repo_name)

            if not failed:
                success(f"Removed {len(repos_to_remove)} board repositories successfully")
            else:
                error_exit(f"Failed to remove {len(failed)} repositories: {', '.join(failed)}")
        return

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

    if not _handle_existing_boards(board_mgr, repos_to_download, verbose, force):
        return

    description = (f"Downloading {len(repos_to_download)} board repositories..."
                  if repos_to_download
                  else "Downloading board definition files...")

    with progress_spinner(description, no_progress=ctx.no_progress) as task:
        try:
            if repos_to_download:
                # Download specific boards (install() raises exception on failure)
                for board in repos_to_download:
                    deps_mgr.install(board, force=force)
            else:
                # Download all board dependencies
                # install_group returns Dict[str, Optional[Exception]] where None = success
                results = deps_mgr.install_group("boards", force=force)
                failed = [k for k, v in results.items() if v is not None]
                if failed:
                    error_exit(f"Failed to download boards: {', '.join(failed)}")
        except Exception as e:
            error_exit(f"Failed to download board files: {e}")

    success("Board repositories downloaded:")

    downloaded_repos = board_mgr.list_downloaded_repositories()
    if repos_to_download:
        # Filter to show only requested repos that were downloaded
        downloaded_repos = [r for r in downloaded_repos if r in repos_to_download]

    boards_by_repo = {}
    total_boards = 0

    for repo_name in sorted(downloaded_repos):
        console.print(f"      • {repo_name}")
        repo_path = board_files_dir / repo_name
        board_files = board_mgr.find_board_files(repo_path)
        board_count = len(board_files)
        total_boards += board_count

        if verbose:
            boards = board_mgr.extract_board_names(board_files)
            if boards:
                boards_by_repo[repo_name] = boards

    if total_boards > 0:
        console.print(f"  [dim]{total_boards} total board definitions downloaded[/dim]")

    if verbose:
        _show_board_summary(boards_by_repo, "Board definitions by repository")
