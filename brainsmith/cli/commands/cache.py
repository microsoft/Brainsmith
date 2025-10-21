"""Cache management commands for manifest-based plugin discovery."""

import logging

import click
from rich.table import Table

from brainsmith.manifest_generator import (
    generate_all_manifests,
    clear_manifests,
    get_cache_dir,
    load_manifest,
    is_manifest_valid,
)
from ..utils import console, confirm_or_abort

logger = logging.getLogger(__name__)


@click.group(help="Manage plugin manifest cache")
def cache():
    """Manage plugin manifest cache."""
    pass


@cache.command("status")
def cache_status():
    """Show manifest cache status."""
    console.print("\n[bold]Manifest Cache Status[/bold]\n")

    cache_dir = get_cache_dir()
    console.print(f"Cache directory: [cyan]{cache_dir}[/cyan]")

    # List manifest files
    manifests = list(cache_dir.glob("*.yaml"))

    if not manifests:
        console.print("\n[yellow]No manifests cached[/yellow]")
        console.print("Run [cyan]brainsmith cache generate[/cyan] to create manifests")
        return

    # Create table
    table = Table(title="Cached Manifests", show_header=True)
    table.add_column("Package", style="cyan")
    table.add_column("Version", style="green")
    table.add_column("Kernels", justify="right")
    table.add_column("Backends", justify="right")
    table.add_column("Steps", justify="right")
    table.add_column("Status", style="bold")

    for manifest_file in sorted(manifests):
        package_name = manifest_file.stem
        manifest = load_manifest(package_name)

        if manifest:
            valid = is_manifest_valid(package_name)
            status = "[green]✓ Valid[/green]" if valid else "[yellow]⚠ Outdated[/yellow]"

            table.add_row(
                package_name,
                manifest.get('version', 'unknown'),
                str(len(manifest.get('kernels', []))),
                str(len(manifest.get('backends', []))),
                str(len(manifest.get('steps', []))),
                status
            )
        else:
            table.add_row(
                package_name,
                "?",
                "?",
                "?",
                "?",
                "[red]✗ Invalid[/red]"
            )

    console.print(table)

    # Show usage hint
    console.print("\n[dim]Use [cyan]brainsmith cache generate[/cyan] to update outdated manifests[/dim]")
    console.print("[dim]Use [cyan]brainsmith cache clear[/cyan] to remove all cached manifests[/dim]")


@cache.command("generate")
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Regenerate even if valid manifests exist"
)
def cache_generate(force: bool):
    """Generate manifest for FINN package.

    Scans FINN package and generates manifest file containing plugin metadata.
    Manifests enable fast plugin discovery without importing heavy dependencies.

    By default, only regenerates missing manifests. Use --force to regenerate.

    Note: Manifests are not auto-invalidated on package upgrades. Set
    eager_plugin_discovery=true in config for automatic regeneration, or
    manually regenerate after upgrades.
    """
    console.print("\n[bold]Generating Plugin Manifests[/bold]\n")

    if force:
        console.print("[yellow]Force mode: Regenerating all manifests[/yellow]\n")

    # Generate manifests
    with console.status("[bold green]Scanning packages..."):
        results = generate_all_manifests(force=force)

    # Show results
    table = Table(show_header=True)
    table.add_column("Package", style="cyan")
    table.add_column("Status", style="bold")

    for package_name, success in results.items():
        if success:
            manifest = load_manifest(package_name)
            if manifest:
                status = (
                    f"[green]✓ Generated[/green] "
                    f"({len(manifest['kernels'])} kernels, "
                    f"{len(manifest['backends'])} backends)"
                )
            else:
                status = "[yellow]⚠ Generated but could not load[/yellow]"
        else:
            status = "[red]✗ Failed (package not installed?)[/red]"

        table.add_row(package_name, status)

    console.print(table)

    # Show success count
    success_count = sum(1 for s in results.values() if s)
    total_count = len(results)

    if success_count == total_count:
        console.print(f"\n[green]✓ Successfully generated {success_count}/{total_count} manifests[/green]")
    else:
        console.print(f"\n[yellow]⚠ Generated {success_count}/{total_count} manifests[/yellow]")

    console.print("\n[dim]Manifests will be used automatically for faster plugin discovery[/dim]")


@cache.command("clear")
@click.option(
    "--yes",
    "-y",
    "skip_confirm",
    is_flag=True,
    help="Skip confirmation prompt"
)
def cache_clear(skip_confirm: bool):
    """Clear all cached manifests.

    This removes all generated manifest files from the cache directory.
    Manifests will be regenerated on next use or when explicitly generated
    via 'brainsmith cache generate'.
    """
    cache_dir = get_cache_dir()
    manifests = list(cache_dir.glob("*.yaml"))

    if not manifests:
        console.print("[yellow]No manifests to clear[/yellow]")
        return

    # Show what will be deleted
    console.print(f"\n[bold]Clearing Manifest Cache[/bold]\n")
    console.print(f"Cache directory: [cyan]{cache_dir}[/cyan]")
    console.print(f"Manifests to delete: [yellow]{len(manifests)}[/yellow]")

    for manifest_file in manifests:
        console.print(f"  • {manifest_file.name}")

    # Confirm
    confirm_or_abort("\nAre you sure you want to delete these manifests?", skip=skip_confirm)

    # Clear
    count = clear_manifests()
    console.print(f"\n[green]✓ Deleted {count} manifest files[/green]")
    console.print("\n[dim]Run [cyan]brainsmith cache generate[/cyan] to recreate manifests[/dim]")


@cache.command("info")
def cache_info():
    """Show information about manifest-based plugin discovery.

    Explains how manifest-based discovery works and its performance benefits.
    """
    console.print("\n[bold]Manifest-Based Plugin Discovery[/bold]\n")

    console.print(
        "Brainsmith uses [cyan]manifest files[/cyan] to speed up plugin discovery. "
        "Instead of importing all plugin classes (which can take 6+ seconds), "
        "manifests contain pre-scanned metadata that loads in milliseconds.\n"
    )

    console.print("[bold]How it works:[/bold]")
    console.print("  1. [cyan]Generate[/cyan]: Scan installed packages (Brainsmith, FINN, QONNX) for plugins")
    console.print("  2. [cyan]Cache[/cyan]: Save plugin metadata to {project_dir}/plugins/")
    console.print("  3. [cyan]Load[/cyan]: Read manifests instead of importing classes")
    console.print("  4. [cyan]Lazy Import[/cyan]: Import classes only when actually used\n")

    console.print("[bold]Performance:[/bold]")
    console.print("  • Without manifests: ~6.4s discovery time")
    console.print("  • With manifests: ~50ms discovery time")
    console.print("  • [green]~128x faster![/green]\n")

    console.print("[bold]Development workflow:[/bold]")
    console.print("  Enable automatic manifest regeneration during development:")
    console.print("    [cyan]# brainsmith_config.yaml")
    console.print("    eager_plugin_discovery: true[/cyan]")
    console.print("  Or manually regenerate after code changes:")
    console.print("    [cyan]brainsmith cache generate --force[/cyan]\n")

    console.print("[bold]Commands:[/bold]")
    console.print("  [cyan]brainsmith cache status[/cyan]    - Show cached manifests")
    console.print("  [cyan]brainsmith cache generate[/cyan]  - Generate/update manifests")
    console.print("  [cyan]brainsmith cache clear[/cyan]     - Delete all manifests")
    console.print("  [cyan]brainsmith cache info[/cyan]      - Show this information\n")
