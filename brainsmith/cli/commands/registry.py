# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Component discovery and management commands."""

from collections import defaultdict

import click
from rich.table import Table

# Import registry functions lazily inside function to keep --help fast
from ..context import ApplicationContext
from ..utils import console, progress_spinner


def _validate_components(names: list, getter) -> dict:
    """Validate components by attempting to load them.

    Args:
        names: Component names to validate
        getter: Function to load component (get_kernel, get_step, etc.)

    Returns:
        Dict mapping failed component names to error messages
    """
    errors = {}
    for name in names:
        try:
            getter(name)
        except Exception as e:
            errors[name] = str(e)
    return errors


@click.command(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--verbose", "-v", is_flag=True, help="Show detailed component information")
@click.option("--rebuild", "-r", is_flag=True, help="Rebuild cache and validate all components (slower)")
@click.pass_obj
def registry(ctx: ApplicationContext, verbose: bool, rebuild: bool) -> None:
    """Shows all registered components (steps, kernels, backends) organized by source:

    - brainsmith: Core Brainsmith components
    - finn: FINN framework components
    - qonnx: QONNX framework components
    - user: Custom components from configured sources
    - <pkg>: Entry point components from installed packages

    By default, shows fast metadata listing from cache. Use --rebuild to
    force cache regeneration and validate all components by importing them.
    """
    config = ctx.get_effective_config()

    # Import registry functions only when command executes (not for --help)
    from brainsmith.registry import (
        discover_components,
        get_all_component_metadata,
        list_backends,
        list_kernels,
        list_steps,
    )

    # Trigger discovery (with rebuild if requested)
    discover_components(use_cache=not rebuild, force_refresh=rebuild)

    all_steps = list_steps()
    all_kernels = list_kernels()
    all_backends = list_backends()

    # Group by source
    def group_by_source(components):
        by_source = defaultdict(list)
        for comp in components:
            source, name = comp.split(":", 1)
            by_source[source].append(name)
        return by_source

    steps_by_source = group_by_source(all_steps)
    kernels_by_source = group_by_source(all_kernels)
    backends_by_source = group_by_source(all_backends)

    # Get all sources
    all_sources = sorted(set(
        list(steps_by_source.keys()) +
        list(kernels_by_source.keys()) +
        list(backends_by_source.keys())
    ))

    # Show all component sources (configured + discovered)
    sources_table = Table(title="Component Sources")
    sources_table.add_column("Source", style="cyan")
    sources_table.add_column("Type", style="white")
    sources_table.add_column("Path", style="white")
    sources_table.add_column("Status", justify="center")

    # Add core namespace (brainsmith)
    brainsmith_path = config.bsmith_dir / 'brainsmith'
    sources_table.add_row(
        "brainsmith",
        "core",
        str(brainsmith_path),
        "[green]✓[/green]" if brainsmith_path.exists() else "[red]✗[/red]"
    )

    # Add discovered entry points
    try:
        from importlib.metadata import entry_points
        eps = entry_points(group='brainsmith.plugins')
        for ep in eps:
            # Entry point path is typically in deps_dir but could be anywhere
            # Show "auto-discovered" instead of guessing path
            ep_path = config.deps_dir / ep.name
            if ep_path.exists():
                status = "[green]✓[/green]"
                path_display = str(ep_path)
            else:
                status = "[dim]auto[/dim]"
                path_display = "(auto-discovered)"
            sources_table.add_row(ep.name, "entry point", path_display, status)
    except Exception as e:
        console.print(f"[yellow]Warning: Could not scan entry points: {e}[/yellow]")

    # Add configured filesystem sources
    for source_name, path in sorted(config.component_sources.items()):
        exists = "✓" if path.exists() else "✗"
        exists_colored = f"[green]{exists}[/green]" if exists == "✓" else f"[red]{exists}[/red]"
        sources_table.add_row(source_name, "filesystem", str(path), exists_colored)

    console.print(sources_table)
    console.print()

    # Summary table by source
    summary_table = Table(title="Component Summary by Source")
    summary_table.add_column("Source", style="cyan")
    summary_table.add_column("Steps", justify="right")
    summary_table.add_column("Kernels", justify="right")
    summary_table.add_column("Backends", justify="right")

    total_steps = 0
    total_kernels = 0
    total_backends = 0

    for source in all_sources:
        steps_count = len(steps_by_source.get(source, []))
        kernels_count = len(kernels_by_source.get(source, []))
        backends_count = len(backends_by_source.get(source, []))

        total_steps += steps_count
        total_kernels += kernels_count
        total_backends += backends_count

        summary_table.add_row(
            source,
            str(steps_count),
            str(kernels_count),
            str(backends_count)
        )

    summary_table.add_row(
        "[bold]Total",
        f"[bold]{total_steps}",
        f"[bold]{total_kernels}",
        f"[bold]{total_backends}"
    )

    console.print(summary_table)

    # Validate components if requested (after showing tables)
    validation_errors = {}
    if rebuild:
        from brainsmith.registry import get_backend, get_kernel, get_step

        with progress_spinner("Validating components...", transient=False, no_progress=ctx.no_progress) as task:
            validation_errors.update(_validate_components(all_kernels, get_kernel))
            validation_errors.update(_validate_components(all_backends, get_backend))
            validation_errors.update(_validate_components(all_steps, get_step))

        console.print()  # Blank line after spinner
        if validation_errors:
            console.print(f"[bold red]Found {len(validation_errors)} error(s):[/bold red]\n")
            for component, error in sorted(validation_errors.items()):
                console.print(f"  [red]✗[/red] {component}")
                console.print(f"    [dim]{error}[/dim]")
            console.print()
        else:
            console.print("[bold green]✅ All components validated successfully[/bold green]\n")

    # Show detailed listings if verbose
    if verbose:
        console.print()

        # Steps by source
        if all_steps:
            console.print("[bold cyan]STEPS[/bold cyan]")
            for source in sorted(steps_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(steps_by_source[source]):
                    console.print(f"    • {name}")

        # Kernels by source
        if all_kernels:
            console.print("\n[bold cyan]KERNELS[/bold cyan]")
            all_metadata = get_all_component_metadata()
            for source in sorted(kernels_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(kernels_by_source[source]):
                    full_name = f"{source}:{name}"

                    # Check validation status
                    validation_marker = ""
                    if rebuild:
                        if full_name in validation_errors:
                            validation_marker = " [red]✗ FAILED[/red]"
                        else:
                            validation_marker = " [green]✓[/green]"

                    try:
                        meta = all_metadata.get(full_name)
                        has_infer = "[green]✓[/green]" if meta and meta.kernel_infer else "[red]✗[/red]"
                        console.print(f"    • {name:30} (infer={has_infer}){validation_marker}")
                    except Exception as e:
                        console.print(f"    • {name:30} [red](error: {e})[/red]")

        # Backends by source
        if all_backends:
            console.print("\n[bold cyan]BACKENDS[/bold cyan]")
            all_metadata = get_all_component_metadata()
            for source in sorted(backends_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(backends_by_source[source]):
                    full_name = f"{source}:{name}"

                    # Check validation status
                    validation_marker = ""
                    if rebuild:
                        if full_name in validation_errors:
                            validation_marker = " [red]✗ FAILED[/red]"
                        else:
                            validation_marker = " [green]✓[/green]"

                    try:
                        meta = all_metadata.get(full_name)
                        target = meta.backend_target if meta else "N/A"
                        lang = meta.backend_language if meta else "N/A"
                        console.print(f"    • {name:25} → {target:25} ({lang}){validation_marker}")
                    except Exception as e:
                        console.print(f"    • {name:25} [red](error: {e})[/red]")

    # Footer hints
    hints = []
    if not verbose:
        hints.append("--verbose to see detailed component listings")
    if not rebuild:
        hints.append("--rebuild to regenerate cache and validate all components")

    if hints:
        console.print(f"\n[dim]Use {' or '.join(hints)}[/dim]")
