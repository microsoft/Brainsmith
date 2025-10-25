# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plugin discovery and management commands."""

import click
from rich.table import Table
from collections import defaultdict

from brainsmith.loader import (
    list_steps, list_kernels, list_backends,
    get_backend_metadata, _component_index
)
from brainsmith.registry import registry
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


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbose', '-v', is_flag=True, help='Show detailed component information')
@click.option('--validate', is_flag=True, help='Validate all components by importing them (slower)')
@click.option('--refresh', is_flag=True, help='Rebuild component manifest cache (ignore existing cache)')
@click.pass_obj
def plugins(app_ctx: ApplicationContext, verbose: bool, validate: bool, refresh: bool) -> None:
    """Shows all available steps, kernels, and backends, organized by source:

    - brainsmith: Core Brainsmith components
    - finn: FINN framework components
    - qonnx: QONNX framework components
    - user: Custom plugins from configured sources
    - <pkg>: Entry point plugins from installed packages

    By default, shows fast metadata listing from cache. Use --refresh to
    rebuild the cache, --validate to import all components and check for errors.
    """
    config = app_ctx.get_effective_config()

    # Trigger discovery (with refresh if requested)
    from brainsmith.loader import discover_plugins
    discover_plugins(use_cache=not refresh, force_refresh=refresh)

    all_steps = list_steps()
    all_kernels = list_kernels()
    all_backends = list_backends()

    # Group by source
    def group_by_source(components):
        by_source = defaultdict(list)
        for comp in components:
            source, name = comp.split(':', 1)
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

    # Show plugin sources from config
    plugin_sources = config.plugin_sources
    if plugin_sources:
        sources_table = Table(title="Configured Plugin Sources")
        sources_table.add_column("Source", style="cyan")
        sources_table.add_column("Path", style="white")
        sources_table.add_column("Exists", justify="center")

        for source_name, path in sorted(plugin_sources.items()):
            exists = "✓" if path.exists() else "✗"
            exists_colored = f"[green]{exists}[/green]" if exists == "✓" else f"[red]{exists}[/red]"
            sources_table.add_row(source_name, str(path), exists_colored)

        console.print(sources_table)
        console.print()

    # Summary table by source
    summary_table = Table(title="Plugin Summary by Source")
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
    if validate:
        from brainsmith.loader import get_kernel, get_backend, get_step

        with progress_spinner("Validating components...", transient=False, no_progress=app_ctx.no_progress) as task:
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
            console.print("[bold green]✓ All components validated successfully[/bold green]\n")

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
            console.print(f"\n[bold cyan]KERNELS[/bold cyan]")
            for source in sorted(kernels_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(kernels_by_source[source]):
                    full_name = f"{source}:{name}"

                    # Check validation status
                    validation_marker = ""
                    if validate:
                        if full_name in validation_errors:
                            validation_marker = " [red]✗ FAILED[/red]"
                        else:
                            validation_marker = " [green]✓[/green]"

                    try:
                        meta = _component_index.get(full_name)
                        has_infer = '✓' if meta and meta.import_spec and meta.import_spec.extra.get('infer_transform') else '✗'
                        console.print(f"    • {name:30} (infer={has_infer}){validation_marker}")
                    except Exception as e:
                        console.print(f"    • {name:30} [red](error: {e})[/red]")

        # Backends by source
        if all_backends:
            console.print(f"\n[bold cyan]BACKENDS[/bold cyan]")
            for source in sorted(backends_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(backends_by_source[source]):
                    full_name = f"{source}:{name}"

                    # Check validation status
                    validation_marker = ""
                    if validate:
                        if full_name in validation_errors:
                            validation_marker = " [red]✗ FAILED[/red]"
                        else:
                            validation_marker = " [green]✓[/green]"

                    try:
                        meta = get_backend_metadata(full_name)
                        target = meta.get('target_kernel', 'N/A')
                        lang = meta.get('language', 'N/A')
                        console.print(f"    • {name:25} → {target:25} ({lang}){validation_marker}")
                    except Exception as e:
                        console.print(f"    • {name:25} [red](error: {e})[/red]")

    # Footer hints
    hints = []
    if not verbose:
        hints.append("--verbose to see detailed component listings")
    if not validate:
        hints.append("--validate to test import all components")

    if hints:
        console.print(f"\n[dim]Use {' or '.join(hints)}[/dim]")
