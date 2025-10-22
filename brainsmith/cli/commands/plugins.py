# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Plugin discovery and management commands."""

import click
from rich.table import Table
from collections import defaultdict

from brainsmith.loader import (
    list_steps, list_kernels, list_all_backends,
    _get_kernel_metadata, get_backend_metadata
)
from brainsmith.registry import registry
from ..context import ApplicationContext
from ..utils import console


@click.command(context_settings={'help_option_names': ['-h', '--help']})
@click.option('--verbose', '-v', is_flag=True, help='Show detailed component information')
@click.pass_obj
def plugins(app_ctx: ApplicationContext, verbose: bool) -> None:
    """Shows all available steps, kernels, and backends, organized by source:

    - brainsmith: Core Brainsmith components
    - finn: FINN framework components
    - qonnx: QONNX framework components
    - user: Custom plugins from configured sources
    - <pkg>: Entry point plugins from installed packages
    """
    config = app_ctx.get_effective_config()

    # Trigger discovery
    all_steps = list_steps()
    all_kernels = list_kernels()
    all_backends = list_all_backends()

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
                    try:
                        meta = _get_kernel_metadata(full_name)
                        has_infer = '✓' if meta.get('infer') else '✗'
                        console.print(f"    • {name:30} (infer={has_infer})")
                    except Exception as e:
                        console.print(f"    • {name:30} [red](error: {e})[/red]")

        # Backends by source
        if all_backends:
            console.print(f"\n[bold cyan]BACKENDS[/bold cyan]")
            for source in sorted(backends_by_source.keys()):
                console.print(f"\n  [green]{source}:[/green]")
                for name in sorted(backends_by_source[source]):
                    full_name = f"{source}:{name}"
                    try:
                        meta = get_backend_metadata(full_name)
                        target = meta.get('target_kernel', 'N/A')
                        lang = meta.get('language', 'N/A')
                        console.print(f"    • {name:25} → {target:25} ({lang})")
                    except Exception as e:
                        console.print(f"    • {name:25} [red](error: {e})[/red]")

    # Footer (only show verbose hint if not already verbose)
    if not verbose:
        console.print(f"\n[dim]Use --verbose to see detailed component listings[/dim]")
