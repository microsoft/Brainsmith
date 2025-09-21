# Adapted from FINN-plus (https://github.com/eki-project/finn-plus)
# Copyright (c) 2020-2025, AMD/Xilinx and Paderborn University
# Licensed under BSD License - see FINN-plus repository for full license text

from __future__ import annotations

import click
import importlib
import os
import shlex
import subprocess
import sys
from pathlib import Path
from rich.console import Console

from brainsmith.interface import IS_POSIX
from brainsmith.interface.interface_utils import (
    assert_path_valid,
    write_yaml,
)
from brainsmith.interface.console import error, status, warning
from brainsmith.interface.manage_deps import install_pyxsi, update_dependencies
from brainsmith.interface.manage_tests import run_test
from brainsmith.config import load_config, export_to_environment, validate_and_report, get_config


# Resolves the path to modules which are not part of the FINN package hierarchy
def _resolve_module_path(name: str) -> str:
    # Try to import the module via importlib - allows "-" in names and resolve
    # the absolute path to the first candidate location as a string
    try:
        return str(importlib.import_module(name).__path__[0])
    except ModuleNotFoundError:
        # Try a different location if notebooks have not been found, maybe we
        # are in the Git repository root and should look there as well...
        try:
            return str(importlib.import_module(f"finn.{name}").__path__[0])
        except ModuleNotFoundError:
            warning(f"Could not resolve {name}. FINN might not work properly.")
    # Return the empty string as a default...
    return ""





def prepare_env(
    deps: Path | None,
    flow_config: Path,
    build_dir: Path | None,
    num_workers: int,
    is_test_run: bool = False,
    skip_dep_update: bool = False,
) -> None:
    """
    Prepares Brainsmith environment variables using Pydantic configuration.
    
    Priority order:
    1. Command line arguments
    2. Project settings file (brainsmith_settings.yaml)
    3. Environment variables
    4. Default values (env_defaults.yaml)
    """
    # Prepare CLI overrides
    cli_overrides = {}
    if build_dir:
        cli_overrides["bsmith_build_dir"] = str(build_dir.absolute())
    if deps:
        cli_overrides["bsmith_deps_dir"] = str(deps.absolute())
        cli_overrides["finn"] = {"finn_deps_dir": str(deps.absolute())}
    if num_workers and num_workers > 0:
        cli_overrides["finn"] = cli_overrides.get("finn", {})
        cli_overrides["finn"]["num_default_workers"] = num_workers
    
    # Load configuration using Pydantic
    config = load_config(cli_overrides=cli_overrides)
    
    # Validate configuration and report issues
    validate_and_report(config, raise_on_error=True)
    
    # Export to environment for backward compatibility
    export_to_environment(config, verbose=(config.debug.enabled))
    
    # Set FINN_DEPS for compatibility (not in standard export)
    os.environ["FINN_DEPS"] = str(config.bsmith_deps_dir)
    
    # Update dependencies if needed
    if not skip_dep_update:
        status(f"Using dependency path: {config.bsmith_deps_dir}")
        update_dependencies()
    else:
        warning("Skipping dependency updates!")
    
    # Install pyXSI
    pyxsi_status = install_pyxsi()
    if pyxsi_status:
        status("pyXSI installed successfully.")
    else:
        error("pyXSI installation failed.")
        sys.exit(1)
    
    # Create build directory if needed
    if not config.bsmith_build_dir.exists():
        config.bsmith_build_dir.mkdir(parents=True)
    status(f"Build directory set to: {config.bsmith_build_dir}")
    
    # Report worker count
    import multiprocessing
    workers = config.finn.num_default_workers or int(multiprocessing.cpu_count() * 0.75)
    status(f"Using {workers} workers.")
    
    # Resolve paths to some not properly packaged components...
    os.environ["FINN_RTLLIB"] = _resolve_module_path("finn-rtllib")
    os.environ["FINN_CUSTOM_HLS"] = _resolve_module_path("custom_hls")
    os.environ["FINN_QNN_DATA"] = _resolve_module_path("qnn-data")
    os.environ["FINN_NOTEBOOKS"] = _resolve_module_path("notebooks")
    os.environ["FINN_TESTS"] = _resolve_module_path("tests")




@click.group()
def main_group() -> None:
    pass


@click.command(help="Build a hardware design")
@click.option("--dependency-path", "-d", default="")
@click.option("--build-path", "-b", help="Specify a build temp path of your choice", default="")
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.option(
    "--skip-dep-update",
    "-s",
    is_flag=True,
    help="Whether to skip the dependency update. Can be changed in settings via"
    "AUTOMATIC_DEPENDENCY_UPDATES: false",
)
@click.option(
    "--start",
    default="",
    help="If no start_step is given in the dataflow build config, "
    "this starts the flow from the given step.",
)
@click.option(
    "--stop",
    default="",
    help="If no stop_step is given in the dataflow build config, "
    "this stops the flow at the given step.",
)
@click.argument("config")
@click.argument("model")
def build(
    dependency_path: str,
    build_path: str,
    num_workers: int,
    skip_dep_update: bool,
    start: str,
    stop: str,
    config: str,
    model: str,
) -> None:
    config_path = Path(config).expanduser()
    model_path = Path(model).expanduser()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    assert_path_valid(config_path)
    assert_path_valid(model_path)
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    status(f"Starting FINN build with config {config_path.name} and model {model_path.name}!")
    prepare_env(
        dep_path,
        config_path,
        build_dir,
        num_workers,
        skip_dep_update=skip_dep_update,
    )

    # Can import from finn now, since all deps are installed
    # and all environment variables are set correctly
    from finn.builder.build_dataflow import build_dataflow_cfg
    from finn.builder.build_dataflow_config import DataflowBuildConfig

    status("Creating dataflow build config...")
    dfbc: DataflowBuildConfig | None = None
    match config_path.suffix:
        case ".yaml" | ".yml":
            with config_path.open() as f:
                dfbc = DataflowBuildConfig.from_yaml(f.read())
        case ".json":
            with config_path.open() as f:
                dfbc = DataflowBuildConfig.from_json(f.read())
        case _:
            error(
                f"Unknown config file type: {config_path.name}. "
                "Valid formats are: .json, .yml, .yaml"
            )
            sys.exit(1)
    if dfbc is None:
        error("Failed to generate dataflow build config!")
        sys.exit(1)

    # Set start and stop steps
    if dfbc.start_step is None and start != "":
        dfbc.start_step = start
    if dfbc.stop_step is None and stop != "":
        dfbc.stop_step = stop

    # Set output directory to where the config lies, not where FINN lies
    if not Path(dfbc.output_dir).is_absolute():
        dfbc.output_dir = str((config_path.parent / dfbc.output_dir).absolute())
    status(f"Output directory is {dfbc.output_dir}")

    # Add path of config to sys.path so that custom steps can be found
    sys.path.append(str(config_path.parent.absolute()))

    Console().rule(
        f"[bold cyan]Running FINN with config[/bold cyan][bold orange1] "
        f"{config_path.name}[/bold orange1][bold cyan] on model [/bold cyan]"
        f"[bold orange1]{model_path.name}[/bold orange1]"
    )
    build_dataflow_cfg(str(model_path), dfbc)


@click.command(help="Run a script in a FINN environment")
@click.option("--dependency-path", "-d", default="")
@click.option("--build-path", "-b", help="Specify a build temp path of your choice", default="")
@click.option(
    "--skip-dep-update",
    "-s",
    is_flag=True,
    help="Whether to skip the dependency update. Can be changed in settings via"
    "AUTOMATIC_DEPENDENCY_UPDATES: false",
)
@click.option(
    "--num-workers",
    "-n",
    help="Number of parallel workers for FINN to use. When -1, automatically use 75% of cores",
    default=-1,
    show_default=True,
)
@click.argument("script")
def run(
    dependency_path: str, build_path: str, skip_dep_update: bool, num_workers: int, script: str
) -> None:
    script_path = Path(script).expanduser()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    assert_path_valid(script_path)
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    prepare_env(
        dep_path,
        script_path,
        build_dir,
        num_workers,
        skip_dep_update=skip_dep_update,
    )
    Console().rule(
        f"[bold cyan]Starting script "
        f"[/bold cyan][bold orange1]{script_path.name}[/bold orange1]"
    )
    subprocess.run(
        shlex.split(f"{sys.executable} {script_path.name}", posix=IS_POSIX), cwd=script_path.parent
    )


@click.command(help="Run a given benchmark configuration.")
@click.option("--bench_config", help="Name or path of experiment configuration file", required=True)
@click.option("--dependency-path", "-d", default="")
@click.option("--num-workers", "-n", default=-1, show_default=True)
@click.option(
    "--build-path",
    "-b",
    help="Specify a build temp path of your choice",
    default="",
)
def bench(bench_config: str, dependency_path: str, num_workers: int, build_path: str) -> None:
    console = Console()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    prepare_env(dep_path, Path(), build_dir, num_workers)
    console.rule("RUNNING BENCHMARK")

    # Late import because we need prepare_finn to setup remaining dependencies first
    from finn.benchmarking.bench import start_bench_run

    exit_code = start_bench_run(bench_config)
    sys.exit(exit_code)


@click.command(help="Run a given test. Uses /tmp/FINN_TMP as the temporary file location")
@click.option(
    "--variant",
    "-v",
    help="Which test to execute (quick, quicktest_ci, full_ci)",
    default="quick",
    show_default=True,
)
@click.option("--dependency-path", "-d", default="")
@click.option("--num-workers", "-n", default=-1, show_default=True)
@click.option("--num-test-workers", "-t", default="auto", show_default=True)
@click.option(
    "--build-path",
    "-b",
    help="Specify a build temp path of your choice",
    default="",
)
def test(
    variant: str, dependency_path: str, num_workers: int, num_test_workers: str, build_path: str
) -> None:
    console = Console()
    build_dir = Path(build_path).expanduser() if build_path != "" else None
    dep_path = Path(dependency_path).expanduser() if dependency_path != "" else None
    prepare_env(dep_path, Path(), build_dir, num_workers, is_test_run=True)
    status(f"Using {num_test_workers} test workers")
    console.rule("RUNNING TESTS")
    run_test(variant, num_test_workers)


@click.group(help="Dependency management")
def deps() -> None:
    pass


@click.command(help="Update or install dependencies to the given path")
@click.option(
    "--path",
    "-p",
    help="Path to install to",
    default="",
    show_default=True,
)
def update(path: str) -> None:
    dep_path = Path(path).expanduser() if path != "" else None
    prepare_env(dep_path, Path(), None, 1)


@click.group(help="Manage FINN settings")
def config() -> None:
    # TODO: Config remove?
    pass




@click.command("list", help="List the current configuration")
def config_list() -> None:
    from brainsmith.config import get_config
    config = get_config()
    console = Console()
    # TODO: Pretty print the full config
    console.print("Current Brainsmith configuration:")
    console.print(f"bsmith_dir: {config.bsmith_dir}")
    console.print(f"bsmith_build_dir: {config.bsmith_build_dir}")
    console.print(f"bsmith_deps_dir: {config.bsmith_deps_dir}")


@click.command("get", help="Get a specific configuration value")
@click.argument("key")
def config_get(key: str) -> None:
    from brainsmith.config import get_config
    config = get_config()
    # TODO: Implement dot notation navigation
    console = Console()
    try:
        value = getattr(config, key)
        console.print(f"[blue]{key}[/blue]: {value}")
    except AttributeError:
        error(f"Configuration key '{key}' not found")
        sys.exit(1)




@click.command(
    "init",
    help="Create a template brainsmith_settings.yaml in the current directory",
)
def config_init() -> None:
    from brainsmith.config.migrate import generate_example_settings
    p = Path("brainsmith_settings.yaml")
    if p.exists():
        error("brainsmith_settings.yaml already exists")
        sys.exit(1)
    example = generate_example_settings()
    if not write_yaml(example, p):
        error(f"Writing to {p} failed!")
        sys.exit(1)
    status(f"Created {p}")


def main() -> None:
    config.add_command(config_list)
    config.add_command(config_init)
    config.add_command(config_get)
    deps.add_command(update)
    main_group.add_command(config)
    main_group.add_command(deps)
    main_group.add_command(build)
    main_group.add_command(bench)
    main_group.add_command(test)
    main_group.add_command(run)
    main_group()


if __name__ == "__main__":
    main()
