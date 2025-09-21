# Adapted from FINN-plus (https://github.com/eki-project/finn-plus)
# Copyright (c) 2020-2025, AMD/Xilinx and Paderborn University
# Licensed under BSD License - see FINN-plus repository for full license text

from __future__ import annotations

import sys
import yaml
from pathlib import Path
from rich.console import Console


def assert_path_valid(p: Path) -> None:
    """Check if the path exists, if not print an error message and exit with an error code"""
    if not p.exists():
        Console().print(f"[bold red]File or directory {p} does not exist. Stopping...[/bold red]")
        sys.exit(1)




def read_yaml(p: Path) -> dict | None:
    """Read a yaml file and return its contents. If the file does not exist, return None"""
    if p.exists():
        with p.open() as f:
            return yaml.load(f, yaml.Loader)
    else:
        return None


def write_yaml(data: dict, p: Path) -> bool:
    """Try writing the given data to a yaml file. If this fails, return false otherwise
    true"""
    try:
        with p.open("w+") as f:
            yaml.dump(data, f, yaml.Dumper)
            return True
    except (OSError, yaml.error.YAMLError):
        return False
