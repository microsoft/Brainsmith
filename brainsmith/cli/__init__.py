# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Brainsmith command-line interface.

Provides two CLI entry points with shared implementation:

1. brainsmith - Administrative toolkit
   Commands: config, setup, smith (subcommand)
   Usage: brainsmith config show

2. smith - Operational toolkit
   Commands: dfc, kernel
   Usage: smith dfc model.onnx blueprint.yaml

Architecture:
- Shared implementation via create_cli() factory in cli.py
- Configuration managed through ApplicationContext (context.py)
- Commands auto-receive context via @click.pass_obj decorator

Entry points defined in pyproject.toml.
"""

from .cli import brainsmith_main, smith_main

__all__ = [
    "brainsmith_main",
    "smith_main",
]
