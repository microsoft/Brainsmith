# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helper functions for setup commands."""

from pathlib import Path

from brainsmith._internal.io.dependencies import DependencyManager


def _is_cnpy_installed(deps_mgr: DependencyManager) -> bool:
    cnpy_dir = deps_mgr.deps_dir / "cnpy"
    return (cnpy_dir / "cnpy.h").exists()


def _are_hlslib_headers_installed(deps_mgr: DependencyManager) -> bool:
    hlslib_dir = deps_mgr.deps_dir / "finn-hlslib"
    return (hlslib_dir / "tb").exists()


def _is_finnxsim_built() -> bool:
    from finn import xsi
    return xsi.is_available()
