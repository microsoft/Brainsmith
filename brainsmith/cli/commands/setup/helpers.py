# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""Shared helper functions for setup commands."""

from pathlib import Path

from brainsmith._internal.io.dependencies import DependencyManager


def _is_cnpy_installed(deps_mgr: DependencyManager) -> bool:
    """Check if cnpy C++ library is installed.

    Args:
        deps_mgr: DependencyManager instance

    Returns:
        True if cnpy header file exists
    """
    cnpy_dir = deps_mgr.deps_dir / "cnpy"
    return (cnpy_dir / "cnpy.h").exists()


def _are_hlslib_headers_installed(deps_mgr: DependencyManager) -> bool:
    """Check if FINN HLS library headers are installed.

    Args:
        deps_mgr: DependencyManager instance

    Returns:
        True if finn-hlslib testbench directory exists
    """
    hlslib_dir = deps_mgr.deps_dir / "finn-hlslib"
    return (hlslib_dir / "tb").exists()


def _is_finnxsim_built() -> bool:
    """Check if FINN XSI (Xilinx simulation) module is available.

    Returns:
        True if finn-xsim is built and importable
    """
    from finn import xsi
    return xsi.is_available()
