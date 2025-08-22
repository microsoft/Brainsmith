"""RTL Parser for Kernel Integrator.

This package provides functionality to parse SystemVerilog RTL files and extract
information needed by the Kernel Integrator to create FINN-compatible
hardware kernels.
"""

from .parser import RTLParser

__all__ = [
    "RTLParser",
]