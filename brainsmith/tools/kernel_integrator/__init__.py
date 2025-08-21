"""
Kernel Integrator

Simple system for generating FINN-compatible AutoHWCustomOp implementations 
from SystemVerilog RTL.
"""

from .cli import main
from .generator import KernelGenerator
from .rtl_parser.parser import RTLParser
from .metadata import KernelMetadata

__version__ = "4.0.0"
__all__ = [
    "KernelGenerator",
    "RTLParser",
    "KernelMetadata",
    "main",
]