"""
Kernel Integrator

Simple system for generating FINN-compatible AutoHWCustomOp implementations 
from SystemVerilog RTL.
"""

from .cli import main
from .generator import KernelGenerator

__version__ = "4.0.0"
__all__ = [
    "KernelGenerator",
    "main",
]