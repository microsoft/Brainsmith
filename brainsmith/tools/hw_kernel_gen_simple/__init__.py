"""
Hardware Kernel Generator - Simplified Implementation

A clean, minimal implementation of the HWKG that eliminates enterprise bloat
while preserving all core functionality.

This simplified version reduces the codebase from ~6000 lines to ~750 lines
by removing unnecessary orchestration layers, factory patterns, and 
enterprise abstractions.

Usage:
    python -m brainsmith.tools.hw_kernel_gen_simple rtl_file.sv compiler_data.py -o output/
"""

__version__ = "1.0.0"
__author__ = "Brainsmith Team"

from .cli import main
from .config import Config
from .data import HWKernel, GenerationResult

__all__ = ['main', 'Config', 'HWKernel', 'GenerationResult']