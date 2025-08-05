"""
Kernel Integrator (KI)

The KI is a clean, modular system for generating FINN-compatible AutoHWCustomOp 
implementations from SystemVerilog RTL with extensible generator architecture.

Key Features:
- KernelIntegrator: Orchestrates generation workflow using modular generators
- GeneratorManager: Auto-discovery and management of generators
- Extensible generator system with custom context processing
- Template system with runtime parameter extraction
- Enhanced data structures with performance tracking
- Clean CLI interface with debug support
- Complete RTL-to-AutoHWCustomOp pipeline
"""

import warnings

from .types.generation import GenerationResult, PerformanceMetrics, GenerationValidationResult
from .types.config import Config
from .cli import main
from .kernel_integrator import KernelIntegrator, KernelIntegratorError
from .generators import GeneratorManager, GeneratorBase

__version__ = "4.0.0"
__all__ = [
    "KernelIntegrator",
    "KernelIntegratorError",
    "GeneratorManager",
    "GeneratorBase",
    "GenerationResult",
    "GenerationValidationResult", 
    "PerformanceMetrics",
    "Config",
    "main",
]