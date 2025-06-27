"""
Hardware Kernel Generator (HWKG) - Phase 4 Modular System

Phase 4 provides a clean, modular system for generating FINN-compatible AutoHWCustomOp 
implementations from SystemVerilog RTL with extensible generator architecture.

Key Features:
- KernelIntegrator: Orchestrates generation workflow using modular generators
- GeneratorManager: Auto-discovery and management of generators
- Extensible generator system with custom context processing
- Phase 2 template system with runtime parameter extraction
- Enhanced data structures with performance tracking
- Clean CLI interface with debug support
- Complete RTL-to-AutoHWCustomOp pipeline

Phase 4 Architecture:
- RTL Parser: SystemVerilog parsing with BDIM pragma support
- KernelIntegrator: High-level orchestration of generation workflow
- GeneratorManager: Discovery and management of modular generators
- Generators: Modular, extensible artifact generation (hw_custom_op, rtl_wrapper, rtl_backend)
- Enhanced data structures: Unified GenerationResult with rich tracking
"""

import warnings

# Core Phase 4 components
from .data import GenerationResult, GenerationValidationResult, PerformanceMetrics
from .config import Config
from .cli import main
from .kernel_integrator import KernelIntegrator, KernelIntegratorError
from .generators import GeneratorManager, GeneratorBase

__version__ = "4.0.0"  # Phase 4 version
__all__ = [
    # Core Phase 4 components
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