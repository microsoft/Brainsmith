"""
Hardware Kernel Generator (HWKG) - Phase 3 Unified System

Phase 3 provides a clean, unified system for generating FINN-compatible AutoHWCustomOp 
implementations from SystemVerilog RTL with full end-to-end support.

Key Features:
- UnifiedGenerator: Single interface for all generation tasks
- Phase 2 template system with runtime parameter extraction
- Enhanced data structures with performance tracking
- Clean CLI interface with debug support
- Complete RTL-to-AutoHWCustomOp pipeline

Phase 3 Architecture:
- RTL Parser: SystemVerilog parsing with BDIM pragma support
- UnifiedGenerator: Template-based code generation
- ResultHandler: File writing and metadata generation
- Enhanced data structures: Unified GenerationResult with rich tracking
"""

import warnings

# Core Phase 3 components
from .data import GenerationResult, ValidationResult, PerformanceMetrics
from .config import Config, LegacyConfig
from .cli import main
from .unified_generator import UnifiedGenerator, UnifiedGeneratorError
from .result_handler import ResultHandler

# Legacy import compatibility with deprecation warnings
def __getattr__(name):
    """Handle legacy imports with deprecation warnings."""
    
    # Legacy generator imports
    if name == "HWCustomOpGenerator":
        warnings.warn(
            "HWCustomOpGenerator is deprecated. Use UnifiedGenerator instead.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(f"HWCustomOpGenerator was removed in Phase 3. Use UnifiedGenerator instead.")
    
    elif name == "RTLBackendGenerator":
        warnings.warn(
            "RTLBackendGenerator is deprecated and removed. Use UnifiedGenerator for RTL wrapper generation.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(f"RTLBackendGenerator was removed in Phase 3. Use UnifiedGenerator instead.")
    
    elif name == "TestSuiteGenerator":
        warnings.warn(
            "TestSuiteGenerator is deprecated and removed. Use UnifiedGenerator for test suite generation.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(f"TestSuiteGenerator was removed in Phase 3. Use UnifiedGenerator instead.")
    
    elif name == "GeneratorBase":
        warnings.warn(
            "GeneratorBase is deprecated and removed. Use UnifiedGenerator directly.",
            DeprecationWarning,
            stacklevel=2
        )
        raise ImportError(f"GeneratorBase was removed in Phase 3. Use UnifiedGenerator instead.")
    
    # Unknown attribute
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")

__version__ = "3.0.0"  # Phase 3 version
__all__ = [
    # Core Phase 3 components
    "UnifiedGenerator",
    "UnifiedGeneratorError", 
    "ResultHandler",
    "GenerationResult",
    "ValidationResult", 
    "PerformanceMetrics",
    "Config",
    "LegacyConfig",
    "main",
]