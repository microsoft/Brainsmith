"""
Unified Hardware Kernel Generator (HWKG).

This module provides the unified HWKG system that combines RTL parsing with
Interface-Wise Dataflow Modeling, replacing the old template-heavy approach
with DataflowModel-powered instantiation.

This implements the complete HWKG Axiom 10: Unified Architecture Principle
by providing a single integrated system instead of dual architectures.

Core Components:
- UnifiedHWKGGenerator: Main generation class using DataflowModel
- RTLDataflowConverter: RTL → DataflowModel conversion (alias)
- PragmaProcessor: Enhanced pragma processing (alias)

Usage:
    from brainsmith.tools.unified_hwkg import UnifiedHWKGGenerator
    
    generator = UnifiedHWKGGenerator()
    result = generator.generate_from_rtl(rtl_file, compiler_data, output_dir)
"""

from ..unified_hwkg.generator import UnifiedHWKGGenerator
from ..unified_hwkg.converter import RTLDataflowConverter  # Alias
from ..unified_hwkg.pragma_processor import PragmaToStrategyConverter  # Alias

__all__ = [
    'UnifiedHWKGGenerator',
    'RTLDataflowConverter',
    'PragmaToStrategyConverter'
]

# Version info
__version__ = "1.0.0"
__description__ = "Unified Hardware Kernel Generator with Interface-Wise Dataflow Modeling"