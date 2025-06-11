"""
Hardware Kernel Generator (HWKG)

A implementation that combines the best of hw_kernel_gen and hw_kernel_gen_simple:
- Simple by default, powerful when needed
- Enhanced with optional BDIM pragma sophistication 
- Based on hw_kernel_gen_simple foundation with selective enhancements
- Follows Interface-Wise Dataflow Modeling axioms

Key Features:
- Rich HWKernel data model with smart property inference
- Safe extraction methods with error resilience
- Clean CLI interface with feature flags for complexity levels
- Template compatibility with existing Jinja2 templates
- Optional BDIM pragma processing for advanced use cases
"""

from .data import GenerationResult
from .rtl_parser.data import HWKernel
from .config import Config
from .cli import main

__version__ = "1.0.0"
__all__ = ["HWKernel", "GenerationResult", "Config", "main"]