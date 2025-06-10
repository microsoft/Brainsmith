"""
Unified RTL parser following RTL Parser Axioms.

Enhanced version of hw_kernel_gen_simple parser with optional BDIM 
sophistication while maintaining error resilience and safe extraction methods.
"""

from .unified_parser import UnifiedRTLParser, parse_rtl_file

__all__ = ["UnifiedRTLParser", "parse_rtl_file"]