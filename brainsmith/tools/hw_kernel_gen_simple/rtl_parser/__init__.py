"""
Simple RTL parser interface for HWKG.

Provides a clean interface to the RTL parser without exposing implementation complexity.
"""

from .simple_parser import parse_rtl_file, RTLData

__all__ = ['parse_rtl_file', 'RTLData']