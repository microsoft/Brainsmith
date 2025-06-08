"""
Kernels Library - Week 2 Implementation

Organizes existing custom_op/ functionality into a structured library
providing kernel discovery, registration, and parameter mapping.
"""

from .library import KernelsLibrary
from .registry import KernelRegistry, discover_kernels
from .mapping import ParameterMapper

__all__ = [
    'KernelsLibrary',
    'KernelRegistry', 
    'discover_kernels',
    'ParameterMapper'
]

# Version info
__version__ = "1.0.0"