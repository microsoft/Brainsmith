"""
Brainsmith Libraries - Week 2 Implementation

Core library system providing organized access to kernels, transforms,
hardware optimization, and analysis functionality.
"""

from .base import BaseLibrary, LibraryRegistry, LibraryManager

# Import concrete libraries
try:
    from .kernels import KernelsLibrary
except ImportError:
    KernelsLibrary = None

try:
    from .transforms import TransformsLibrary
except ImportError:
    TransformsLibrary = None

try:
    from .hw_optim import HardwareOptimizationLibrary
except ImportError:
    HardwareOptimizationLibrary = None

try:
    from .analysis import AnalysisLibrary
except ImportError:
    AnalysisLibrary = None

__all__ = [
    'BaseLibrary',
    'LibraryRegistry', 
    'LibraryManager',
    'KernelsLibrary',
    'TransformsLibrary',
    'HardwareOptimizationLibrary',
    'AnalysisLibrary'
]

# Version info
__version__ = "0.4.0"  # Week 2 implementation