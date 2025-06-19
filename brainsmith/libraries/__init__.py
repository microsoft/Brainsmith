"""
Brainsmith Libraries Package

Collection of specialized libraries for AI/ML hardware acceleration:
- analysis: Analysis tools and profiling capabilities
- automation: Batch processing and parameter sweep automation
- blueprints: Hardware accelerator blueprint library
- kernels: Hardware kernel implementations and registry
- operators: Custom ONNX operators and future Canonical Op objects
- transforms: Model transformation operations and steps
"""

# Import submodule registries for unified access
from . import analysis
from . import automation  
from . import blueprints
from . import kernels
from . import operators
from . import transforms

__all__ = [
    "analysis",
    "automation", 
    "blueprints",
    "kernels",
    "operators",
    "transforms"
]