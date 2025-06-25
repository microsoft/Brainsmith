"""
Brainsmith Core V3 - Clean DSE Architecture

This package implements the new three-phase DSE architecture:
1. Design Space Constructor - Parse blueprints into design spaces
2. Design Space Explorer - Systematically explore configurations  
3. Build Runner - Execute builds and collect metrics
"""

__version__ = "3.0.0"

# Phase 1 is now implemented
from .phase1 import forge, ForgeAPI, DesignSpace

# Future phases
# from .phase2 import explore
# from .phase3 import run

__all__ = ["forge", "ForgeAPI", "DesignSpace"]