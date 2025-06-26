"""
Brainsmith Core V3 - Clean DSE Architecture

This package implements the new three-phase DSE architecture:
1. Design Space Constructor - Parse blueprints into design spaces
2. Design Space Explorer - Systematically explore configurations  
3. Build Runner - Execute builds and collect metrics
"""

__version__ = "3.0.0"

# Phase 1: Design Space Constructor
from .phase1 import forge, ForgeAPI, DesignSpace

# Phase 2: Design Space Explorer
from .phase2 import explore

# Phase 3: Build Runner
from .phase3 import create_build_runner_factory

__all__ = [
    # Phase 1
    "forge", 
    "ForgeAPI", 
    "DesignSpace",
    
    # Phase 2
    "explore",
    
    # Phase 3
    "create_build_runner_factory",
]