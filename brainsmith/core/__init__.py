"""
BrainSmith Core - Simple and Focused

Core functionality for FPGA accelerator design space exploration.
Provides essential tools aligned with North Star goals: Functions Over Frameworks.
"""

from .api import forge, validate_blueprint
from .metrics import DSEMetrics

# Import from new infrastructure locations for backward compatibility
from ..infrastructure.dse.design_space import DesignSpace

__version__ = "0.5.0"
__all__ = ['forge', 'validate_blueprint', 'DesignSpace', 'DSEMetrics']