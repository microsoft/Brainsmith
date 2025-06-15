"""
BrainSmith Core - Simple and Focused

Core functionality for FPGA accelerator design space exploration.
Provides essential tools aligned with North Star goals: Functions Over Frameworks.

North Star Promise: result = brainsmith.core.forge('model.onnx', 'blueprint.yaml')
Essential Functions: forge + 12 helpers + 3 classes for complete FPGA workflow
"""

# Primary North Star function
from .api import forge, validate_blueprint
from .api_v2 import forge_v2, validate_blueprint_v2
from . import finn_v2
from .metrics import DSEMetrics

# Essential classes (3 core concepts)
from .dse.design_space import DesignSpace
from .dse.interface import DSEInterface

# North Star Helper Functions (12 essential functions)
# 1-4: Automation helpers
from brainsmith.libraries.automation import (
    parameter_sweep,
    batch_process,
    find_best,
    aggregate_stats
)

# 5-7: Hooks and event management
from .hooks import (
    log_optimization_event,
    register_event_handler
)

# 8: FINN accelerator building
from .finn import build_accelerator

# 9-11: Data management and analysis
from .data import (
    collect_dse_metrics as get_analysis_data,
    export_metrics as export_results
)

# 12: Design space management
from .dse import sample_design_space

# Registry infrastructure (for advanced users)
from .registry import BaseRegistry, ComponentInfo
from .hooks.registry import HooksRegistry, get_hooks_registry

__version__ = "0.5.0"

# North Star API - Primary function + 12 helpers + 3 classes
__all__ = [
    # PRIMARY FUNCTION (North Star Promise)
    'forge',
    
    # 12 ESSENTIAL HELPER FUNCTIONS
    'parameter_sweep',        # 1. Parameter exploration
    'find_best',              # 2. Result optimization (alias: find_best_result)
    'batch_process',          # 3. Batch processing
    'aggregate_stats',        # 4. Statistical analysis
    'log_optimization_event', # 5. Event logging
    'register_event_handler', # 6. Event handling
    'build_accelerator',      # 7. FINN accelerator building
    'get_analysis_data',      # 8. Data analysis (alias: collect_dse_metrics)
    'export_results',         # 9. Data export (alias: export_metrics)
    'sample_design_space',    # 10. Design space sampling
    'validate_blueprint',     # 11. Blueprint validation
    # 12th function will be added based on needs
    
    # 3 ESSENTIAL CLASSES (Core concepts)
    'DesignSpace',            # Design space representation
    'DSEInterface',           # Design space exploration interface
    'DSEMetrics',             # Metrics collection and analysis
    
    # REGISTRY INFRASTRUCTURE (Advanced)
    'BaseRegistry',
    'ComponentInfo',
    'HooksRegistry',
    'get_hooks_registry'
]

# Convenience aliases for North Star compliance
find_best_result = find_best  # Alias for more descriptive name
