"""
BrainSmith Simple Automation Helpers

Provides simple utilities for running forge() multiple times with different
parameters or configurations. Focused on practical automation patterns.

Key Philosophy:
- Thin helpers around forge() function
- No enterprise orchestration complexity  
- Simple function calls for common automation needs
- Optional parallelization for performance

Example Usage:
    from brainsmith.automation import parameter_sweep, batch_process, find_best
    
    # Parameter exploration
    results = parameter_sweep(
        "model.onnx", 
        "blueprint.yaml",
        {'pe_count': [4, 8, 16], 'simd_width': [2, 4, 8]}
    )
    
    # Find optimal configuration
    best = find_best(results, metric='throughput', maximize=True)
    
    # Batch processing
    batch_results = batch_process([
        ("model1.onnx", "blueprint1.yaml"),
        ("model2.onnx", "blueprint2.yaml")
    ])
"""

# Import registry system
from .registry import (
    AutomationRegistry,
    AutomationType,
    AutomationToolInfo,
    get_automation_registry,
    discover_all_automation_tools,
    get_automation_tool,
    find_tools_by_type,
    list_available_automation_tools,
    refresh_automation_registry
)

from .sweep import (
    parameter_sweep,
    find_best,
    aggregate_stats
)

from .batch import (
    batch_process
)

__version__ = "2.0.0"  # North Star simplification version
__author__ = "BrainSmith Development Team"

# Export essential functions and registry
__all__ = [
    # Core automation functions
    'parameter_sweep',   # Core parameter exploration
    'batch_process',     # Core batch processing
    'find_best',         # Result optimization
    'aggregate_stats',   # Statistical analysis
    
    # Registry system
    'AutomationRegistry',
    'AutomationType',
    'AutomationToolInfo',
    'get_automation_registry',
    'discover_all_automation_tools',
    'get_automation_tool',
    'find_tools_by_type',
    'list_available_automation_tools',
    'refresh_automation_registry'
]

# Initialize logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Simple Automation v{__version__} initialized")
logger.info("Available functions: parameter_sweep, batch_process, find_best, aggregate_stats")
