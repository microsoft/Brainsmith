"""
BrainSmith Simple Automation Helpers

Provides simple utilities for common automation patterns in FPGA design space exploration.
Instead of complex workflow orchestration, these helpers make it easy to run forge() 
multiple times with different parameters or configurations.

Key Philosophy:
- Simple helpers that leverage existing forge() function
- No enterprise workflow orchestration
- Focus on practical automation patterns users actually need
- Minimal complexity, maximum utility

Example Usage:
    from brainsmith.automation import parameter_sweep, batch_process
    
    # Parameter sweep
    results = parameter_sweep(
        "model.onnx", 
        "blueprint.yaml",
        {'pe_count': [4, 8, 16], 'simd_width': [2, 4, 8]}
    )
    
    # Batch processing
    results = batch_process([
        ("model1.onnx", "blueprint1.yaml"),
        ("model2.onnx", "blueprint2.yaml")
    ])
"""

from .parameter_sweep import (
    parameter_sweep,
    grid_search,
    random_search
)

from .batch_processing import (
    batch_process,
    multi_objective_runs,
    configuration_sweep
)

from .utils import (
    aggregate_results,
    find_best_result,
    find_top_results,
    save_automation_results,
    load_automation_results,
    compare_automation_runs
)

__version__ = "0.1.0"
__author__ = "BrainSmith Development Team"

# Export simple automation helpers
__all__ = [
    # Parameter exploration
    'parameter_sweep',
    'grid_search', 
    'random_search',
    
    # Batch processing
    'batch_process',
    'multi_objective_runs',
    'configuration_sweep',
    
    # Result analysis
    'aggregate_results',
    'find_best_result',
    'find_top_results',
    'save_automation_results',
    'load_automation_results',
    'compare_automation_runs'
]

# Initialize logging
import logging
logger = logging.getLogger(__name__)
logger.info(f"BrainSmith Simple Automation v{__version__} initialized")
logger.info("Available helpers: parameter_sweep, batch_process, result aggregation")
