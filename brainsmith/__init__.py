# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

"""
BrainSmith: Simple FPGA accelerator design space exploration

Organized by user workflow stages:
• Core DSE (5-minute success)
• Automation (15-minute success)  
• Analysis & Monitoring (30-minute success)
• Advanced Building (1-hour success)
• Extensibility (contributor-focused)

North Star Promise: result = brainsmith.forge('model.onnx', 'blueprint.yaml')
"""

# Explicit dependency check - fail fast if missing
from .dependencies import check_installation
check_installation()

# === 🔧 CUSTOM OPERATORS REGISTRATION ===
# Register custom operators with QONNX using modern entry points approach
# Import kernels to ensure they register themselves with QONNX
try:
    # Import all kernel modules to trigger @register_op decorators
    from .libraries.kernels.layernorm import layernorm
    from .libraries.kernels.softmax import hwsoftmax
    from .libraries.kernels.shuffle import shuffle
    from .libraries.kernels.crop import crop
    import logging
    logging.getLogger(__name__).info("BrainSmith custom operators registered successfully")
except ImportError as e:
    import logging
    logging.getLogger(__name__).warning(f"Some custom operators not available: {e}")

# === 🎯 CORE DSE (5-minute success) ===
from .core.api import forge, validate_blueprint
from .core.dse.combination_generator import ComponentCombination as DesignSpace
from .core.dse.space_explorer import DesignSpaceExplorer as DSEInterface
from .core.metrics import DSEMetrics

# === ⚡ AUTOMATION (15-minute success) ===
from .libraries.automation import (
    parameter_sweep,    # Explore parameter combinations
    batch_process,      # Process multiple models
    find_best,          # Find optimal results
    aggregate_stats     # Statistical summaries
)

# === 📊 ANALYSIS & MONITORING (30-minute success) ===
from .core.hooks import (
    log_optimization_event,     # Event tracking
    register_event_handler      # Custom monitoring
)
from .core.data import (
    collect_dse_metrics as get_analysis_data,  # Data extraction
    export_metrics as export_results           # Data export
)

# === 🔧 ADVANCED BUILDING (1-hour success) ===
from .core.finn import FINNEvaluationBridge as build_accelerator      # FINN integration
from .core.dse.combination_generator import generate_component_combinations as sample_design_space     # Sampling

# === 🔌 EXTENSIBILITY (contributor-focused) ===
from .core.registry import BaseRegistry, ComponentInfo
from .core.hooks.registry import HooksRegistry, get_hooks_registry

# === 📋 STRUCTURED EXPORTS ===
__all__ = [
    # === CORE DSE (Start here - 5 minutes to success) ===
    'forge',              # Primary function: model + blueprint → accelerator
    'validate_blueprint', # Validate configuration before DSE
    'DesignSpace',        # Design space representation  
    'DSEInterface',       # Design space exploration engine
    'DSEMetrics',         # Performance metrics collection
    
    # === AUTOMATION (Scale up - 15 minutes to success) ===
    'parameter_sweep',    # Explore parameter combinations automatically
    'batch_process',      # Process multiple model/blueprint pairs
    'find_best',          # Find optimal results by metric
    'aggregate_stats',    # Generate statistical summaries
    
    # === ANALYSIS & MONITORING (Integrate - 30 minutes to success) ===
    'log_optimization_event',   # Track optimization events
    'register_event_handler',   # Custom monitoring and callbacks
    'get_analysis_data',        # Extract data for external analysis
    'export_results',           # Export to pandas, CSV, JSON
    
    # === ADVANCED BUILDING (Master - 1 hour to success) ===
    'build_accelerator',        # Generate FINN accelerator
    'sample_design_space',      # Advanced design space sampling
    
    # === EXTENSIBILITY (Contributors) ===
    'BaseRegistry',             # Foundation for component discovery
    'ComponentInfo',            # Component metadata interface
    'HooksRegistry',            # Plugin and handler management
    'get_hooks_registry'        # Registry access
]

# === 🎯 WORKFLOW HELPERS ===
class workflows:
    """Common workflow patterns for quick access"""
    
    @staticmethod
    def quick_dse(model_path: str, blueprint_path: str):
        """5-minute workflow: Basic DSE"""
        return forge(model_path, blueprint_path)
    
    @staticmethod 
    def parameter_exploration(model_path: str, blueprint_path: str, params: dict):
        """15-minute workflow: Parameter sweep + optimization"""
        results = parameter_sweep(model_path, blueprint_path, params)
        return find_best(results, metric='throughput')
    
    @staticmethod
    def full_analysis(model_path: str, blueprint_path: str, params: dict, export_path: str = None):
        """30-minute workflow: Full DSE + analysis + export"""
        results = parameter_sweep(model_path, blueprint_path, params)
        best = find_best(results, metric='throughput')
        stats = aggregate_stats(results)
        data = get_analysis_data(results)
        
        if export_path:
            export_results(data, export_path)
            
        return {'best': best, 'stats': stats, 'data': data}

# === 📚 LEARNING PATH ===
def help():
    """Show learning path for new users"""
    return """
🎯 BrainSmith Learning Path

⏱️  5 minutes:  result = brainsmith.forge('model.onnx', 'blueprint.yaml')
⏱️  15 minutes: results = brainsmith.parameter_sweep(model, blueprint, params)
⏱️  30 minutes: data = brainsmith.get_analysis_data(results)
⏱️  1 hour:     accelerator = brainsmith.build_accelerator(model, blueprint)

📖 Full documentation: https://brainsmith.readthedocs.io
🔧 Examples: brainsmith.workflows.quick_dse(model, blueprint)
"""

# Convenience aliases for North Star compliance
find_best_result = find_best  # Alias for more descriptive name

# Version information
__version__ = "1.0.0"  # Clean refactor version
__author__ = "Microsoft Research"
__description__ = "Simple FPGA accelerator design space exploration platform"
