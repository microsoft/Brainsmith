"""
FINN Integration Platform

This module provides deep integration with FINN's four-category interface for dataflow
accelerator design optimization.

Key Components:
1. FINN Integration Engine: Core integration with FINN's four-category interface
2. Configuration Managers: Specialized managers for each FINN category
3. Build Result Processing: Comprehensive analysis of FINN build outputs
4. Error Handling Framework: Robust error diagnosis and recovery
5. Build Orchestration: Sophisticated build management with monitoring

Example Usage:
    from brainsmith.finn import FINNIntegrationEngine
    
    # Initialize FINN integration
    engine = FINNIntegrationEngine()
    
    # Configure FINN interface from Brainsmith parameters
    finn_config = engine.configure_finn_interface(brainsmith_config)
    
    # Execute FINN build with monitoring
    result = engine.execute_finn_build(finn_config, design_point)
"""

# Core engine
from .engine import FINNIntegrationEngine

# Data types and configurations
from .types import (
    FINNInterfaceConfig,
    ModelOpsConfig,
    ModelTransformsConfig,
    HwKernelsConfig,
    HwOptimizationConfig,
    EnhancedFINNResult,
    FINNBuildResult,
    PerformanceMetrics,
    ResourceAnalysis,
    TimingAnalysis,
    BuildEnvironment,
    OptimizationStrategy,
    BuildStatus
)

# Configuration managers
from .model_ops_manager import ModelOpsManager
from .model_transforms_manager import ModelTransformsManager
from .hw_kernels_manager import HwKernelsManager
from .hw_optimization_manager import HwOptimizationManager

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Core integration engine
    'FINNIntegrationEngine',
    
    # Data types and configurations
    'FINNInterfaceConfig',
    'ModelOpsConfig',
    'ModelTransformsConfig',
    'HwKernelsConfig',
    'HwOptimizationConfig',
    'EnhancedFINNResult',
    'FINNBuildResult',
    'PerformanceMetrics',
    'ResourceAnalysis',
    'TimingAnalysis',
    'BuildEnvironment',
    'OptimizationStrategy',
    'BuildStatus',
    
    # Configuration managers
    'ModelOpsManager',
    'ModelTransformsManager',
    'HwKernelsManager',
    'HwOptimizationManager'
]

# Module-level convenience functions
def create_finn_engine() -> FINNIntegrationEngine:
    """Create a new FINN Integration Engine instance"""
    return FINNIntegrationEngine()

def get_supported_features() -> dict:
    """Get all supported FINN features"""
    engine = FINNIntegrationEngine()
    return engine.get_supported_features()

def validate_finn_config(config: FINNInterfaceConfig) -> bool:
    """Validate a FINN interface configuration"""
    engine = FINNIntegrationEngine()
    return engine.validate_configuration(config)

# Package information
PACKAGE_INFO = {
    'name': 'FINN Integration Platform',
    'version': __version__,
    'description': 'Deep integration with FINN dataflow accelerator framework',
    'features': [
        'Four-category interface integration',
        'Intelligent configuration management',
        'Enhanced build result processing',
        'Comprehensive error handling',
        'Performance analysis and optimization'
    ],
    'status': 'Production Ready'
}