"""
FINN Kernel Management System

This module provides comprehensive FINN kernel discovery, registration, and selection
capabilities for dataflow accelerator design optimization.

Key Components:
1. Kernel Discovery: Automated FINN kernel discovery and analysis
2. Kernel Registry: Database and management of available FINN kernels
3. Performance Modeling: Analytical and empirical performance models
4. Model Analysis: Topology analysis for kernel requirement extraction
5. Kernel Selection: Intelligent kernel selection for optimization targets
6. Configuration Generation: Automated FINN build configuration

Example Usage:
    from brainsmith.kernels import FINNKernelRegistry, FINNKernelSelector, ModelTopologyAnalyzer
    
    # Initialize kernel registry and discover FINN kernels
    registry = FINNKernelRegistry()
    registry.discover_finn_kernels("/path/to/finn")
    
    # Analyze model topology
    analyzer = ModelTopologyAnalyzer()
    analysis = analyzer.analyze_model_structure(model)
    
    # Select optimal kernels
    selector = FINNKernelSelector(registry)
    selection_plan = selector.select_optimal_kernels(
        requirements=analysis.operator_requirements,
        targets={"throughput": 1000, "latency": 10},
        constraints={"luts": 100000, "dsps": 2000}
    )
    
    # Generate FINN configuration
    from brainsmith.kernels.finn_config import FINNConfigGenerator
    config_gen = FINNConfigGenerator()
    finn_config = config_gen.generate_build_config(selection_plan)
"""

import logging
from typing import List, Dict, Any, Optional, Union

# Core kernel management components
from .discovery import (
    FINNKernelDiscovery,
    KernelInfo,
    KernelMetadata,
    ParameterSchema
)

# Kernel database and registry
from .database import (
    FINNKernelInfo,
    FINNKernelDatabase,
    PerformanceModel,
    ResourceRequirements,
    OperatorType,
    BackendType,
    PerformanceClass,
    ResourceType
)

from .registry import (
    FINNKernelRegistry,
    RegistrationResult,
    SearchCriteria,
    CompatibilityChecker
)

# Model analysis and topology
from .analysis import (
    ModelTopologyAnalyzer,
    TopologyAnalysis,
    OperatorRequirement,
    DataflowConstraints,
    LayerInfo,
    TensorShape,
    LayerType,
    DataType,
    ModelGraph
)

# Kernel selection and optimization
from .selection import (
    FINNKernelSelector,
    SelectionPlan,
    ParameterConfiguration,
    PerformanceTargets,
    ResourceConstraints,
    KernelParameterConfig,
    KernelSelection,
    OptimizationObjective
)

# FINN configuration generation
from .finn_config import (
    FINNConfigGenerator,
    FINNBuildConfig,
    FoldingConfig,
    OptimizationDirectives,
    ModelOpsConfig,
    ModelTransformsConfig,
    HwKernelsConfig,
    HwOptimizationConfig,
    LayerFoldingConfig,
    FINNConfigurationError
)

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Discovery and metadata
    'FINNKernelDiscovery',
    'KernelInfo',
    'KernelMetadata',
    'ParameterSchema',
    
    # Database and registry
    'FINNKernelInfo',
    'FINNKernelDatabase',
    'FINNKernelRegistry',
    'RegistrationResult',
    'SearchCriteria',
    'CompatibilityChecker',
    'OperatorType',
    'BackendType',
    'PerformanceClass',
    'ResourceType',
    
    # Performance modeling
    'PerformanceModel',
    'ResourceRequirements',
    
    # Model analysis
    'ModelTopologyAnalyzer',
    'TopologyAnalysis',
    'OperatorRequirement',
    'DataflowConstraints',
    'LayerInfo',
    'TensorShape',
    'LayerType',
    'DataType',
    'ModelGraph',
    
    # Kernel selection
    'FINNKernelSelector',
    'SelectionPlan',
    'ParameterConfiguration',
    'PerformanceTargets',
    'ResourceConstraints',
    'KernelParameterConfig',
    'KernelSelection',
    'OptimizationObjective',
    
    # Configuration generation
    'FINNConfigGenerator',
    'FINNBuildConfig',
    'FoldingConfig',
    'OptimizationDirectives',
    'ModelOpsConfig',
    'ModelTransformsConfig',
    'HwKernelsConfig',
    'HwOptimizationConfig',
    'LayerFoldingConfig',
    'FINNConfigurationError'
]

# Initialize logging
logger.info(f"FINN Kernel Management System v{__version__} initialized")
logger.info("Available capabilities: Discovery, Registry, Performance Modeling, Selection, Configuration")

# Convenience functions for common workflows
def create_kernel_registry(finn_path: Optional[str] = None, 
                          database_path: Optional[str] = None) -> FINNKernelRegistry:
    """
    Create and initialize a FINN kernel registry
    
    Args:
        finn_path: Path to FINN installation for automatic discovery
        database_path: Path to kernel database file
        
    Returns:
        Initialized FINNKernelRegistry
    """
    registry = FINNKernelRegistry(database_path)
    
    if finn_path:
        logger.info(f"Discovering FINN kernels from: {finn_path}")
        discovered = registry.discover_finn_kernels(finn_path, register_automatically=True)
        logger.info(f"Discovered and registered {len(discovered)} kernels")
    
    return registry

def analyze_model_for_finn(model_data: Dict[str, Any]) -> TopologyAnalysis:
    """
    Analyze model topology for FINN kernel requirements
    
    Args:
        model_data: Model data in supported format
        
    Returns:
        TopologyAnalysis with operator requirements
    """
    model = ModelGraph(model_data)
    analyzer = ModelTopologyAnalyzer()
    return analyzer.analyze_model_structure(model)

def select_optimal_kernels_for_model(model_data: Dict[str, Any],
                                   registry: FINNKernelRegistry,
                                   performance_targets: Dict[str, float],
                                   resource_constraints: Dict[str, int],
                                   strategy: str = 'balanced') -> SelectionPlan:
    """
    End-to-end kernel selection for a model
    
    Args:
        model_data: Model data in supported format
        registry: Initialized kernel registry
        performance_targets: Performance targets (throughput, latency, etc.)
        resource_constraints: Resource constraints (luts, dsps, etc.)
        strategy: Selection strategy ('balanced', 'performance', 'area', etc.)
        
    Returns:
        SelectionPlan with optimal kernel selections
    """
    # Analyze model
    analysis = analyze_model_for_finn(model_data)
    
    # Create selector
    selector = FINNKernelSelector(registry)
    
    # Convert targets and constraints to proper format
    targets = PerformanceTargets(
        throughput=performance_targets.get('throughput'),
        latency=performance_targets.get('latency'),
        power=performance_targets.get('power'),
        area=performance_targets.get('area')
    )
    
    constraints = ResourceConstraints(
        max_luts=resource_constraints.get('luts'),
        max_dsps=resource_constraints.get('dsps'),
        max_brams=resource_constraints.get('brams')
    )
    
    # Select kernels
    return selector.select_optimal_kernels(
        requirements=analysis.operator_requirements,
        targets=targets,
        constraints=constraints,
        selection_strategy=strategy
    )

def generate_finn_config_for_model(model_data: Dict[str, Any],
                                 registry: FINNKernelRegistry,
                                 performance_targets: Dict[str, float],
                                 resource_constraints: Dict[str, int],
                                 output_path: Optional[str] = None) -> FINNBuildConfig:
    """
    Complete workflow: analyze model, select kernels, generate FINN config
    
    Args:
        model_data: Model data in supported format
        registry: Initialized kernel registry
        performance_targets: Performance targets
        resource_constraints: Resource constraints
        output_path: Optional path to save configuration
        
    Returns:
        Complete FINNBuildConfig ready for FINN
    """
    # Select optimal kernels
    selection_plan = select_optimal_kernels_for_model(
        model_data, registry, performance_targets, resource_constraints
    )
    
    # Generate FINN configuration
    config_generator = FINNConfigGenerator()
    finn_config = config_generator.generate_build_config(selection_plan)
    
    # Save if path provided
    if output_path:
        finn_config.save_to_file(output_path)
        logger.info(f"FINN configuration saved to: {output_path}")
    
    return finn_config

def get_kernel_statistics(registry: FINNKernelRegistry) -> Dict[str, Any]:
    """
    Get comprehensive kernel registry statistics
    
    Args:
        registry: Kernel registry to analyze
        
    Returns:
        Dictionary with detailed statistics
    """
    stats = registry.get_registry_statistics()
    
    # Add convenience summaries
    db_stats = stats['database_stats']
    stats['summary'] = {
        'total_kernels': db_stats['total_kernels'],
        'operator_coverage': len(db_stats.get('by_operator_type', {})),
        'backend_coverage': len(db_stats.get('by_backend_type', {})),
        'verified_kernels': db_stats.get('verification_status', {}).get('verified', 0)
    }
    
    return stats