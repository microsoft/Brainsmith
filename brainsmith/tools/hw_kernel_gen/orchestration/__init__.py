"""
Orchestration Components for Hardware Kernel Generator.

This module provides orchestration capabilities for the HWKG, including:
- Generator factory and registry
- Pipeline orchestration and execution
- Workflow engine and management
- Generator lifecycle and pooling
- Integration coordination

Week 3 Implementation: Generator Factory and Pipeline Orchestrator Architecture
"""

from .generator_factory import (
    GeneratorFactory, GeneratorRegistry, GeneratorCache,
    GeneratorConfiguration, create_generator_factory,
    register_generator, get_generator_capabilities
)

from .pipeline_orchestrator import (
    PipelineOrchestrator, PipelineStage, PipelineExecutor,
    StageCoordinator, create_pipeline_orchestrator,
    execute_pipeline, PipelineContext
)

from .generation_workflow import (
    WorkflowEngine, WorkflowDefinition, WorkflowStep,
    WorkflowContext, create_workflow_engine,
    execute_workflow, WorkflowResult
)

from .generator_management import (
    GeneratorManager, GeneratorLifecycle, GeneratorPool,
    GeneratorMetrics, create_generator_manager,
    get_generator_statistics
)

from .integration_orchestrator import (
    IntegrationOrchestrator, ComponentIntegrator,
    ResultsAggregator, ValidationCoordinator,
    create_integration_orchestrator, run_integrated_generation
)

from .workflow_definitions import (
    StandardWorkflows, CustomWorkflowBuilder,
    WorkflowTemplate, create_standard_workflow
)

__all__ = [
    # Generator Factory
    'GeneratorFactory', 'GeneratorRegistry', 'GeneratorCache',
    'GeneratorConfiguration', 'create_generator_factory',
    'register_generator', 'get_generator_capabilities',
    
    # Pipeline Orchestrator
    'PipelineOrchestrator', 'PipelineStage', 'PipelineExecutor',
    'StageCoordinator', 'create_pipeline_orchestrator',
    'execute_pipeline', 'PipelineContext',
    
    # Workflow Engine
    'WorkflowEngine', 'WorkflowDefinition', 'WorkflowStep',
    'WorkflowContext', 'create_workflow_engine',
    'execute_workflow', 'WorkflowResult',
    
    # Generator Management
    'GeneratorManager', 'GeneratorLifecycle', 'GeneratorPool',
    'GeneratorMetrics', 'create_generator_manager',
    'get_generator_statistics',
    
    # Integration Orchestrator
    'IntegrationOrchestrator', 'ComponentIntegrator',
    'ResultsAggregator', 'ValidationCoordinator',
    'create_integration_orchestrator', 'run_integrated_generation',
    
    # Workflow Definitions
    'StandardWorkflows', 'CustomWorkflowBuilder',
    'WorkflowTemplate', 'create_standard_workflow'
]

# Version info
__version__ = "3.0.0"
__phase__ = "Week 3: Generator Factory and Pipeline Orchestrator"