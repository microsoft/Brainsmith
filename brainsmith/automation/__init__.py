"""
Automated Workflows and Integration Framework

This module provides comprehensive automation capabilities for FPGA design optimization,
integrating all BrainSmith components into cohesive automated workflows.

Key Features:
- Automated end-to-end design optimization workflows
- Intelligent design recommendation system
- Adaptive workflow orchestration based on design requirements
- Historical learning and pattern recognition
- Quality-driven automation with confidence thresholds
- Integration of DSE, selection, and analysis components

Main Components:
1. Workflow Engine: Core automation and orchestration
2. Recommendation System: AI-driven design suggestions
3. Quality Assurance: Automated validation and quality control
4. Learning System: Historical pattern recognition and adaptation
5. Integration Layer: Seamless component coordination

Example Usage:
    from brainsmith.automation import AutomationEngine, WorkflowConfiguration
    
    # Create automated workflow
    config = WorkflowConfiguration(
        optimization_budget=3600,  # 1 hour
        quality_threshold=0.85,
        enable_learning=True
    )
    
    engine = AutomationEngine(config)
    
    # Run automated design optimization
    result = engine.optimize_design(
        application_spec="cnn_inference",
        performance_targets={"throughput": 200, "power": 15},
        constraints={"lut_budget": 0.8, "timing_closure": True}
    )
    
    print(f"Best design: {result.recommended_design}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Recommendations: {result.improvement_suggestions}")
"""

import logging
from typing import List, Dict, Any, Optional, Union

# Core automation components (will be implemented)
from .engine import (
    AutomationEngine,
    WorkflowResult,
    WorkflowConfiguration,
    AutomationMetrics
)

# Workflow orchestration
from .workflows import (
    DesignOptimizationWorkflow,
    BenchmarkingWorkflow,
    AnalysisWorkflow,
    WorkflowManager,
    WorkflowStatus,
    WorkflowStep
)

# Recommendation system
from .recommendations import (
    RecommendationEngine,
    DesignRecommendation,
    RecommendationCategory,
    RecommendationConfidence
)

# Quality assurance
from .quality import (
    QualityController,
    QualityMetrics,
    ValidationResult
)

# Learning and adaptation
from .learning import (
    LearningEngine,
    HistoricalPatterns,
    AdaptiveParameters
)

# Integration layer
from .integration import (
    ComponentIntegrator,
    IntegrationResult,
    ComponentStatus
)

# Data models
from .models import (
    AutomationContext,
    WorkflowDefinition,
    DesignTarget,
    OptimizationJob,
    AutomationResult
)

# Utilities
from .utils import (
    create_automation_context,
    validate_workflow_config,
    generate_workflow_id
)

# Setup logging
logger = logging.getLogger(__name__)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Core engine
    'AutomationEngine',
    'WorkflowResult',
    'WorkflowConfiguration',
    'AutomationMetrics',
    
    # Workflow management
    'DesignOptimizationWorkflow',
    'BenchmarkingWorkflow',
    'AnalysisWorkflow',
    'WorkflowManager',
    'WorkflowStatus',
    'WorkflowStep',
    
    # Recommendation system
    'RecommendationEngine',
    'DesignRecommendation',
    'RecommendationCategory',
    'RecommendationConfidence',
    
    # Quality assurance
    'QualityController',
    'QualityMetrics',
    'ValidationResult',
    
    # Learning system
    'LearningEngine',
    'HistoricalPatterns',
    'AdaptiveParameters',
    
    # Integration layer
    'ComponentIntegrator',
    'IntegrationResult',
    'ComponentStatus',
    
    # Data models
    'AutomationContext',
    'WorkflowDefinition',
    'DesignTarget',
    'OptimizationJob',
    'AutomationResult',
    
    # Utilities
    'create_automation_context',
    'validate_workflow_config',
    'generate_workflow_id'
]

# Initialize logging
logger.info(f"Automation Framework v{__version__} initialized")
logger.info("Available automation capabilities: Workflows, Recommendations, Quality Control, Learning")
