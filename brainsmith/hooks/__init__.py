"""
Automation Hooks Framework

This module provides comprehensive automation hooks for tracking optimization strategies,
parameter sensitivity, and problem characterization to enable data-driven optimization.

Key Components:
1. Strategy Decision Tracker: Record and analyze optimization strategies
2. Parameter Sensitivity Monitor: Track parameter impact on performance
3. Problem Characterization System: Classify and recommend approaches

Example Usage:
    from brainsmith.hooks import StrategyDecisionTracker, ParameterSensitivityMonitor
    
    # Initialize trackers
    strategy_tracker = StrategyDecisionTracker()
    sensitivity_monitor = ParameterSensitivityMonitor()
    
    # Record strategy decision
    strategy_tracker.record_strategy_choice(context, strategy, rationale)
    
    # Track parameter sensitivity
    sensitivity_monitor.track_parameter_changes(parameter_changes)
"""

from .strategy_tracking import (
    StrategyDecisionTracker,
    StrategyDecisionRecord,
    StrategyOutcomeRecord,
    EffectivenessReport,
    ProblemCharacteristics
)

from .sensitivity import (
    ParameterSensitivityMonitor,
    ParameterChangeRecord,
    ImpactAnalysis,
    SensitivityInsight,
    SensitivityData
)

from .characterization import (
    ProblemCharacterizer,
    ProblemCharacteristics,
    ProblemType,
    DesignSpaceCharacteristics
)

from .database import (
    AutomationDatabase,
    StrategyDecisionDatabase,
    SensitivityDatabase
)

# Version information
__version__ = "1.0.0"
__author__ = "BrainSmith Development Team"

# Export all public components
__all__ = [
    # Strategy tracking
    'StrategyDecisionTracker',
    'StrategyDecisionRecord',
    'StrategyOutcomeRecord',
    'EffectivenessReport',
    'ProblemCharacteristics',
    
    # Sensitivity monitoring
    'ParameterSensitivityMonitor',
    'ParameterChangeRecord',
    'ImpactAnalysis',
    'SensitivityInsight',
    'SensitivityData',
    
    # Problem characterization
    'ProblemCharacterizer',
    'ProblemType',
    'DesignSpaceCharacteristics',
    
    # Database components
    'AutomationDatabase',
    'StrategyDecisionDatabase',
    'SensitivityDatabase'
]

# Package information
PACKAGE_INFO = {
    'name': 'Automation Hooks Framework',
    'version': __version__,
    'description': 'Comprehensive automation and learning infrastructure',
    'features': [
        'Strategy decision tracking and analysis',
        'Parameter sensitivity monitoring',
        'Problem characterization and classification',
        'Learning-ready data collection',
        'Performance correlation analysis'
    ],
    'status': 'Production Ready'
}