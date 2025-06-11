"""
Simple Optimization Hooks with Strong Extension Points

Streamlined interface that removes academic bloat while maintaining
clean extension points for future capabilities.

This module provides essential optimization event logging with a clean,
extensible architecture that can support sophisticated analysis capabilities
through plugins without affecting the simple core interface.

Basic Usage:
    from brainsmith.hooks import log_optimization_event, log_parameter_change
    
    # Simple event logging
    log_parameter_change('learning_rate', 0.01, 0.005)
    log_optimization_event('dse_completed', {'solutions': 50})

Extension Examples:
    # Future ML analysis plugin
    from brainsmith.hooks.plugins import MLAnalysisPlugin
    hooks.install_plugin('ml_analysis', MLAnalysisPlugin())
    
    # Future statistical monitoring plugin  
    from brainsmith.hooks.plugins import StatisticsPlugin
    hooks.install_plugin('statistics', StatisticsPlugin())
"""

from typing import Any

from .events import (
    log_optimization_event,
    log_parameter_change,
    log_performance_metric,
    log_strategy_decision,
    log_dse_event,
    get_recent_events,
    get_events_by_type,
    get_event_stats,
    clear_event_history,
    register_event_handler,
    register_global_handler,
    create_custom_event_type
)

from .types import (
    OptimizationEvent,
    EventHandler,
    SimpleMetric,
    ParameterChange,
    EventTypes,
    HooksPlugin,
    create_parameter_event,
    create_metric_event,
    create_strategy_event
)

# Import registry system
from .registry import (
    HooksRegistry,
    PluginType,
    PluginInfo,
    HandlerInfo,
    get_hooks_registry,
    discover_all_plugins,
    discover_all_handlers,
    get_plugin_by_name,
    install_hook_plugin,
    list_available_hook_plugins,
    refresh_hooks_registry
)

# Version information
__version__ = "2.0.0"  # Major version for clean refactor
__author__ = "BrainSmith Development Team"

# Clean exports - core functions + extension points + registry
__all__ = [
    # Core event logging functions
    'log_optimization_event',
    'log_parameter_change',
    'log_performance_metric',
    'log_strategy_decision',
    'log_dse_event',
    
    # Event retrieval functions
    'get_recent_events',
    'get_events_by_type',
    'get_event_stats',
    'clear_event_history',
    
    # Extension points
    'register_event_handler',
    'register_global_handler',
    'create_custom_event_type',
    
    # Registry system
    'HooksRegistry',
    'PluginType',
    'PluginInfo',
    'HandlerInfo',
    'get_hooks_registry',
    'discover_all_plugins',
    'discover_all_handlers',
    'get_plugin_by_name',
    'install_hook_plugin',
    'list_available_hook_plugins',
    'refresh_hooks_registry',
    
    # Essential types
    'OptimizationEvent',
    'EventHandler',
    'SimpleMetric',
    'ParameterChange',
    'EventTypes',
    'HooksPlugin',
    
    # Helper functions
    'create_parameter_event',
    'create_metric_event',
    'create_strategy_event'
]

# Module information emphasizing extensibility
MODULE_INFO = {
    'name': 'Simple Optimization Hooks',
    'version': __version__,
    'description': 'Streamlined hooks with strong extension points',
    'features': [
        'Simple event logging',
        'Extensible handler system',
        'Plugin-ready architecture',
        'Future ML/statistics ready',
        'Clean extension interfaces',
        'Memory-efficient core',
        'Optional complexity'
    ],
    'extension_points': [
        'EventHandler interface for custom processing',
        'Plugin system for complex capabilities', 
        'Custom event types for domain-specific events',
        'Global handlers for cross-cutting concerns',
        'Event filters and processors',
        'Serialization and persistence hooks'
    ],
    'complexity_reduction': {
        'files': 'From 5 academic files to 3 simple files',
        'lines': 'From ~2000 academic lines to ~300 simple lines',
        'exports': 'From 19 complex exports to 12 essential exports',
        'philosophy': 'Simple core, extensible future'
    }
}


def get_module_info() -> dict:
    """Get information about the simplified hooks module."""
    return MODULE_INFO.copy()


def show_extension_examples():
    """Display examples of how to extend the hooks system."""
    examples = """
Extension Examples:

1. Custom Event Handler:
    class MyCustomHandler(EventHandler):
        def handle_event(self, event):
            # Custom processing logic
            pass
    
    register_global_handler(MyCustomHandler())

2. Custom Event Type:
    create_custom_event_type('model_validation')
    log_optimization_event('model_validation', {'accuracy': 0.95})

3. Future Plugin (when implemented):
    from brainsmith.hooks.plugins import StatisticsPlugin
    install_plugin('stats', StatisticsPlugin())

4. Event Filtering:
    def my_filter(event):
        return event.event_type == 'parameter_change'
    
    handler = MyHandler()
    handler.should_handle = my_filter
    register_global_handler(handler)
"""
    print(examples)


# Convenience aliases for common usage patterns
def track_parameter(name: str, value: Any, old_value: Any = None) -> None:
    """Convenience function for parameter tracking."""
    if old_value is not None:
        log_parameter_change(name, old_value, value)
    else:
        log_optimization_event('parameter_set', {'parameter': name, 'value': value})


def track_metric(name: str, value: float, **context) -> None:
    """Convenience function for metric tracking."""
    log_performance_metric(name, value, context)


def track_strategy(strategy: str, reason: str = "") -> None:
    """Convenience function for strategy tracking."""
    log_strategy_decision(strategy, reason)


# Legacy compatibility note (no actual compatibility provided)
_MIGRATION_NOTE = """
Migration from Academic Framework:

The previous academic ML/statistics framework has been replaced with
a simple, extensible event system. Key changes:

OLD (Academic):
- StrategyDecisionTracker with ML analysis
- ParameterSensitivityMonitor with statistics  
- ProblemCharacterizer with ML classification
- Complex correlation and effectiveness analysis

NEW (Simple + Extensible):
- log_optimization_event() for basic tracking
- EventHandler interface for custom analysis
- Plugin system for sophisticated capabilities
- Extension points for recreating academic features

To recreate complex analysis:
1. Implement custom EventHandler classes
2. Use the plugin system (when available)
3. Build on the simple event foundation

The simple core provides the same essential functionality
with 90% less complexity and clean extension points.
"""


def show_migration_info():
    """Display migration information."""
    print(_MIGRATION_NOTE)