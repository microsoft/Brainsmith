"""
Example Extensions (Future Capability Demos)

These examples show how to extend the simple core with
sophisticated capabilities without bloating the base system.
"""

from ..types import EventHandler, OptimizationEvent, HooksPlugin
from typing import List, Dict, Any
import logging
from collections import defaultdict
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class SimpleStatisticsHandler(EventHandler):
    """Example: Simple statistics collection (extensible to academic level)."""
    
    def __init__(self):
        self.parameter_changes = []
        self.performance_metrics = []
        self.strategy_decisions = []
        self.dse_events = []
        self._stats_cache = {}
        self._last_update = None
    
    def handle_event(self, event: OptimizationEvent) -> None:
        if event.event_type == 'parameter_change':
            self.parameter_changes.append(event)
        elif event.event_type == 'performance_metric':  
            self.performance_metrics.append(event)
        elif event.event_type == 'strategy_decision':
            self.strategy_decisions.append(event)
        elif event.event_type == 'dse_event':
            self.dse_events.append(event)
        
        # Invalidate cache
        self._stats_cache.clear()
        self._last_update = datetime.now()
    
    def get_parameter_statistics(self) -> Dict[str, Any]:
        """Get basic parameter statistics (extensible to full analysis)."""
        if 'parameters' not in self._stats_cache:
            param_counts = defaultdict(int)
            param_values = defaultdict(list)
            
            for event in self.parameter_changes:
                param_name = event.data.get('parameter', 'unknown')
                param_counts[param_name] += 1
                param_values[param_name].append(event.data.get('new_value'))
            
            self._stats_cache['parameters'] = {
                'total_changes': len(self.parameter_changes),
                'unique_parameters': len(param_counts),
                'parameter_counts': dict(param_counts),
                'parameter_values': dict(param_values),
                'recent_changes': len([e for e in self.parameter_changes[-10:]])
            }
        
        return self._stats_cache['parameters']
    
    def get_performance_statistics(self) -> Dict[str, Any]:
        """Get performance metric statistics."""
        if 'performance' not in self._stats_cache:
            metric_data = defaultdict(list)
            
            for event in self.performance_metrics:
                metric_name = event.data.get('metric', 'unknown')
                metric_value = event.data.get('value', 0.0)
                metric_data[metric_name].append(metric_value)
            
            stats = {}
            for metric, values in metric_data.items():
                if values:
                    stats[metric] = {
                        'count': len(values),
                        'min': min(values),
                        'max': max(values),
                        'avg': sum(values) / len(values),
                        'latest': values[-1]
                    }
            
            self._stats_cache['performance'] = {
                'total_metrics': len(self.performance_metrics),
                'unique_metrics': len(metric_data),
                'metric_statistics': stats
            }
        
        return self._stats_cache['performance']


class SimpleStrategyTracker(EventHandler):
    """Example: Simple strategy tracking (extensible to ML analysis)."""
    
    def __init__(self):
        self.strategy_decisions = []
        self.strategy_performance = defaultdict(list)
    
    def handle_event(self, event: OptimizationEvent) -> None:
        if event.event_type == 'strategy_decision':
            self.strategy_decisions.append(event)
        elif event.event_type == 'performance_metric':
            # Associate performance with recent strategy decisions
            recent_strategy = self._get_recent_strategy()
            if recent_strategy:
                metric_value = event.data.get('value', 0.0)
                self.strategy_performance[recent_strategy].append(metric_value)
    
    def _get_recent_strategy(self) -> str:
        """Get most recent strategy decision."""
        if self.strategy_decisions:
            return self.strategy_decisions[-1].data.get('strategy', 'unknown')
        return 'unknown'
    
    def get_strategy_usage(self) -> Dict[str, Any]:
        """Get basic strategy usage (extensible to effectiveness analysis)."""
        strategies = [e.data.get('strategy', 'unknown') for e in self.strategy_decisions]
        strategy_counts = defaultdict(int)
        for strategy in strategies:
            strategy_counts[strategy] += 1
        
        return {
            'total_decisions': len(self.strategy_decisions),
            'unique_strategies': len(strategy_counts),
            'strategy_counts': dict(strategy_counts),
            'strategy_performance': dict(self.strategy_performance)
        }


class DSEProgressTracker(EventHandler):
    """Example: DSE progress tracking."""
    
    def __init__(self):
        self.dse_events = []
        self.dse_stages = defaultdict(list)
        self.start_time = None
        self.end_time = None
    
    def handle_event(self, event: OptimizationEvent) -> None:
        if event.event_type == 'dse_event':
            self.dse_events.append(event)
            stage = event.data.get('stage', 'unknown')
            self.dse_stages[stage].append(event)
            
            if stage == 'start' and not self.start_time:
                self.start_time = event.timestamp
            elif stage in ['complete', 'failed']:
                self.end_time = event.timestamp
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get DSE progress summary."""
        duration = None
        if self.start_time and self.end_time:
            duration = (self.end_time - self.start_time).total_seconds()
        
        return {
            'total_events': len(self.dse_events),
            'stages_seen': list(self.dse_stages.keys()),
            'events_per_stage': {k: len(v) for k, v in self.dse_stages.items()},
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'duration_seconds': duration
        }


class ExamplePlugin(HooksPlugin):
    """Example plugin showing extension pattern."""
    
    def __init__(self):
        self.statistics_handler = SimpleStatisticsHandler()
        self.strategy_handler = SimpleStrategyTracker()
        self.progress_handler = DSEProgressTracker()
        self._installed = False
    
    def install(self) -> None:
        """Install plugin handlers."""
        if self._installed:
            logger.warning("Plugin already installed")
            return
        
        from .. import register_global_handler
        register_global_handler(self.statistics_handler)
        register_global_handler(self.strategy_handler)
        register_global_handler(self.progress_handler)
        
        self._installed = True
        logger.info("Example plugin installed successfully")
    
    def uninstall(self) -> None:
        """Uninstall plugin handlers."""
        if not self._installed:
            logger.warning("Plugin not installed")
            return
        
        # Note: In a full implementation, we'd remove handlers from registry
        # For now, just mark as uninstalled
        self._installed = False
        logger.info("Example plugin uninstalled")
    
    def get_handlers(self) -> List[EventHandler]:
        """Get plugin handlers."""
        return [
            self.statistics_handler, 
            self.strategy_handler,
            self.progress_handler
        ]
    
    def get_name(self) -> str:
        """Get plugin name."""
        return "ExamplePlugin"
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all handlers."""
        return {
            'parameters': self.statistics_handler.get_parameter_statistics(),
            'performance': self.statistics_handler.get_performance_statistics(),
            'strategies': self.strategy_handler.get_strategy_usage(),
            'dse_progress': self.progress_handler.get_progress_summary()
        }


# Future academic plugins (commented examples)
"""
class MLAnalysisPlugin(HooksPlugin):
    '''Future: ML-based strategy analysis as optional plugin.'''
    
    def get_handlers(self) -> List[EventHandler]:
        return [
            StrategyEffectivenessHandler(),
            ParameterSensitivityHandler(), 
            ProblemClassificationHandler()
        ]

class StatisticsPlugin(HooksPlugin):
    '''Future: Statistical analysis as optional plugin.'''
    
    def get_handlers(self) -> List[EventHandler]:
        return [
            CorrelationAnalysisHandler(),
            SignificanceTestingHandler(),
            SensitivityAnalysisHandler()
        ]

class DatabasePlugin(HooksPlugin):
    '''Future: Persistent storage as optional plugin.'''
    
    def get_handlers(self) -> List[EventHandler]:
        return [
            DatabaseStorageHandler(),
            EventQueryHandler(),
            AnalyticsHandler()
        ]
"""


__all__ = [
    'SimpleStatisticsHandler',
    'SimpleStrategyTracker', 
    'DSEProgressTracker',
    'ExamplePlugin'
]