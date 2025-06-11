#!/usr/bin/env python3
"""
Comprehensive Test Suite for Hooks Simplification

Tests both the simple core and extension capabilities to validate
the 90% complexity reduction while maintaining strong extension points.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any, List

# Import hooks components to test
try:
    from brainsmith.hooks import (
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
        create_custom_event_type,
        OptimizationEvent,
        EventHandler,
        SimpleMetric,
        ParameterChange,
        EventTypes
    )
    
    from brainsmith.hooks.plugins import (
        install_plugin,
        uninstall_plugin,
        get_plugin,
        list_plugins,
        PluginManager
    )
    
    from brainsmith.hooks.plugins.examples import (
        SimpleStatisticsHandler,
        SimpleStrategyTracker,
        DSEProgressTracker,
        ExamplePlugin
    )
    
    HOOKS_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import hooks: {e}")
    HOOKS_AVAILABLE = False


class TestSimpleCoreHooks(unittest.TestCase):
    """Test cases for the simple core hooks functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
        
        # Clear event history before each test
        clear_event_history()
    
    def test_basic_event_logging(self):
        """Test basic optimization event logging."""
        # Log a simple event
        log_optimization_event('test_event', {'key': 'value'})
        
        # Verify event was recorded
        recent_events = get_recent_events(1)
        self.assertEqual(len(recent_events), 1)
        
        event = recent_events[0]
        self.assertEqual(event.event_type, 'test_event')
        self.assertEqual(event.data['key'], 'value')
        self.assertIsInstance(event.timestamp, datetime)
    
    def test_parameter_change_logging(self):
        """Test parameter change event logging."""
        # Log parameter change
        log_parameter_change('learning_rate', 0.01, 0.005)
        
        # Verify event was recorded
        param_events = get_events_by_type('parameter_change')
        self.assertEqual(len(param_events), 1)
        
        event = param_events[0]
        self.assertEqual(event.data['parameter'], 'learning_rate')
        self.assertEqual(event.data['old_value'], 0.01)
        self.assertEqual(event.data['new_value'], 0.005)
        self.assertIsNotNone(event.data.get('change_magnitude'))
    
    def test_performance_metric_logging(self):
        """Test performance metric logging."""
        # Log performance metric
        log_performance_metric('throughput', 150.5, {'units': 'ops/sec'})
        
        # Verify event was recorded
        metric_events = get_events_by_type('performance_metric')
        self.assertEqual(len(metric_events), 1)
        
        event = metric_events[0]
        self.assertEqual(event.data['metric'], 'throughput')
        self.assertEqual(event.data['value'], 150.5)
        self.assertEqual(event.data['context']['units'], 'ops/sec')
    
    def test_strategy_decision_logging(self):
        """Test strategy decision logging."""
        # Log strategy decision
        log_strategy_decision('bayesian', 'Higher efficiency expected', ['random', 'grid'])
        
        # Verify event was recorded
        strategy_events = get_events_by_type('strategy_decision')
        self.assertEqual(len(strategy_events), 1)
        
        event = strategy_events[0]
        self.assertEqual(event.data['strategy'], 'bayesian')
        self.assertEqual(event.data['rationale'], 'Higher efficiency expected')
        self.assertIn('random', event.data['alternatives'])
    
    def test_dse_event_logging(self):
        """Test DSE event logging."""
        # Log DSE event
        log_dse_event('exploration_start', {'design_points': 100, 'strategy': 'adaptive'})
        
        # Verify event was recorded
        dse_events = get_events_by_type('dse_event')
        self.assertEqual(len(dse_events), 1)
        
        event = dse_events[0]
        self.assertEqual(event.data['stage'], 'exploration_start')
        self.assertEqual(event.data['design_points'], 100)
        self.assertEqual(event.data['strategy'], 'adaptive')
    
    def test_event_statistics(self):
        """Test event system statistics."""
        # Clear any previous events
        clear_event_history()
        
        # Get initial stats to get baseline
        initial_stats = get_event_stats()
        initial_count = initial_stats['total_events']
        
        # Log multiple events
        log_optimization_event('test1', {})
        log_parameter_change('param1', 1, 2)
        log_performance_metric('metric1', 100.0)
        
        # Get statistics
        stats = get_event_stats()
        
        # Verify statistics (account for any events that might have been logged during test setup)
        self.assertEqual(stats['total_events'], initial_count + 3)
    
    def test_event_history_management(self):
        """Test event history retrieval and clearing."""
        # Log multiple events
        for i in range(15):
            log_optimization_event(f'test_{i}', {'index': i})
        
        # Test recent events (default 10)
        recent = get_recent_events()
        self.assertEqual(len(recent), 10)
        self.assertEqual(recent[-1].data['index'], 14)  # Most recent
        
        # Test specific count
        recent_5 = get_recent_events(5)
        self.assertEqual(len(recent_5), 5)
        
        # Test clearing history
        clear_event_history()
        recent_after_clear = get_recent_events()
        self.assertEqual(len(recent_after_clear), 0)


class TestCustomEventHandlers(unittest.TestCase):
    """Test cases for custom event handlers and extension points."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
        
        clear_event_history()
    
    def test_custom_event_handler(self):
        """Test custom event handler registration and functionality."""
        
        class TestHandler(EventHandler):
            def __init__(self):
                self.handled_events = []
            
            def handle_event(self, event: OptimizationEvent) -> None:
                self.handled_events.append(event)
        
        # Register custom handler
        handler = TestHandler()
        register_event_handler('test_event', handler)
        
        # Log events
        log_optimization_event('test_event', {'data': 'test'})
        log_optimization_event('other_event', {'data': 'other'})
        
        # Verify handler received only relevant events
        self.assertEqual(len(handler.handled_events), 1)
        self.assertEqual(handler.handled_events[0].event_type, 'test_event')
    
    def test_global_event_handler(self):
        """Test global event handler registration."""
        
        class GlobalTestHandler(EventHandler):
            def __init__(self):
                self.all_events = []
            
            def handle_event(self, event: OptimizationEvent) -> None:
                self.all_events.append(event)
        
        # Register global handler
        handler = GlobalTestHandler()
        register_global_handler(handler)
        
        # Log different types of events
        log_optimization_event('event1', {})
        log_parameter_change('param', 1, 2)
        log_performance_metric('metric', 100.0)
        
        # Verify handler received all events
        self.assertEqual(len(handler.all_events), 3)
        event_types = [e.event_type for e in handler.all_events]
        self.assertIn('event1', event_types)
        self.assertIn('parameter_change', event_types)
        self.assertIn('performance_metric', event_types)
    
    def test_custom_event_types(self):
        """Test custom event type creation."""
        # Create custom event type
        create_custom_event_type('model_validation')
        
        # Use custom event type
        log_optimization_event('model_validation', {'accuracy': 0.95, 'loss': 0.05})
        
        # Verify custom event was logged
        custom_events = get_events_by_type('model_validation')
        self.assertEqual(len(custom_events), 1)
        self.assertEqual(custom_events[0].data['accuracy'], 0.95)


class TestPluginSystem(unittest.TestCase):
    """Test cases for the plugin system functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
        
        clear_event_history()
        
        # Uninstall any existing plugins
        for plugin_name in list_plugins():
            uninstall_plugin(plugin_name)
    
    def test_plugin_manager_basic_operations(self):
        """Test basic plugin manager operations."""
        manager = PluginManager()
        
        # Test empty state
        self.assertEqual(len(manager.list_plugins()), 0)
        self.assertEqual(len(manager.get_all_handlers()), 0)
        
        # Test plugin installation
        plugin = ExamplePlugin()
        manager.install_plugin('test_plugin', plugin)
        
        self.assertEqual(len(manager.list_plugins()), 1)
        self.assertIn('test_plugin', manager.list_plugins())
        self.assertIsNotNone(manager.get_plugin('test_plugin'))
        
        # Test plugin uninstallation
        manager.uninstall_plugin('test_plugin')
        self.assertEqual(len(manager.list_plugins()), 0)
    
    def test_example_plugin_functionality(self):
        """Test the example plugin functionality."""
        plugin = ExamplePlugin()
        install_plugin('example', plugin)
        
        # Generate some events to test handlers
        log_parameter_change('pe_count', 4, 8)
        log_performance_metric('throughput', 120.5)
        log_strategy_decision('bayesian', 'Better convergence')
        log_dse_event('start', {'design_space_size': 100})
        log_dse_event('complete', {'solutions_found': 25})
        
        # Test statistics gathering
        stats = plugin.get_statistics()
        
        # Verify parameter statistics
        param_stats = stats['parameters']
        self.assertEqual(param_stats['total_changes'], 1)
        self.assertEqual(param_stats['unique_parameters'], 1)
        self.assertIn('pe_count', param_stats['parameter_counts'])
        
        # Verify performance statistics
        perf_stats = stats['performance']
        self.assertEqual(perf_stats['total_metrics'], 1)
        self.assertIn('throughput', perf_stats['metric_statistics'])
        
        # Verify strategy statistics
        strategy_stats = stats['strategies']
        self.assertEqual(strategy_stats['total_decisions'], 1)
        self.assertIn('bayesian', strategy_stats['strategy_counts'])
        
        # Verify DSE progress
        dse_stats = stats['dse_progress']
        self.assertEqual(dse_stats['total_events'], 2)
        self.assertIn('start', dse_stats['stages_seen'])
        self.assertIn('complete', dse_stats['stages_seen'])
        
        # Cleanup
        uninstall_plugin('example')
    
    def test_statistics_handler_detailed(self):
        """Test detailed statistics handler functionality."""
        handler = SimpleStatisticsHandler()
        register_global_handler(handler)
        
        # Generate varied parameter changes
        log_parameter_change('learning_rate', 0.01, 0.005)
        log_parameter_change('batch_size', 32, 64)
        log_parameter_change('learning_rate', 0.005, 0.001)
        
        # Generate performance metrics
        log_performance_metric('accuracy', 0.85)
        log_performance_metric('loss', 0.15)
        log_performance_metric('accuracy', 0.90)
        
        # Test parameter statistics
        param_stats = handler.get_parameter_statistics()
        self.assertEqual(param_stats['total_changes'], 3)
        self.assertEqual(param_stats['unique_parameters'], 2)
        self.assertEqual(param_stats['parameter_counts']['learning_rate'], 2)
        self.assertEqual(param_stats['parameter_counts']['batch_size'], 1)
        
        # Test performance statistics
        perf_stats = handler.get_performance_statistics()
        self.assertEqual(perf_stats['total_metrics'], 3)
        self.assertEqual(perf_stats['unique_metrics'], 2)
        
        acc_stats = perf_stats['metric_statistics']['accuracy']
        self.assertEqual(acc_stats['count'], 2)
        self.assertEqual(acc_stats['min'], 0.85)
        self.assertEqual(acc_stats['max'], 0.90)
        self.assertEqual(acc_stats['avg'], 0.875)
    
    def test_strategy_tracker_detailed(self):
        """Test detailed strategy tracker functionality."""
        tracker = SimpleStrategyTracker()
        register_global_handler(tracker)
        
        # Log strategy decisions and performance
        log_strategy_decision('random', 'Initial exploration')
        log_performance_metric('throughput', 100.0)
        log_performance_metric('latency', 10.0)
        
        log_strategy_decision('bayesian', 'Focused optimization')
        log_performance_metric('throughput', 150.0)
        log_performance_metric('latency', 8.0)
        
        log_strategy_decision('random', 'Diversification')
        log_performance_metric('throughput', 90.0)
        
        # Test strategy usage
        usage = tracker.get_strategy_usage()
        self.assertEqual(usage['total_decisions'], 3)
        self.assertEqual(usage['unique_strategies'], 2)
        self.assertEqual(usage['strategy_counts']['random'], 2)
        self.assertEqual(usage['strategy_counts']['bayesian'], 1)
        
        # Test strategy performance association
        strategy_perf = usage['strategy_performance']
        self.assertIn('random', strategy_perf)
        self.assertIn('bayesian', strategy_perf)


class TestCoreAPIIntegration(unittest.TestCase):
    """Test cases for core API integration with hooks."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
        
        clear_event_history()
    
    @patch('brainsmith.core.api._validate_inputs')
    @patch('brainsmith.core.api._load_and_validate_blueprint')
    @patch('brainsmith.core.api._setup_dse_configuration')
    @patch('brainsmith.core.api._run_full_dse')
    @patch('brainsmith.core.api._assemble_results')
    def test_forge_hooks_integration(self, mock_assemble, mock_dse, mock_config, mock_blueprint, mock_validate):
        """Test that forge function properly logs hooks events."""
        from brainsmith.core.api import forge
        
        # Clear events before test
        clear_event_history()
        
        # Mock the dependencies
        mock_validate.return_value = None
        mock_blueprint.return_value = Mock()
        mock_config.return_value = {}
        mock_dse.return_value = Mock()
        mock_assemble.return_value = {'test': 'result'}
        
        # Call forge
        result = forge('test_model.onnx', 'test_blueprint.yaml')
        
        # Verify optimization events were logged
        events = get_recent_events(10)
        
        # Check that we have at least some events
        self.assertGreater(len(events), 0)
        
        # Should have strategy decision, dse events, and completion events
        event_types = [e.event_type for e in events]
        self.assertIn('strategy_decision', event_types)
        self.assertIn('dse_event', event_types)
        self.assertIn('optimization_end', event_types)
        
        # Verify completion event
        end_events = [e for e in events if e.event_type == 'optimization_end']
        self.assertEqual(len(end_events), 1)
        end_event = end_events[0]
        self.assertTrue(end_event.data['success'])


class TestDataTypes(unittest.TestCase):
    """Test cases for hooks data types and helper functions."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
    
    def test_optimization_event_serialization(self):
        """Test OptimizationEvent serialization."""
        # Create event
        event = OptimizationEvent(
            timestamp=datetime.now(),
            event_type='test_event',
            data={'key': 'value'},
            metadata={'meta': 'data'}
        )
        
        # Test to_dict
        event_dict = event.to_dict()
        self.assertEqual(event_dict['event_type'], 'test_event')
        self.assertEqual(event_dict['data']['key'], 'value')
        self.assertEqual(event_dict['metadata']['meta'], 'data')
        self.assertIsInstance(event_dict['timestamp'], str)
        
        # Test from_dict
        reconstructed = OptimizationEvent.from_dict(event_dict)
        self.assertEqual(reconstructed.event_type, 'test_event')
        self.assertEqual(reconstructed.data['key'], 'value')
        self.assertEqual(reconstructed.metadata['meta'], 'data')
    
    def test_simple_metric_from_event(self):
        """Test SimpleMetric creation from event."""
        # Log performance metric
        log_performance_metric('throughput', 150.0, {'units': 'ops/sec'})
        
        # Get the event
        events = get_events_by_type('performance_metric')
        event = events[0]
        
        # Create SimpleMetric from event
        metric = SimpleMetric.from_event(event)
        self.assertIsNotNone(metric)
        self.assertEqual(metric.name, 'throughput')
        self.assertEqual(metric.value, 150.0)
        self.assertEqual(metric.tags['units'], 'ops/sec')
    
    def test_parameter_change_from_event(self):
        """Test ParameterChange creation from event."""
        # Log parameter change
        log_parameter_change('learning_rate', 0.01, 0.005)
        
        # Get the event
        events = get_events_by_type('parameter_change')
        event = events[0]
        
        # Create ParameterChange from event
        param_change = ParameterChange.from_event(event)
        self.assertIsNotNone(param_change)
        self.assertEqual(param_change.parameter, 'learning_rate')
        self.assertEqual(param_change.old_value, 0.01)
        self.assertEqual(param_change.new_value, 0.005)
        self.assertIsNotNone(param_change.change_magnitude)


class TestPerformanceAndMemory(unittest.TestCase):
    """Test cases for performance and memory efficiency."""
    
    def setUp(self):
        """Set up test fixtures."""
        if not HOOKS_AVAILABLE:
            self.skipTest("Hooks not available")
        
        clear_event_history()
    
    def test_memory_limit_enforcement(self):
        """Test that memory handler enforces event limits."""
        # Log more events than the default limit (1000)
        for i in range(1200):
            log_optimization_event(f'test_{i}', {'index': i})
        
        # Verify that only the most recent 1000 events are kept
        recent_events = get_recent_events(1200)
        self.assertLessEqual(len(recent_events), 1000)
        
        # Verify the most recent event is still there
        if recent_events:
            latest_event = recent_events[-1]
            self.assertEqual(latest_event.data['index'], 1199)
    
    def test_error_handling_in_handlers(self):
        """Test error handling in event handlers."""
        
        class FaultyHandler(EventHandler):
            def handle_event(self, event: OptimizationEvent) -> None:
                raise Exception("Simulated handler error")
        
        # Register faulty handler
        faulty_handler = FaultyHandler()
        register_global_handler(faulty_handler)
        
        # Log event - should not raise exception
        try:
            log_optimization_event('test_event', {})
            # If we get here, error was handled gracefully
            success = True
        except Exception:
            success = False
        
        self.assertTrue(success, "Event logging should handle handler errors gracefully")


if __name__ == '__main__':
    # Run tests with detailed output
    unittest.main(verbosity=2)