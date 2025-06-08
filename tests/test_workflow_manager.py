"""
Test suite for WorkflowManager

Tests workflow coordination and management capabilities
using existing components with predefined workflow types.
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import time

# Import components to test
try:
    from brainsmith.core.workflow import WorkflowManager, WorkflowType, WorkflowStatus
    from brainsmith.core.design_space_orchestrator import DesignSpaceOrchestrator
except ImportError as e:
    print(f"Warning: Could not import workflow components: {e}")
    WorkflowManager = None
    WorkflowType = None
    WorkflowStatus = None
    DesignSpaceOrchestrator = None


class MockOrchestrator:
    """Mock orchestrator for testing."""
    
    def __init__(self):
        self.blueprint = Mock()
        self.blueprint.name = "test_blueprint"
        self.orchestration_results = {}
    
    def orchestrate_exploration(self, exit_point):
        """Mock orchestration that returns predictable results."""
        mock_result = Mock()
        mock_result.analysis = {
            'exit_point': exit_point,
            'method': f'existing_tools_for_{exit_point}',
            'components_source': 'existing_only'
        }
        
        # Simulate different execution times
        time.sleep(0.01)  # Small delay to simulate work
        
        self.orchestration_results[exit_point] = mock_result
        return mock_result


class TestWorkflowType(unittest.TestCase):
    """Test cases for WorkflowType enum."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WorkflowType is None:
            self.skipTest("WorkflowType not available")
    
    def test_workflow_type_values(self):
        """Test that workflow type enum has expected values."""
        self.assertEqual(WorkflowType.FAST.value, "fast")
        self.assertEqual(WorkflowType.STANDARD.value, "standard")
        self.assertEqual(WorkflowType.COMPREHENSIVE.value, "comprehensive")
    
    def test_workflow_type_creation(self):
        """Test creating workflow types from strings."""
        self.assertEqual(WorkflowType("fast"), WorkflowType.FAST)
        self.assertEqual(WorkflowType("standard"), WorkflowType.STANDARD)
        self.assertEqual(WorkflowType("comprehensive"), WorkflowType.COMPREHENSIVE)
        
        # Test invalid workflow type
        with self.assertRaises(ValueError):
            WorkflowType("invalid_type")


class TestWorkflowStatus(unittest.TestCase):
    """Test cases for WorkflowStatus enum."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WorkflowStatus is None:
            self.skipTest("WorkflowStatus not available")
    
    def test_workflow_status_values(self):
        """Test that workflow status enum has expected values."""
        self.assertEqual(WorkflowStatus.PENDING.value, "pending")
        self.assertEqual(WorkflowStatus.RUNNING.value, "running")
        self.assertEqual(WorkflowStatus.COMPLETED.value, "completed")
        self.assertEqual(WorkflowStatus.FAILED.value, "failed")


class TestWorkflowManager(unittest.TestCase):
    """Test cases for WorkflowManager."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WorkflowManager is None:
            self.skipTest("WorkflowManager not available")
        
        self.mock_orchestrator = MockOrchestrator()
        self.workflow_manager = WorkflowManager(self.mock_orchestrator)
    
    def test_workflow_manager_initialization(self):
        """Test that workflow manager initializes correctly."""
        self.assertIsNotNone(self.workflow_manager)
        self.assertEqual(self.workflow_manager.orchestrator, self.mock_orchestrator)
        self.assertIsInstance(self.workflow_manager.workflow_history, list)
        self.assertEqual(len(self.workflow_manager.workflow_history), 0)
        self.assertIsNone(self.workflow_manager.current_workflow)
    
    def test_fast_workflow_execution(self):
        """Test execution of fast workflow (roofline analysis)."""
        result = self.workflow_manager.execute_existing_workflow("fast")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'fast')
        self.assertIn('result', result)
        self.assertIn('metadata', result)
        
        # Check workflow result details
        workflow_result = result['result']
        self.assertEqual(workflow_result['workflow_type'], 'fast')
        self.assertEqual(workflow_result['exit_point'], 'roofline')
        self.assertEqual(workflow_result['analysis_type'], 'analytical_bounds')
        self.assertIn('dse_result', workflow_result)
        self.assertEqual(workflow_result['libraries_used'], ['analysis'])
        
        # Check metadata
        metadata = result['metadata']
        self.assertEqual(metadata['type'], 'fast')
        self.assertEqual(metadata['status'], 'completed')
        self.assertIn('duration', metadata)
    
    def test_standard_workflow_execution(self):
        """Test execution of standard workflow (dataflow analysis)."""
        result = self.workflow_manager.execute_existing_workflow("standard")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'standard')
        
        # Check workflow result details
        workflow_result = result['result']
        self.assertEqual(workflow_result['workflow_type'], 'standard')
        self.assertEqual(workflow_result['exit_point'], 'dataflow_analysis')
        self.assertEqual(workflow_result['analysis_type'], 'dataflow_estimation')
        self.assertIn('dse_result', workflow_result)
        self.assertEqual(workflow_result['libraries_used'], ['transforms', 'kernels', 'analysis'])
    
    def test_comprehensive_workflow_execution(self):
        """Test execution of comprehensive workflow (full generation)."""
        result = self.workflow_manager.execute_existing_workflow("comprehensive")
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'comprehensive')
        
        # Check workflow result details
        workflow_result = result['result']
        self.assertEqual(workflow_result['workflow_type'], 'comprehensive')
        self.assertEqual(workflow_result['exit_point'], 'dataflow_generation')
        self.assertEqual(workflow_result['analysis_type'], 'complete_generation')
        self.assertIn('dse_result', workflow_result)
        self.assertEqual(workflow_result['libraries_used'], ['transforms', 'kernels', 'hw_optim', 'analysis'])
        self.assertIn('output_artifacts', workflow_result)
    
    def test_invalid_workflow_type(self):
        """Test that invalid workflow types raise ValueError."""
        with self.assertRaises(ValueError) as context:
            self.workflow_manager.execute_existing_workflow("invalid_workflow")
        
        self.assertIn("Invalid workflow type", str(context.exception))
    
    def test_workflow_history_tracking(self):
        """Test that workflow history is tracked correctly."""
        initial_count = len(self.workflow_manager.get_workflow_history())
        
        # Execute multiple workflows
        workflows = ["fast", "standard", "comprehensive"]
        for workflow_type in workflows:
            self.workflow_manager.execute_existing_workflow(workflow_type)
        
        # Check history was updated
        history = self.workflow_manager.get_workflow_history()
        self.assertEqual(len(history), initial_count + len(workflows))
        
        # Check history entries
        for i, workflow_type in enumerate(workflows):
            entry = history[initial_count + i]
            self.assertEqual(entry['type'], workflow_type)
            self.assertEqual(entry['status'], 'completed')
            self.assertIn('duration', entry)
            self.assertIn('result_summary', entry)
    
    def test_current_workflow_status(self):
        """Test current workflow status tracking."""
        # No current workflow initially
        status = self.workflow_manager.get_current_workflow_status()
        self.assertIsNone(status)
        
        # Mock a long-running workflow to test status
        with patch.object(self.mock_orchestrator, 'orchestrate_exploration') as mock_orchestrate:
            def slow_orchestrate(exit_point):
                time.sleep(0.1)  # Simulate some work
                mock_result = Mock()
                mock_result.analysis = {'exit_point': exit_point, 'method': 'test'}
                return mock_result
            
            mock_orchestrate.side_effect = slow_orchestrate
            
            # Start workflow in a way that allows status checking
            # This is simplified for testing - in reality would need threading
            result = self.workflow_manager.execute_existing_workflow("standard")
            
            # Workflow should be completed now
            status = self.workflow_manager.get_current_workflow_status()
            self.assertIsNone(status)  # Should be None after completion
    
    def test_workflow_statistics(self):
        """Test workflow execution statistics."""
        # Execute some workflows
        workflows = ["fast", "standard", "fast", "comprehensive", "standard"]
        for workflow_type in workflows:
            self.workflow_manager.execute_existing_workflow(workflow_type)
        
        stats = self.workflow_manager.get_workflow_statistics()
        
        self.assertIsInstance(stats, dict)
        self.assertEqual(stats['total_workflows'], len(workflows))
        
        # Check by type counts
        expected_counts = {"fast": 2, "standard": 2, "comprehensive": 1}
        self.assertEqual(stats['by_type'], expected_counts)
        
        # Check by status (all should be completed)
        self.assertEqual(stats['by_status']['completed'], len(workflows))
        
        # Check success rate
        self.assertEqual(stats['success_rate'], 1.0)
        
        # Check average duration
        self.assertGreater(stats['average_duration'], 0)
    
    def test_workflow_error_handling(self):
        """Test workflow error handling."""
        # Mock orchestrator that raises exception
        error_orchestrator = Mock()
        error_orchestrator.blueprint = Mock()
        error_orchestrator.blueprint.name = "error_test"
        error_orchestrator.orchestrate_exploration.side_effect = Exception("Test error")
        
        error_workflow_manager = WorkflowManager(error_orchestrator)
        
        # Execute workflow that will fail
        result = error_workflow_manager.execute_existing_workflow("fast")
        
        # Should handle error gracefully
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'fast')
        self.assertEqual(result['result']['status'], 'failed')
        self.assertIn('error', result['result'])
        
        # Check error was recorded in history
        history = error_workflow_manager.get_workflow_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]['status'], 'failed')
        self.assertIn('error', history[0])
    
    def test_time_estimation(self):
        """Test workflow time estimation."""
        # Test time estimation for different workflow types
        test_cases = [
            ("fast", 15.0, "~15 seconds"),
            ("standard", 60.0, "~1 minutes"),
            ("comprehensive", 300.0, "~5 minutes")
        ]
        
        for workflow_type, elapsed_time, expected_format in test_cases:
            remaining = self.workflow_manager._estimate_remaining_time(workflow_type, elapsed_time)
            self.assertIsInstance(remaining, str)
            # Just check that it returns a reasonable format
            self.assertTrue("second" in remaining or "minute" in remaining)
    
    def test_workflow_cancellation(self):
        """Test workflow cancellation functionality."""
        # No current workflow to cancel
        result = self.workflow_manager.cancel_current_workflow()
        self.assertFalse(result)
        
        # This test is simplified since actual cancellation would require
        # more complex threading/async implementation
    
    def test_create_workflow_from_blueprint(self):
        """Test creating workflow from blueprint configuration."""
        # Mock blueprint with workflow configuration
        mock_blueprint = Mock()
        mock_blueprint.name = "test_blueprint"
        mock_blueprint.get_workflow_config = Mock(return_value={
            'type': 'comprehensive',
            'custom_param': 'value'
        })
        
        result = self.workflow_manager.create_workflow_from_blueprint(mock_blueprint)
        
        self.assertIsInstance(result, dict)
        self.assertEqual(result['workflow'], 'comprehensive')
        
        # Verify blueprint method was called
        mock_blueprint.get_workflow_config.assert_called_once()
    
    def test_workflow_with_custom_config(self):
        """Test workflow execution with custom configuration."""
        custom_config = {
            'param1': 'value1',
            'param2': 'value2',
            'output_dir': '/tmp/test'
        }
        
        result = self.workflow_manager.execute_existing_workflow(
            "standard", 
            **custom_config
        )
        
        # Check custom config was included
        metadata = result['metadata']
        self.assertEqual(metadata['config'], custom_config)


class TestWorkflowManagerIntegration(unittest.TestCase):
    """Integration tests for WorkflowManager with realistic scenarios."""
    
    def setUp(self):
        """Set up test fixtures."""
        if WorkflowManager is None:
            self.skipTest("WorkflowManager not available")
    
    def test_workflow_progression(self):
        """Test progression through different workflow types."""
        orchestrator = MockOrchestrator()
        workflow_manager = WorkflowManager(orchestrator)
        
        # Simulate user starting with fast analysis and progressing
        workflows = ["fast", "standard", "comprehensive"]
        results = []
        
        for workflow_type in workflows:
            result = workflow_manager.execute_existing_workflow(workflow_type)
            results.append(result)
        
        # Verify progression
        self.assertEqual(len(results), 3)
        
        # Check that each workflow used the appropriate libraries
        library_counts = [
            len(results[0]['result']['libraries_used']),  # fast: analysis only
            len(results[1]['result']['libraries_used']),  # standard: +transforms, kernels
            len(results[2]['result']['libraries_used'])   # comprehensive: +hw_optim
        ]
        
        # Should have increasing library usage
        self.assertLessEqual(library_counts[0], library_counts[1])
        self.assertLessEqual(library_counts[1], library_counts[2])
    
    def test_workflow_performance_tracking(self):
        """Test performance tracking across multiple workflows."""
        orchestrator = MockOrchestrator()
        workflow_manager = WorkflowManager(orchestrator)
        
        # Execute multiple workflows and track performance
        num_workflows = 5
        for i in range(num_workflows):
            workflow_type = ["fast", "standard", "comprehensive"][i % 3]
            workflow_manager.execute_existing_workflow(workflow_type)
        
        # Get performance statistics
        stats = workflow_manager.get_workflow_statistics()
        
        self.assertEqual(stats['total_workflows'], num_workflows)
        self.assertGreater(stats['average_duration'], 0)
        self.assertEqual(stats['success_rate'], 1.0)  # All should succeed
        
        # Check that different workflow types have different characteristics
        history = workflow_manager.get_workflow_history()
        fast_durations = [h['duration'] for h in history if h['type'] == 'fast']
        comprehensive_durations = [h['duration'] for h in history if h['type'] == 'comprehensive']
        
        # Fast workflows should generally be quicker (though this is mocked)
        if fast_durations and comprehensive_durations:
            self.assertIsInstance(fast_durations[0], float)
            self.assertIsInstance(comprehensive_durations[0], float)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)