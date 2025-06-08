"""
Unit tests for DSE analysis capabilities.

Tests Pareto frontier analysis, statistical analysis, and result processing.
"""

import unittest
import sys
import os
from unittest.mock import Mock
import numpy as np

# Add brainsmith to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from brainsmith.dse.analysis import (
    ParetoAnalyzer, DSEAnalyzer, ParetoPoint, SensitivityAnalysis, ConvergenceAnalysis
)
from brainsmith.dse.interface import DSEObjective, OptimizationObjective
from brainsmith.core.result import BrainsmithResult, DSEResult
from brainsmith.core.design_space import DesignSpace, DesignPoint, ParameterDefinition, ParameterType


class TestParetoAnalyzer(unittest.TestCase):
    """Test Pareto frontier analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.objectives = [
            DSEObjective("metric1", OptimizationObjective.MAXIMIZE),
            DSEObjective("metric2", OptimizationObjective.MAXIMIZE)
        ]
        self.analyzer = ParetoAnalyzer(self.objectives)
    
    def create_mock_result(self, metric1_value, metric2_value):
        """Create a mock result with specified metric values."""
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.metric1 = metric1_value
        result.metrics.metric2 = metric2_value
        return result
    
    def test_single_objective_analyzer(self):
        """Test analyzer behavior with single objective."""
        single_obj = [DSEObjective("metric1", OptimizationObjective.MAXIMIZE)]
        analyzer = ParetoAnalyzer(single_obj)
        
        self.assertFalse(analyzer.is_multi_objective)
    
    def test_multi_objective_analyzer(self):
        """Test analyzer behavior with multiple objectives."""
        self.assertTrue(self.analyzer.is_multi_objective)
    
    def test_dominance_checking(self):
        """Test dominance relationship checking."""
        # Test clear dominance
        a = [10, 8]  # Better on both
        b = [5, 6]   # Worse on both
        self.assertTrue(self.analyzer._dominates(a, b))
        self.assertFalse(self.analyzer._dominates(b, a))
        
        # Test non-dominance (trade-off)
        c = [10, 6]  # Better on first, worse on second
        d = [8, 8]   # Worse on first, better on second
        self.assertFalse(self.analyzer._dominates(c, d))
        self.assertFalse(self.analyzer._dominates(d, c))
        
        # Test equal solutions
        e = [5, 5]
        f = [5, 5]
        self.assertFalse(self.analyzer._dominates(e, f))
    
    def test_pareto_frontier_computation(self):
        """Test Pareto frontier computation with various trade-offs."""
        # Create results with known Pareto relationships
        results = [
            self.create_mock_result(10, 5),   # Point A
            self.create_mock_result(8, 8),    # Point B (non-dominated)
            self.create_mock_result(6, 10),   # Point C (non-dominated)
            self.create_mock_result(5, 6),    # Point D (dominated by B)
            self.create_mock_result(12, 3),   # Point E (non-dominated)
        ]
        
        pareto_points = self.analyzer.compute_pareto_frontier(results)
        
        # Should have 4 non-dominated points (A, B, C, E)
        self.assertEqual(len(pareto_points), 4)
        
        # Check that dominated point D is not included
        pareto_objectives = [p.objective_values for p in pareto_points]
        self.assertNotIn([5, 6], pareto_objectives)
    
    def test_single_objective_pareto(self):
        """Test Pareto frontier for single objective (should return best solutions)."""
        single_obj = [DSEObjective("metric1", OptimizationObjective.MAXIMIZE)]
        analyzer = ParetoAnalyzer(single_obj)
        
        results = [
            self.create_mock_result(10, 0),
            self.create_mock_result(15, 0),  # Best
            self.create_mock_result(8, 0),
            self.create_mock_result(12, 0),
        ]
        
        pareto_points = analyzer.compute_pareto_frontier(results)
        
        # Should return top solutions, sorted by objective value
        self.assertLessEqual(len(pareto_points), 10)  # Max 10 for single objective
        self.assertEqual(pareto_points[0].objective_values[0], 15)  # Best first
    
    def test_crowding_distance_computation(self):
        """Test NSGA-II style crowding distance computation."""
        results = [
            self.create_mock_result(10, 5),
            self.create_mock_result(8, 7),
            self.create_mock_result(6, 9),
            self.create_mock_result(4, 10),
        ]
        
        pareto_points = self.analyzer.compute_pareto_frontier(results)
        
        # All points should be non-dominated in this case
        self.assertEqual(len(pareto_points), 4)
        
        # Boundary points should have infinite crowding distance
        boundary_points = [p for p in pareto_points if p.crowding_distance == float('inf')]
        self.assertEqual(len(boundary_points), 2)  # Two boundary points
        
        # Interior points should have finite positive crowding distance
        interior_points = [p for p in pareto_points if p.crowding_distance != float('inf')]
        for point in interior_points:
            self.assertGreater(point.crowding_distance, 0)
    
    def test_trade_off_analysis(self):
        """Test trade-off analysis between objectives."""
        results = [
            self.create_mock_result(10, 2),
            self.create_mock_result(8, 5),
            self.create_mock_result(6, 7),
            self.create_mock_result(4, 9),
            self.create_mock_result(2, 10),
        ]
        
        pareto_points = self.analyzer.compute_pareto_frontier(results)
        trade_offs = self.analyzer.analyze_trade_offs(pareto_points)
        
        self.assertIsInstance(trade_offs, dict)
        self.assertIn("correlations", trade_offs)
        self.assertIn("objective_ranges", trade_offs)
        self.assertIn("pareto_size", trade_offs)
        
        # Should detect negative correlation between objectives
        correlation = trade_offs["correlations"]["metric1"]["metric2"]
        self.assertLess(correlation, 0)  # Should be negative correlation
    
    def test_hypervolume_2d(self):
        """Test 2D hypervolume computation."""
        # Create a simple 2D Pareto front
        objective_matrix = np.array([
            [4, 1],
            [3, 2], 
            [2, 3],
            [1, 4]
        ])
        
        reference_point = np.array([0, 0])
        hypervolume = self.analyzer._compute_hypervolume(objective_matrix, reference_point)
        
        # Should compute meaningful hypervolume
        self.assertGreater(hypervolume, 0)
        self.assertIsInstance(hypervolume, float)
    
    def test_hypervolume_3d(self):
        """Test 3D hypervolume computation (Monte Carlo approximation)."""
        # Create 3D points
        objective_matrix = np.array([
            [4, 1, 2],
            [3, 2, 3], 
            [2, 3, 4],
            [1, 4, 1]
        ])
        
        reference_point = np.array([0, 0, 0])
        hypervolume = self.analyzer._hypervolume_3d(objective_matrix, reference_point, n_samples=1000)
        
        # Should compute meaningful hypervolume estimate
        self.assertGreater(hypervolume, 0)
        self.assertIsInstance(hypervolume, float)
    
    def test_empty_results_handling(self):
        """Test handling of empty result sets."""
        pareto_points = self.analyzer.compute_pareto_frontier([])
        self.assertEqual(len(pareto_points), 0)
        
        trade_offs = self.analyzer.analyze_trade_offs([])
        self.assertIn("trade_offs", trade_offs)


class TestDSEAnalyzer(unittest.TestCase):
    """Test comprehensive DSE analysis functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.design_space = DesignSpace("test_space")
        
        # Add parameters
        param1 = ParameterDefinition("param1", ParameterType.CONTINUOUS, range_values=[0.0, 1.0])
        param2 = ParameterDefinition("param2", ParameterType.INTEGER, range_values=[1, 10])
        self.design_space.add_parameter(param1)
        self.design_space.add_parameter(param2)
        
        self.objectives = [
            DSEObjective("performance.throughput", OptimizationObjective.MAXIMIZE),
            DSEObjective("hardware.power", OptimizationObjective.MINIMIZE)
        ]
        
        self.analyzer = DSEAnalyzer(self.design_space, self.objectives)
    
    def create_mock_dse_result(self, n_results=10):
        """Create a mock DSE result with specified number of results."""
        results = []
        for i in range(n_results):
            result = Mock(spec=BrainsmithResult)
            result.metrics = Mock()
            result.metrics.performance = Mock()
            result.metrics.performance.throughput = 100 + i * 10
            result.metrics.hardware = Mock()
            result.metrics.hardware.power = 50 - i * 2
            results.append(result)
        
        dse_result = Mock(spec=DSEResult)
        dse_result.results = results
        dse_result.total_time_seconds = 300.0
        dse_result.strategy = "test_strategy"
        dse_result.objectives = ["performance.throughput", "hardware.power"]
        dse_result.design_space = self.design_space
        
        return dse_result
    
    def test_comprehensive_analysis(self):
        """Test comprehensive DSE result analysis."""
        dse_result = self.create_mock_dse_result(20)
        analysis = self.analyzer.analyze_dse_result(dse_result)
        
        # Check main analysis sections
        self.assertIn("summary", analysis)
        self.assertIn("statistical_analysis", analysis)
        self.assertIn("pareto_analysis", analysis)
        
        # Check summary section
        summary = analysis["summary"]
        self.assertEqual(summary["total_evaluations"], 20)
        self.assertEqual(summary["total_time_seconds"], 300.0)
        self.assertEqual(summary["strategy"], "test_strategy")
    
    def test_statistical_analysis(self):
        """Test statistical analysis of DSE results."""
        dse_result = self.create_mock_dse_result(15)
        analysis = self.analyzer._analyze_statistics(dse_result)
        
        # Should have statistics for each objective
        self.assertIn("performance.throughput", analysis)
        self.assertIn("hardware.power", analysis)
        
        # Check statistical measures
        throughput_stats = analysis["performance.throughput"]
        self.assertIn("mean", throughput_stats)
        self.assertIn("std", throughput_stats)
        self.assertIn("min", throughput_stats)
        self.assertIn("max", throughput_stats)
        self.assertIn("median", throughput_stats)
        self.assertIn("q25", throughput_stats)
        self.assertIn("q75", throughput_stats)
        
        # Verify reasonable values
        self.assertGreater(throughput_stats["mean"], 0)
        self.assertGreaterEqual(throughput_stats["max"], throughput_stats["mean"])
        self.assertLessEqual(throughput_stats["min"], throughput_stats["mean"])
    
    def test_convergence_analysis(self):
        """Test convergence analysis functionality."""
        dse_result = self.create_mock_dse_result(25)
        convergence = self.analyzer._analyze_convergence(dse_result)
        
        self.assertIsInstance(convergence, ConvergenceAnalysis)
        self.assertEqual(len(convergence.best_values_over_time), 25)
        self.assertIsInstance(convergence.improvement_rate, float)
        self.assertIsInstance(convergence.convergence_score, float)
        self.assertIsInstance(convergence.estimated_convergence, bool)
        
        # Best values should be monotonically non-decreasing for maximization
        best_values = convergence.best_values_over_time
        for i in range(1, len(best_values)):
            self.assertGreaterEqual(best_values[i], best_values[i-1])
    
    def test_analysis_report_generation(self):
        """Test human-readable analysis report generation."""
        dse_result = self.create_mock_dse_result(12)
        report = self.analyzer.generate_analysis_report(dse_result)
        
        self.assertIsInstance(report, str)
        self.assertIn("DSE Analysis Report", report)
        self.assertIn("Summary", report)
        self.assertIn("Total Evaluations: 12", report)
        self.assertIn("Strategy: test_strategy", report)
    
    def test_analysis_export(self):
        """Test analysis export functionality."""
        import tempfile
        import json
        import os
        
        dse_result = self.create_mock_dse_result(8)
        analysis = self.analyzer.analyze_dse_result(dse_result)
        
        # Export to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            filepath = f.name
        
        try:
            self.analyzer.export_analysis(analysis, filepath)
            
            # Verify file was created and contains valid JSON
            self.assertTrue(os.path.exists(filepath))
            
            with open(filepath, 'r') as f:
                loaded_analysis = json.load(f)
            
            self.assertIsInstance(loaded_analysis, dict)
            self.assertIn("summary", loaded_analysis)
        
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.unlink(filepath)


class TestParetoPoint(unittest.TestCase):
    """Test ParetoPoint data structure."""
    
    def test_pareto_point_creation(self):
        """Test ParetoPoint creation and properties."""
        design_point = DesignPoint()
        result = Mock(spec=BrainsmithResult)
        objective_values = [10.0, 5.0]
        
        pareto_point = ParetoPoint(
            design_point=design_point,
            result=result,
            objective_values=objective_values,
            rank=1,
            crowding_distance=2.5
        )
        
        self.assertEqual(pareto_point.design_point, design_point)
        self.assertEqual(pareto_point.result, result)
        self.assertEqual(pareto_point.objective_values, objective_values)
        self.assertEqual(pareto_point.rank, 1)
        self.assertEqual(pareto_point.crowding_distance, 2.5)


class TestSensitivityAnalysis(unittest.TestCase):
    """Test sensitivity analysis data structures."""
    
    def test_sensitivity_analysis_creation(self):
        """Test SensitivityAnalysis creation."""
        param_importance = {"param1": 0.8, "param2": 0.6}
        param_correlations = {"param1": {"objective1": 0.7}}
        
        sensitivity = SensitivityAnalysis(
            parameter_importance=param_importance,
            parameter_correlations=param_correlations
        )
        
        self.assertEqual(sensitivity.parameter_importance, param_importance)
        self.assertEqual(sensitivity.parameter_correlations, param_correlations)
        self.assertIsNone(sensitivity.sobol_indices)
        self.assertIsNone(sensitivity.morris_effects)


class TestConvergenceAnalysis(unittest.TestCase):
    """Test convergence analysis data structures."""
    
    def test_convergence_analysis_creation(self):
        """Test ConvergenceAnalysis creation."""
        best_values = [10, 15, 18, 20, 20, 20]
        pareto_sizes = [1, 2, 3, 3, 3, 3]
        stagnation_periods = [(3, 5)]
        
        convergence = ConvergenceAnalysis(
            best_values_over_time=best_values,
            pareto_size_over_time=pareto_sizes,
            improvement_rate=2.5,
            stagnation_periods=stagnation_periods,
            convergence_score=0.95,
            estimated_convergence=True
        )
        
        self.assertEqual(convergence.best_values_over_time, best_values)
        self.assertEqual(convergence.pareto_size_over_time, pareto_sizes)
        self.assertEqual(convergence.improvement_rate, 2.5)
        self.assertEqual(convergence.stagnation_periods, stagnation_periods)
        self.assertEqual(convergence.convergence_score, 0.95)
        self.assertTrue(convergence.estimated_convergence)


class TestAnalysisEdgeCases(unittest.TestCase):
    """Test edge cases and error handling in analysis."""
    
    def setUp(self):
        """Set up minimal test environment."""
        self.design_space = DesignSpace("minimal")
        self.objectives = [DSEObjective("metric1", OptimizationObjective.MAXIMIZE)]
        self.analyzer = DSEAnalyzer(self.design_space, self.objectives)
    
    def test_empty_dse_result(self):
        """Test analysis with empty DSE result."""
        dse_result = Mock(spec=DSEResult)
        dse_result.results = []
        dse_result.total_time_seconds = 0.0
        dse_result.strategy = "empty"
        dse_result.objectives = []
        dse_result.design_space = self.design_space
        
        analysis = self.analyzer.analyze_dse_result(dse_result)
        
        # Should handle empty results gracefully
        self.assertIsInstance(analysis, dict)
        self.assertEqual(analysis["summary"]["total_evaluations"], 0)
    
    def test_invalid_objective_evaluation(self):
        """Test handling of invalid objective evaluation."""
        objectives = [DSEObjective("nonexistent.metric", OptimizationObjective.MAXIMIZE)]
        analyzer = ParetoAnalyzer(objectives)
        
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        # Don't set the nonexistent metric
        
        # Should handle missing metrics gracefully
        pareto_points = analyzer.compute_pareto_frontier([result])
        self.assertEqual(len(pareto_points), 0)  # Should filter out invalid results
    
    def test_single_result_analysis(self):
        """Test analysis with only one result."""
        result = Mock(spec=BrainsmithResult)
        result.metrics = Mock()
        result.metrics.metric1 = 100.0
        
        dse_result = Mock(spec=DSEResult)
        dse_result.results = [result]
        dse_result.total_time_seconds = 10.0
        dse_result.strategy = "single"
        dse_result.objectives = ["metric1"]
        dse_result.design_space = self.design_space
        
        analysis = self.analyzer.analyze_dse_result(dse_result)
        
        # Should handle single result gracefully
        self.assertEqual(analysis["summary"]["total_evaluations"], 1)
        self.assertIn("statistical_analysis", analysis)


if __name__ == '__main__':
    unittest.main()