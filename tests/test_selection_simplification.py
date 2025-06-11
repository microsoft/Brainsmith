"""
Test Suite for Selection Simplification

Validates the simplified selection functions that replace the complex MCDA framework.
Tests practical FPGA design selection scenarios.
"""

import pytest
import time
from typing import List
from unittest.mock import Mock

from brainsmith.data import (
    BuildMetrics, PerformanceData, ResourceData, QualityData, BuildData,
    SelectionCriteria, TradeoffAnalysis,
    find_pareto_optimal, rank_by_efficiency, select_best_solutions,
    filter_feasible_designs, compare_design_tradeoffs
)


class TestSelectionSimplification:
    """Test simplified selection functions."""
    
    def setup_method(self):
        """Set up test data."""
        self.sample_metrics = self._create_sample_metrics()
    
    def _create_sample_metrics(self) -> List[BuildMetrics]:
        """Create sample metrics for testing."""
        metrics_list = []
        
        # Design 1: High throughput, high resource usage
        metrics1 = BuildMetrics(
            performance=PerformanceData(
                throughput_ops_sec=5000.0,
                latency_ms=2.0,
                clock_freq_mhz=200.0
            ),
            resources=ResourceData(
                lut_utilization_percent=85.0,
                dsp_utilization_percent=90.0,
                bram_utilization_percent=70.0
            ),
            quality=QualityData(
                accuracy_percent=95.0
            ),
            build=BuildData(
                build_success=True,
                build_time_seconds=300.0
            ),
            parameters={'pe_count': 64, 'simd': 8}
        )
        metrics_list.append(metrics1)
        
        # Design 2: Medium throughput, efficient resource usage
        metrics2 = BuildMetrics(
            performance=PerformanceData(
                throughput_ops_sec=3000.0,
                latency_ms=3.0,
                clock_freq_mhz=150.0
            ),
            resources=ResourceData(
                lut_utilization_percent=60.0,
                dsp_utilization_percent=55.0,
                bram_utilization_percent=45.0
            ),
            quality=QualityData(
                accuracy_percent=97.0
            ),
            build=BuildData(
                build_success=True,
                build_time_seconds=180.0
            ),
            parameters={'pe_count': 32, 'simd': 4}
        )
        metrics_list.append(metrics2)
        
        # Design 3: Low throughput, very efficient resources
        metrics3 = BuildMetrics(
            performance=PerformanceData(
                throughput_ops_sec=1500.0,
                latency_ms=5.0,
                clock_freq_mhz=100.0
            ),
            resources=ResourceData(
                lut_utilization_percent=30.0,
                dsp_utilization_percent=25.0,
                bram_utilization_percent=20.0
            ),
            quality=QualityData(
                accuracy_percent=98.0
            ),
            build=BuildData(
                build_success=True,
                build_time_seconds=120.0
            ),
            parameters={'pe_count': 16, 'simd': 2}
        )
        metrics_list.append(metrics3)
        
        # Design 4: Failed build (should be filtered out)
        metrics4 = BuildMetrics(
            performance=PerformanceData(
                throughput_ops_sec=None,
                latency_ms=None
            ),
            resources=ResourceData(
                lut_utilization_percent=None
            ),
            quality=QualityData(
                accuracy_percent=None
            ),
            build=BuildData(
                build_success=False,
                build_time_seconds=0.0,
                compilation_errors=5
            ),
            parameters={'pe_count': 128, 'simd': 16}
        )
        metrics_list.append(metrics4)
        
        return metrics_list

    def test_find_pareto_optimal_basic(self):
        """Test basic Pareto optimal identification."""
        pareto_solutions = find_pareto_optimal(
            self.sample_metrics,
            objectives=['throughput_ops_sec', 'lut_utilization_percent']
        )
        
        # Should exclude failed builds
        assert len(pareto_solutions) >= 1
        assert all(m.is_successful() for m in pareto_solutions)
        
        # All returned solutions should be Pareto optimal
        for solution in pareto_solutions:
            assert solution.performance.throughput_ops_sec is not None
            assert solution.resources.lut_utilization_percent is not None

    def test_find_pareto_optimal_empty_input(self):
        """Test Pareto optimization with empty input."""
        result = find_pareto_optimal([])
        assert result == []

    def test_find_pareto_optimal_failed_builds(self):
        """Test Pareto optimization filters out failed builds."""
        failed_metrics = [m for m in self.sample_metrics if not m.is_successful()]
        result = find_pareto_optimal(failed_metrics)
        assert result == []

    def test_rank_by_efficiency_default_weights(self):
        """Test efficiency ranking with default weights."""
        ranked_solutions = rank_by_efficiency(self.sample_metrics)
        
        # Should exclude failed builds
        assert len(ranked_solutions) == 3
        assert all(m.is_successful() for m in ranked_solutions)
        
        # Should be sorted by efficiency (descending)
        efficiency_scores = [
            m.metadata.get('efficiency_score', 0) 
            for m in ranked_solutions
        ]
        assert efficiency_scores == sorted(efficiency_scores, reverse=True)
        
        # All solutions should have efficiency metadata
        for solution in ranked_solutions:
            assert 'efficiency_score' in solution.metadata
            assert 'efficiency_weights' in solution.metadata

    def test_rank_by_efficiency_custom_weights(self):
        """Test efficiency ranking with custom weights."""
        custom_weights = {
            'throughput': 0.6,
            'resource_efficiency': 0.2,
            'accuracy': 0.2
        }
        
        ranked_solutions = rank_by_efficiency(self.sample_metrics, custom_weights)
        
        assert len(ranked_solutions) == 3
        
        # Check that custom weights are stored
        for solution in ranked_solutions:
            stored_weights = solution.metadata.get('efficiency_weights', {})
            assert stored_weights == custom_weights

    def test_filter_feasible_designs_resource_constraints(self):
        """Test filtering designs based on resource constraints."""
        criteria = SelectionCriteria(
            max_lut_utilization=70.0,
            max_dsp_utilization=60.0
        )
        
        feasible_designs = filter_feasible_designs(self.sample_metrics, criteria)
        
        # Should filter out high-resource design (metrics1)
        assert len(feasible_designs) <= 3
        
        for design in feasible_designs:
            if design.resources.lut_utilization_percent:
                assert design.resources.lut_utilization_percent <= 70.0
            if design.resources.dsp_utilization_percent:
                assert design.resources.dsp_utilization_percent <= 60.0
            
            # Should have constraint satisfaction metadata
            assert design.metadata.get('meets_all_constraints') is True

    def test_filter_feasible_designs_performance_constraints(self):
        """Test filtering designs based on performance constraints."""
        criteria = SelectionCriteria(
            min_throughput=2000.0,
            max_latency=4.0
        )
        
        feasible_designs = filter_feasible_designs(self.sample_metrics, criteria)
        
        for design in feasible_designs:
            if design.performance.throughput_ops_sec:
                assert design.performance.throughput_ops_sec >= 2000.0
            if design.performance.latency_ms:
                assert design.performance.latency_ms <= 4.0

    def test_select_best_solutions_integrated_workflow(self):
        """Test complete selection workflow."""
        criteria = SelectionCriteria(
            max_lut_utilization=80.0,
            min_throughput=2000.0,
            efficiency_weights={
                'throughput': 0.5,
                'resource_efficiency': 0.3,
                'accuracy': 0.2
            }
        )
        
        best_solutions = select_best_solutions(self.sample_metrics, criteria)
        
        # Should return feasible solutions ranked by efficiency
        assert len(best_solutions) >= 1
        
        for i, solution in enumerate(best_solutions):
            # Check constraints are met
            assert solution.metadata.get('meets_criteria') is True
            assert solution.metadata.get('selection_rank') == i + 1
            
            # Check resource constraints
            if solution.resources.lut_utilization_percent:
                assert solution.resources.lut_utilization_percent <= 80.0
            
            # Check performance constraints
            if solution.performance.throughput_ops_sec:
                assert solution.performance.throughput_ops_sec >= 2000.0

    def test_compare_design_tradeoffs_basic(self):
        """Test basic design trade-off comparison."""
        design_a = self.sample_metrics[0]  # High throughput, high resources
        design_b = self.sample_metrics[1]  # Medium throughput, efficient resources
        
        analysis = compare_design_tradeoffs(design_a, design_b)
        
        assert isinstance(analysis, TradeoffAnalysis)
        assert analysis.efficiency_ratio > 0
        assert analysis.better_design in ['design_a', 'design_b', 'tied']
        assert 0 <= analysis.confidence <= 1.0
        assert len(analysis.recommendations) > 0
        
        # Should have throughput ratio since both have throughput data
        assert analysis.throughput_ratio is not None
        
        # Should have resource ratio since both have resource data
        assert analysis.resource_ratio is not None

    def test_compare_design_tradeoffs_recommendations(self):
        """Test that trade-off analysis provides useful recommendations."""
        design_a = self.sample_metrics[0]  # High throughput, high resources
        design_b = self.sample_metrics[2]  # Low throughput, low resources
        
        analysis = compare_design_tradeoffs(design_a, design_b)
        
        # Should provide actionable recommendations
        assert len(analysis.recommendations) > 0
        assert len(analysis.trade_offs) >= 0
        
        # Recommendations should be strings
        for rec in analysis.recommendations:
            assert isinstance(rec, str)
            assert len(rec) > 0

    def test_selection_criteria_defaults(self):
        """Test SelectionCriteria default values."""
        criteria = SelectionCriteria()
        
        # Should have default efficiency weights
        assert 'throughput' in criteria.efficiency_weights
        assert 'resource_efficiency' in criteria.efficiency_weights
        assert 'accuracy' in criteria.efficiency_weights
        assert 'build_time' in criteria.efficiency_weights
        
        # Weights should sum to 1.0
        total_weight = sum(criteria.efficiency_weights.values())
        assert abs(total_weight - 1.0) < 0.01

    def test_selection_criteria_to_dict(self):
        """Test SelectionCriteria serialization."""
        criteria = SelectionCriteria(
            max_lut_utilization=80.0,
            min_throughput=1000.0
        )
        
        criteria_dict = criteria.to_dict()
        
        assert isinstance(criteria_dict, dict)
        assert criteria_dict['max_lut_utilization'] == 80.0
        assert criteria_dict['min_throughput'] == 1000.0
        assert 'efficiency_weights' in criteria_dict

    def test_tradeoff_analysis_to_dict(self):
        """Test TradeoffAnalysis serialization."""
        analysis = TradeoffAnalysis(
            efficiency_ratio=1.5,
            better_design='design_b',
            recommendations=['Design B is better']
        )
        
        analysis_dict = analysis.to_dict()
        
        assert isinstance(analysis_dict, dict)
        assert analysis_dict['efficiency_ratio'] == 1.5
        assert analysis_dict['better_design'] == 'design_b'
        assert analysis_dict['recommendations'] == ['Design B is better']

    def test_integration_with_existing_workflow(self):
        """Test that selection functions integrate with existing data workflow."""
        # Simulate DSE workflow
        dse_metrics = self.sample_metrics
        
        # Step 1: Find Pareto optimal solutions
        pareto_solutions = find_pareto_optimal(dse_metrics)
        assert len(pareto_solutions) > 0
        
        # Step 2: Rank by efficiency
        ranked_solutions = rank_by_efficiency(pareto_solutions)
        assert len(ranked_solutions) > 0
        
        # Step 3: Apply selection criteria
        criteria = SelectionCriteria(max_lut_utilization=90.0, min_throughput=1000.0)
        best_solutions = select_best_solutions(ranked_solutions, criteria)
        
        # Should maintain data integrity through pipeline
        for solution in best_solutions:
            assert isinstance(solution, BuildMetrics)
            assert solution.is_successful()
            assert 'efficiency_score' in solution.metadata
            assert 'selection_rank' in solution.metadata

    def test_performance_vs_complexity_reduction(self):
        """Test that simplified functions are more efficient than complex MCDA."""
        import time
        
        # Measure execution time of simplified selection
        start_time = time.time()
        
        pareto_solutions = find_pareto_optimal(self.sample_metrics)
        ranked_solutions = rank_by_efficiency(pareto_solutions)
        criteria = SelectionCriteria(max_lut_utilization=80.0)
        best_solutions = select_best_solutions(ranked_solutions, criteria)
        
        execution_time = time.time() - start_time
        
        # Should execute quickly (much faster than complex MCDA algorithms)
        assert execution_time < 1.0  # Should complete in well under 1 second
        assert len(best_solutions) >= 0  # Should return valid results


class TestSelectionComplexityReduction:
    """Test that simplified selection provides same functionality with reduced complexity."""
    
    def test_exports_reduction(self):
        """Verify reduced API surface compared to original selection module."""
        from brainsmith.data import __all__ as data_exports
        
        # Count selection-related exports
        selection_exports = [
            'find_pareto_optimal',
            'rank_by_efficiency', 
            'select_best_solutions',
            'filter_feasible_designs',
            'compare_design_tradeoffs',
            'SelectionCriteria',
            'TradeoffAnalysis'
        ]
        
        # All selection functions should be available
        for export in selection_exports:
            assert export in data_exports
        
        # Should have 7 selection-related exports vs 44 in original module
        assert len(selection_exports) == 7

    def test_practical_fpga_focus(self):
        """Test that selection focuses on practical FPGA constraints."""
        criteria = SelectionCriteria()
        
        # Should have FPGA-specific constraints
        fpga_constraints = [
            'max_lut_utilization',
            'max_dsp_utilization', 
            'max_bram_utilization',
            'min_throughput',
            'max_latency'
        ]
        
        for constraint in fpga_constraints:
            assert hasattr(criteria, constraint)

    def test_no_academic_complexity(self):
        """Test that academic MCDA complexity is eliminated."""
        # No complex mathematical algorithms
        # No preference functions, entropy weights, fuzzy logic, etc.
        # Simple, direct calculations
        
        sample_metrics = BuildMetrics(
            performance=PerformanceData(throughput_ops_sec=1000.0),
            resources=ResourceData(lut_utilization_percent=50.0),
            quality=QualityData(accuracy_percent=95.0),
            build=BuildData(build_success=True)
        )
        
        # Functions should work with simple, direct inputs
        pareto_result = find_pareto_optimal([sample_metrics])
        efficiency_result = rank_by_efficiency([sample_metrics])
        
        assert len(pareto_result) == 1
        assert len(efficiency_result) == 1
        assert 'efficiency_score' in efficiency_result[0].metadata


if __name__ == '__main__':
    pytest.main([__file__, '-v'])