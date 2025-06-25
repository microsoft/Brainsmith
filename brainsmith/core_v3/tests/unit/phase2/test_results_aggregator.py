"""Unit tests for the results aggregator."""

import pytest
from datetime import datetime

from brainsmith.core_v3.phase2.results_aggregator import ResultsAggregator
from brainsmith.core_v3.phase2.data_structures import (
    BuildConfig,
    BuildResult,
    BuildStatus,
    ExplorationResults,
)
from brainsmith.core_v3.phase1.data_structures import (
    BuildMetrics,
    GlobalConfig,
)


class TestResultsAggregator:
    """Test the ResultsAggregator class."""
    
    @pytest.fixture
    def exploration_results(self):
        """Create exploration results for testing."""
        return ExplorationResults(
            design_space_id="dse_test123",
            start_time=datetime.now(),
            end_time=datetime.now()
        )
    
    @pytest.fixture
    def sample_configs(self):
        """Create sample configurations."""
        configs = []
        for i in range(5):
            config = BuildConfig(
                id=f"config_{i:03d}",
                design_space_id="dse_test123",
                kernels=[("Gemm", ["rtl"])],
                transforms={"default": ["quantize"]},
                preprocessing=[],
                postprocessing=[],
                build_steps=["synth"],
                config_flags={},
                global_config=GlobalConfig(),
                combination_index=i,
                total_combinations=5
            )
            configs.append(config)
        return configs
    
    def test_add_result(self, exploration_results):
        """Test adding results to the aggregator."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add a successful result
        result = BuildResult(
            config_id="config_001",
            status=BuildStatus.SUCCESS
        )
        result.metrics = BuildMetrics(
            throughput=1000.0,
            latency=10.0,
            clock_frequency=250.0,
            lut_utilization=0.6,
            dsp_utilization=0.4,
            bram_utilization=0.3,
            total_power=10.0,
            accuracy=0.98
        )
        
        aggregator.add_result(result)
        
        assert len(exploration_results.evaluations) == 1
        assert exploration_results.evaluated_count == 1
        assert exploration_results.success_count == 1
    
    def test_find_best_config(self, exploration_results, sample_configs):
        """Test finding the best configuration."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add configs to exploration results
        for config in sample_configs:
            exploration_results.add_config(config)
        
        # Add results with varying throughput
        throughputs = [800.0, 1200.0, 1000.0, 900.0, 1100.0]
        for i, (config, throughput) in enumerate(zip(sample_configs, throughputs)):
            result = BuildResult(
                config_id=config.id,
                status=BuildStatus.SUCCESS
            )
            result.metrics = BuildMetrics(
                throughput=throughput,
                latency=10.0,
                clock_frequency=250.0,
                lut_utilization=0.6,
                dsp_utilization=0.4,
                bram_utilization=0.3,
                total_power=10.0,
                accuracy=0.98
            )
            aggregator.add_result(result)
        
        # Finalize and check best config
        aggregator.finalize()
        
        assert exploration_results.best_config is not None
        assert exploration_results.best_config.id == "config_001"  # Highest throughput
    
    def test_find_best_config_no_results(self, exploration_results):
        """Test finding best config with no successful results."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add only failed results
        for i in range(3):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.FAILED,
                error_message="Build failed"
            )
            aggregator.add_result(result)
        
        aggregator.finalize()
        
        assert exploration_results.best_config is None
    
    def test_find_pareto_optimal(self, exploration_results, sample_configs):
        """Test finding Pareto optimal configurations."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add configs
        for config in sample_configs:
            exploration_results.add_config(config)
        
        # Add results with different trade-offs
        # Format: (throughput, avg_resource_usage)
        metrics_data = [
            (1000.0, 0.3),  # High throughput, low resources - Pareto
            (800.0, 0.5),   # Lower throughput, higher resources - Dominated
            (1200.0, 0.8),  # Highest throughput, high resources - Pareto
            (900.0, 0.2),   # Medium throughput, lowest resources - Pareto
            (700.0, 0.6),   # Low throughput, high resources - Dominated
        ]
        
        for config, (throughput, avg_resource) in zip(sample_configs, metrics_data):
            result = BuildResult(
                config_id=config.id,
                status=BuildStatus.SUCCESS
            )
            result.metrics = BuildMetrics(
                throughput=throughput,
                latency=10.0,
                clock_frequency=250.0,
                lut_utilization=avg_resource,
                dsp_utilization=avg_resource,
                bram_utilization=avg_resource,
                total_power=10.0,
                accuracy=0.98
            )
            aggregator.add_result(result)
        
        aggregator.finalize()
        
        # Should have 3 Pareto optimal configs
        assert len(exploration_results.pareto_optimal) == 3
        
        # Check that the Pareto configs are correct
        pareto_ids = {c.id for c in exploration_results.pareto_optimal}
        assert "config_000" in pareto_ids  # High throughput, low resources
        assert "config_002" in pareto_ids  # Highest throughput
        assert "config_003" in pareto_ids  # Lowest resources
    
    def test_calculate_metrics_summary(self, exploration_results):
        """Test calculating metrics summary statistics."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add results with varying metrics
        for i in range(4):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.SUCCESS
            )
            result.metrics = BuildMetrics(
                throughput=1000.0 + i * 100,  # 1000, 1100, 1200, 1300
                latency=10.0 + i,              # 10, 11, 12, 13
                clock_frequency=250.0,
                lut_utilization=0.5 + i * 0.1, # 0.5, 0.6, 0.7, 0.8
                dsp_utilization=0.4,
                bram_utilization=0.3,
                total_power=10.0,
                accuracy=0.98
            )
            aggregator.add_result(result)
        
        aggregator.finalize()
        
        summary = exploration_results.metrics_summary
        
        # Check throughput statistics
        assert "throughput" in summary
        assert summary["throughput"]["min"] == 1000.0
        assert summary["throughput"]["max"] == 1300.0
        assert summary["throughput"]["mean"] == 1150.0
        assert summary["throughput"]["std"] > 0
        
        # Check latency statistics
        assert "latency" in summary
        assert summary["latency"]["min"] == 10.0
        assert summary["latency"]["max"] == 13.0
        assert summary["latency"]["mean"] == 11.5
    
    def test_get_top_n_configs(self, exploration_results, sample_configs):
        """Test getting top N configurations."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add configs
        for config in sample_configs:
            exploration_results.add_config(config)
        
        # Add results with varying throughput
        throughputs = [800.0, 1200.0, 1000.0, 900.0, 1100.0]
        for config, throughput in zip(sample_configs, throughputs):
            result = BuildResult(
                config_id=config.id,
                status=BuildStatus.SUCCESS
            )
            result.metrics = BuildMetrics(
                throughput=throughput,
                latency=10.0,
                clock_frequency=250.0,
                lut_utilization=0.6,
                dsp_utilization=0.4,
                bram_utilization=0.3,
                total_power=10.0,
                accuracy=0.98
            )
            aggregator.add_result(result)
        
        # Get top 3 configs
        top_configs = aggregator.get_top_n_configs(n=3)
        
        assert len(top_configs) == 3
        
        # Check order (highest throughput first)
        assert top_configs[0][0].id == "config_001"  # 1200.0
        assert top_configs[1][0].id == "config_004"  # 1100.0
        assert top_configs[2][0].id == "config_002"  # 1000.0
        
        # Check that results are included
        assert top_configs[0][1].metrics.throughput == 1200.0
    
    def test_get_failed_summary(self, exploration_results):
        """Test getting failure summary."""
        aggregator = ResultsAggregator(exploration_results)
        
        # Add various failed results
        error_messages = [
            "Timing constraints not met",
            "Resource utilization exceeded",
            "Timing constraints not met",
            "Synthesis failed",
            "Timing constraints not met",
        ]
        
        for i, error in enumerate(error_messages):
            result = BuildResult(
                config_id=f"config_{i:03d}",
                status=BuildStatus.FAILED,
                error_message=error
            )
            aggregator.add_result(result)
        
        # Get failure summary
        failure_summary = aggregator.get_failed_summary()
        
        assert len(failure_summary) == 3
        assert failure_summary["Timing constraints not met"] == 3
        assert failure_summary["Resource utilization exceeded"] == 1
        assert failure_summary["Synthesis failed"] == 1