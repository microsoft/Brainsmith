############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for DSE performance evaluator"""

import pytest
from brainsmith.core.dataflow.core.types import INT16, InterfaceDirection
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.kernel import Kernel
from brainsmith.core.dataflow.core.graph import DataflowGraph
from brainsmith.core.dataflow.adfg.scheduler import SRTAScheduler, SchedulabilityResult
from brainsmith.core.dataflow.dse.evaluator import (
    PerformanceEvaluator, PerformanceMetrics
)


class TestPerformanceMetrics:
    """Test performance metrics data structure"""
    
    def test_metrics_creation(self):
        """Test creating performance metrics"""
        metrics = PerformanceMetrics(
            throughput=100.0,
            latency=1000,
            fps=30.0,
            resource_usage={"DSP": 100, "BRAM": 50},
            power_estimate=10.5
        )
        
        assert metrics.throughput == 100.0
        assert metrics.latency == 1000
        assert metrics.fps == 30.0
        assert metrics.resource_usage["DSP"] == 100
        assert metrics.power_estimate == 10.5
    
    def test_metrics_repr(self):
        """Test string representation"""
        metrics = PerformanceMetrics(
            throughput=150.5,
            latency=2000,
            fps=60.2
        )
        
        repr_str = repr(metrics)
        assert "150.5" in repr_str
        assert "2000" in repr_str
        assert "60.2" in repr_str


class TestPerformanceEvaluator:
    """Test performance evaluation"""
    
    def create_simple_graph(self):
        """Create simple test graph"""
        # Two-kernel pipeline
        k1 = Kernel(
            name="producer",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (256,), (32,))
            ],
            latency_cycles=(100, 80),
            priming_cycles=10,
            flush_cycles=5,
            resources={"DSP": 16, "BRAM": 4}
        )
        
        k2 = Kernel(
            name="consumer",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (256,), (32,))
            ],
            latency_cycles=(50, 40),
            priming_cycles=5,
            flush_cycles=2,
            resources={"DSP": 8, "BRAM": 2}
        )
        
        graph = DataflowGraph()
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_edge("producer", "out", "consumer", "in")
        
        return graph
    
    def test_basic_evaluation(self):
        """Test basic performance evaluation"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator(frequency_mhz=200.0)
        
        metrics = evaluator.evaluate(graph, batch_size=1)
        
        # Check metrics are computed
        assert metrics.throughput > 0
        assert metrics.latency > 0
        assert metrics.fps > 0
        assert len(metrics.resource_usage) > 0
        assert metrics.power_estimate > 0
    
    def test_throughput_calculation(self):
        """Test throughput calculation"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator(frequency_mhz=200.0)
        
        # Create mock schedule
        from brainsmith.core.dataflow.adfg.actor import ActorTiming
        schedule = SchedulabilityResult(
            schedulable=True,
            actor_timings={
                "producer": ActorTiming("producer", 200, 200, 100, 0, 0.5),
                "consumer": ActorTiming("consumer", 200, 200, 50, 0, 0.25)
            },
            total_utilization=0.75,
            hyperperiod=200
        )
        
        metrics = evaluator.evaluate(graph, schedule=schedule, batch_size=1)
        
        # Throughput = frequency / hyperperiod * batch_size
        # = 200e6 / 200 * 1 = 1e6 inferences/sec
        assert metrics.throughput == pytest.approx(1e6, rel=0.01)
    
    def test_latency_calculation(self):
        """Test latency calculation"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator()
        
        # Single batch
        metrics = evaluator.evaluate(graph, batch_size=1)
        
        # Latency should include priming + execution + flush
        # Priming: 10 + 5 = 15
        # Execution: critical path = 100 + 50 = 150
        # Flush: 5 + 2 = 7
        # Total: 15 + 150 + 7 = 172
        assert metrics.latency >= 172
    
    def test_batch_latency(self):
        """Test latency with batching"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator()
        
        # Create schedule with II=100
        from brainsmith.core.dataflow.adfg.actor import ActorTiming
        schedule = SchedulabilityResult(
            schedulable=True,
            actor_timings={
                "producer": ActorTiming("producer", 100, 100, 100, 0, 1.0),
                "consumer": ActorTiming("consumer", 100, 100, 50, 0, 0.5)
            },
            total_utilization=1.5,
            hyperperiod=100
        )
        
        # Batch of 5
        metrics = evaluator.evaluate(graph, schedule=schedule, batch_size=5)
        
        # Latency = priming + critical_path + (batch-1)*II + flush
        # = 15 + 150 + 4*100 + 7 = 572
        assert metrics.latency >= 572
    
    def test_sparsity_effects(self):
        """Test sparsity on effective throughput"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator()
        
        # Base case: no sparsity
        metrics_base = evaluator.evaluate(graph, batch_size=1)
        
        # With 50% sparsity (50% of inputs skipped)
        metrics_sparse = evaluator.evaluate(
            graph, 
            batch_size=1,
            input_sparsity={"in": 0.5}
        )
        
        # Effective throughput should double with 50% sparsity
        assert metrics_sparse.fps == pytest.approx(2 * metrics_base.fps, rel=0.1)
    
    def test_resource_estimation(self):
        """Test resource usage estimation"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator()
        
        metrics = evaluator.evaluate(graph)
        
        # Should aggregate resources
        assert metrics.resource_usage["DSP"] == 24  # 16 + 8
        assert metrics.resource_usage["BRAM"] == 6   # 4 + 2
        assert "bandwidth_gbps" in metrics.resource_usage
    
    def test_power_estimation(self):
        """Test power estimation"""
        graph = self.create_simple_graph()
        evaluator = PerformanceEvaluator()
        
        metrics = evaluator.evaluate(graph)
        
        # Power should be reasonable
        assert 0 < metrics.power_estimate < 100  # Less than 100W
        
        # Higher utilization should increase power
        # Create high utilization schedule
        from brainsmith.core.dataflow.adfg.actor import ActorTiming
        high_util_schedule = SchedulabilityResult(
            schedulable=True,
            actor_timings={
                "producer": ActorTiming("producer", 100, 100, 100, 0, 1.0),
                "consumer": ActorTiming("consumer", 100, 100, 50, 0, 0.5)
            },
            total_utilization=1.5,  # Over-utilized
            hyperperiod=100
        )
        
        metrics_high = evaluator.evaluate(graph, schedule=high_util_schedule)
        
        # Power should scale with activity
        assert metrics_high.power_estimate >= metrics.power_estimate
    
    def test_unschedulable_config(self):
        """Test handling of unschedulable configurations"""
        # Create overloaded graph
        k1 = Kernel(
            name="heavy",
            interfaces=[Interface("out", InterfaceDirection.OUTPUT, INT16, (1024,), (1024,))],
            latency_cycles=(10000, 10000)  # Very long latency
        )
        
        graph = DataflowGraph()
        graph.add_kernel(k1)
        
        evaluator = PerformanceEvaluator()
        
        # Force unschedulable by creating bad schedule
        unschedulable = SchedulabilityResult(
            schedulable=False,
            actor_timings={},
            total_utilization=2.0,
            hyperperiod=0,
            failure_reason="Utilization too high"
        )
        
        metrics = evaluator.evaluate(graph, schedule=unschedulable)
        
        # Should return worst-case metrics
        assert metrics.throughput == 0.0
        assert metrics.latency == float('inf')
        assert metrics.processor_utilization > 1.0


class TestConfigurationComparison:
    """Test comparing multiple configurations"""
    
    def test_compare_configs(self):
        """Test configuration comparison"""
        evaluator = PerformanceEvaluator()
        
        # Create mock configurations with different trade-offs
        configs = [
            # Config 0: High throughput, high power
            (None, PerformanceMetrics(
                throughput=1000.0, latency=100, fps=1000.0,
                power_estimate=50.0
            )),
            # Config 1: Low latency, medium power
            (None, PerformanceMetrics(
                throughput=500.0, latency=50, fps=500.0,
                power_estimate=30.0
            )),
            # Config 2: Low power, low performance
            (None, PerformanceMetrics(
                throughput=200.0, latency=200, fps=200.0,
                power_estimate=10.0
            ))
        ]
        
        comparison = evaluator.compare_configurations(configs)
        
        assert comparison["n_configs"] == 3
        assert comparison["throughput_range"] == (200.0, 1000.0)
        assert comparison["latency_range"] == (50, 200)
        assert comparison["power_range"] == (10.0, 50.0)
        
        # Best indices
        assert comparison["best_throughput_idx"] == 0
        assert comparison["best_latency_idx"] == 1
        assert comparison["best_power_idx"] == 2
    
    def test_pareto_optimal(self):
        """Test Pareto optimality detection"""
        evaluator = PerformanceEvaluator()
        
        configs = [
            # Config 0: Dominated by config 1 (worse in all metrics)
            (None, PerformanceMetrics(
                throughput=100.0, latency=100, fps=100.0,
                power_estimate=50.0
            )),
            # Config 1: Pareto optimal
            (None, PerformanceMetrics(
                throughput=200.0, latency=50, fps=200.0,
                power_estimate=30.0
            )),
            # Config 2: Pareto optimal (trades throughput for power)
            (None, PerformanceMetrics(
                throughput=150.0, latency=60, fps=150.0,
                power_estimate=20.0
            ))
        ]
        
        pareto_indices = evaluator._find_pareto_optimal(configs)
        
        # Config 0 is dominated, configs 1 and 2 are Pareto optimal
        assert 0 not in pareto_indices
        assert 1 in pareto_indices
        assert 2 in pareto_indices