############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for design space explorer"""

import pytest
from brainsmith.core.dataflow.types import INT16, InterfaceDirection
from brainsmith.core.dataflow.interface import Interface
from brainsmith.core.dataflow.kernel import Kernel
from brainsmith.core.dataflow.graph import DataflowGraph
from brainsmith.core.dataflow.dse.config import (
    ParallelismConfig, DSEConstraints, ConfigurationSpace
)
from brainsmith.core.dataflow.dse.explorer import (
    DesignSpaceExplorer, DSEResult
)
from brainsmith.core.dataflow.dse.evaluator import PerformanceMetrics


class TestDSEResult:
    """Test DSE result structure"""
    
    def test_result_creation(self):
        """Test creating DSE result"""
        config = ParallelismConfig(interface_pars={("k1", "in"): 8})
        metrics = PerformanceMetrics(
            throughput=100.0,
            latency=1000,
            fps=100.0,
            power_estimate=10.0
        )
        
        result = DSEResult(
            config=config,
            metrics=metrics,
            feasible=True
        )
        
        assert result.config == config
        assert result.metrics == metrics
        assert result.feasible
        assert len(result.violation_reasons) == 0
    
    def test_infeasible_result(self):
        """Test infeasible result with violations"""
        config = ParallelismConfig()
        metrics = PerformanceMetrics(0.0, float('inf'), 0.0)
        
        result = DSEResult(
            config=config,
            metrics=metrics,
            feasible=False,
            violation_reasons=["DSP limit exceeded", "Not schedulable"]
        )
        
        assert not result.feasible
        assert len(result.violation_reasons) == 2


class TestDesignSpaceExplorer:
    """Test design space exploration"""
    
    def create_test_graph(self):
        """Create simple test graph"""
        k1 = Kernel(
            name="k1",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (256,), (32,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (128,), (16,))
            ],
            latency_cycles=(100, 80),
            resources={"DSP": 16, "BRAM": 4}
        )
        
        k2 = Kernel(
            name="k2",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (128,), (16,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (64,), (8,))
            ],
            latency_cycles=(50, 40),
            resources={"DSP": 8, "BRAM": 2}
        )
        
        graph = DataflowGraph()
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_edge("k1", "out", "k2", "in")
        
        return graph
    
    def test_basic_exploration(self):
        """Test basic design space exploration"""
        graph = self.create_test_graph()
        constraints = DSEConstraints(
            min_parallelism=1,
            max_parallelism=8,
            allowed_parallelisms=[1, 2, 4, 8]
        )
        
        explorer = DesignSpaceExplorer(graph, constraints)
        
        # Create small config space
        space = ConfigurationSpace()
        space.add_interface("k1", "in", [1, 2, 4])
        space.add_interface("k1", "out", [1, 2, 4])
        
        results = explorer.explore(space)
        
        # Should explore 3x3 = 9 configurations
        assert len(results) == 9
        
        # Results should be sorted by feasibility and throughput
        for i in range(1, len(results)):
            if results[i-1].feasible and results[i].feasible:
                # Both feasible: check throughput order
                assert results[i-1].metrics.throughput >= results[i].metrics.throughput
            else:
                # Feasible should come before infeasible
                assert results[i-1].feasible >= results[i].feasible
    
    def test_constraint_checking(self):
        """Test constraint violation detection"""
        graph = self.create_test_graph()
        
        # Very restrictive constraints
        constraints = DSEConstraints(
            max_dsp=20,  # Only allows minimal parallelism
            min_throughput=1000.0  # High throughput requirement
        )
        
        explorer = DesignSpaceExplorer(graph, constraints)
        
        # Try high parallelism config
        config = ParallelismConfig(
            interface_pars={
                ("k1", "in"): 8,
                ("k1", "out"): 8,
                ("k2", "in"): 8,
                ("k2", "out"): 8
            }
        )
        
        result = explorer._evaluate_config(config, batch_size=1)
        
        # Should be infeasible due to DSP constraint
        assert not result.feasible
        assert any("DSP" in v for v in result.violation_reasons)
    
    def test_default_space_generation(self):
        """Test automatic configuration space generation"""
        graph = self.create_test_graph()
        constraints = DSEConstraints(
            allowed_parallelisms=[1, 2, 4]
        )
        
        explorer = DesignSpaceExplorer(graph, constraints)
        space = explorer._generate_default_space()
        
        # Should include input/output interfaces
        assert ("k1", "in") in space.interface_options
        assert ("k1", "out") in space.interface_options
        assert ("k2", "in") in space.interface_options
        assert ("k2", "out") in space.interface_options
        
        # Should have coupling for connected interfaces
        assert len(space.coupled_interfaces) > 0
    
    def test_caching(self):
        """Test configuration caching"""
        graph = self.create_test_graph()
        constraints = DSEConstraints()
        explorer = DesignSpaceExplorer(graph, constraints)
        
        # Create config space with duplicate configs
        space = ConfigurationSpace()
        space.add_interface("k1", "in", [4])  # Only one option
        
        # Manually add the same config twice
        config1 = ParallelismConfig(interface_pars={("k1", "in"): 4})
        config2 = ParallelismConfig(interface_pars={("k1", "in"): 4})
        
        # Explore should use cache for second occurrence
        results = explorer.explore(space)
        
        # Check cache was used
        cache_key = explorer._config_cache_key(config1)
        assert cache_key in explorer._cache
        assert len(explorer._cache) == 1  # Only one unique config
    
    def test_pareto_optimal_finding(self):
        """Test Pareto optimal configuration finding"""
        graph = self.create_test_graph()
        constraints = DSEConstraints()
        explorer = DesignSpaceExplorer(graph, constraints)
        
        # Create mock results with different trade-offs
        results = [
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(100, 100, 100, power_estimate=10),
                feasible=True
            ),
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(200, 50, 200, power_estimate=20),
                feasible=True
            ),
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(150, 75, 150, power_estimate=15),
                feasible=True
            ),
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(50, 200, 50, power_estimate=5),
                feasible=True
            )
        ]
        
        pareto = explorer.find_pareto_optimal(results)
        
        # Results 1 (best latency) and 3 (best power) should be Pareto optimal
        # Result 0 is dominated by result 2 (better in all metrics)
        assert len(pareto) >= 2
        assert results[1] in pareto  # Best latency
        assert results[3] in pareto  # Best power
    
    def test_progress_callback(self):
        """Test progress callback during exploration"""
        graph = self.create_test_graph()
        constraints = DSEConstraints()
        explorer = DesignSpaceExplorer(graph, constraints)
        
        progress_calls = []
        
        def progress_callback(current, total):
            progress_calls.append((current, total))
        
        space = ConfigurationSpace()
        space.add_interface("k1", "in", [1, 2])
        
        explorer.explore(space, progress_callback=progress_callback)
        
        # Should have received progress updates
        assert len(progress_calls) > 0
        # Final call should be (n-1, n) where n is total configs
        assert progress_calls[-1][0] == progress_calls[-1][1] - 1
    
    def test_results_summary(self):
        """Test results summarization"""
        graph = self.create_test_graph()
        constraints = DSEConstraints()
        explorer = DesignSpaceExplorer(graph, constraints)
        
        # Create mix of feasible and infeasible results
        results = [
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(100, 100, 100, 
                                         resource_usage={"DSP": 50, "BRAM": 10},
                                         power_estimate=10),
                feasible=True
            ),
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(200, 50, 200,
                                         resource_usage={"DSP": 100, "BRAM": 20},
                                         power_estimate=20),
                feasible=True
            ),
            DSEResult(
                config=ParallelismConfig(),
                metrics=PerformanceMetrics(0, float('inf'), 0),
                feasible=False,
                violation_reasons=["DSP limit exceeded", "Not schedulable"]
            )
        ]
        
        summary = explorer.summarize_results(results)
        
        assert summary["n_configs"] == 3
        assert summary["n_feasible"] == 2
        assert summary["feasibility_rate"] == 2/3
        
        # Performance ranges
        assert summary["throughput_range"] == (100, 200)
        assert summary["latency_range"] == (50, 100)
        assert summary["power_range"] == (10, 20)
        
        # Resource ranges
        assert summary["dsp_range"] == (50, 100)
        assert summary["bram_range"] == (10, 20)
        
        # Common violations
        assert len(summary["common_violations"]) > 0