############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for buffer sizing ILP"""

import pytest
from brainsmith.core.dataflow.adfg.actor import ADFGActor
from brainsmith.core.dataflow.adfg.buffer_ilp import (
    BufferSizingILP, BufferConfig, BufferSolution, HAS_PULP
)


# Skip all tests if PuLP not available
pytestmark = pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")


class TestBufferConfig:
    """Test buffer configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = BufferConfig()
        
        assert config.memory_limit is None
        assert config.min_buffer_size == 1
        assert config.objective == "min_total"
        assert config.memory_banks == 1
        assert config.enforce_back_pressure == True
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Invalid minimum size
        with pytest.raises(ValueError, match="Minimum buffer size"):
            BufferConfig(min_buffer_size=0)
        
        # Invalid max < min
        with pytest.raises(ValueError, match="Maximum buffer size"):
            BufferConfig(min_buffer_size=10, max_buffer_size=5)
        
        # Invalid objective
        with pytest.raises(ValueError, match="Unknown objective"):
            BufferConfig(objective="invalid")
        
        # Invalid memory banks
        with pytest.raises(ValueError, match="at least one memory bank"):
            BufferConfig(memory_banks=0)


class TestBufferSizingBasic:
    """Test basic buffer sizing functionality"""
    
    def test_empty_graph(self):
        """Test buffer sizing for empty graph"""
        ilp = BufferSizingILP()
        solution = ilp.solve([], [])
        
        assert solution.feasible
        assert solution.total_memory == 0
        assert len(solution.buffer_sizes) == 0
    
    def test_simple_edge(self):
        """Test buffer sizing for single edge"""
        actors = [
            ADFGActor("prod", wcet=10, rates={"out": [2]}),
            ADFGActor("cons", wcet=20, rates={"in": [3]})
        ]
        edges = [("prod", "out", "cons", "in")]
        
        ilp = BufferSizingILP()
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert ("prod", "cons") in solution.buffer_sizes
        
        # Buffer must be at least max(2, 3) = 3
        assert solution.buffer_sizes[("prod", "cons")] >= 3
    
    def test_pipeline(self):
        """Test buffer sizing for pipeline"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [1]}),
            ADFGActor("a2", wcet=15, rates={"in": [1], "out": [2]}),
            ADFGActor("a3", wcet=20, rates={"in": [2]})
        ]
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        ilp = BufferSizingILP()
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert len(solution.buffer_sizes) == 2
        assert solution.total_memory >= 3  # At least 1 + 2


class TestBufferObjectives:
    """Test different optimization objectives"""
    
    def test_minimize_total(self):
        """Test minimizing total buffer memory"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [3]}),
            ADFGActor("a2", wcet=20, rates={"in": [2], "out": [4]}),
            ADFGActor("a3", wcet=15, rates={"in": [5]})
        ]
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        config = BufferConfig(objective="min_total")
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # Solution minimizes sum of buffer sizes
    
    def test_minimize_max(self):
        """Test minimizing maximum buffer size"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out1": [10], "out2": [1]}),
            ADFGActor("a2", wcet=20, rates={"in": [10]}),
            ADFGActor("a3", wcet=15, rates={"in": [1]})
        ]
        edges = [
            ("a1", "out1", "a2", "in"),
            ("a1", "out2", "a3", "in")
        ]
        
        config = BufferConfig(objective="min_max")
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # One edge needs size 10, other needs size 1
        # min_max should balance them somewhat
    
    def test_balanced_sizing(self):
        """Test balanced buffer sizing"""
        actors = [
            ADFGActor("source", wcet=10, rates={"out1": [2], "out2": [2]}),
            ADFGActor("sink1", wcet=20, rates={"in": [2]}),
            ADFGActor("sink2", wcet=20, rates={"in": [2]})
        ]
        edges = [
            ("source", "out1", "sink1", "in"),
            ("source", "out2", "sink2", "in")
        ]
        
        config = BufferConfig(objective="balanced")
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # Both edges have same rates, should get same size
        sizes = list(solution.buffer_sizes.values())
        assert len(set(sizes)) == 1  # All same size


class TestMemoryConstraints:
    """Test memory constraints"""
    
    def test_memory_limit(self):
        """Test total memory limit constraint"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [5]}),
            ADFGActor("a2", wcet=20, rates={"in": [5], "out": [5]}),
            ADFGActor("a3", wcet=15, rates={"in": [5]})
        ]
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        # Tight memory limit
        config = BufferConfig(memory_limit=10)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert solution.total_memory <= 10
        assert solution.memory_utilization <= 1.0
    
    def test_infeasible_memory_limit(self):
        """Test infeasible memory constraint"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [10]}),
            ADFGActor("a2", wcet=20, rates={"in": [10]})
        ]
        edges = [("a1", "out", "a2", "in")]
        
        # Impossible constraint (need at least 10)
        config = BufferConfig(memory_limit=5)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert not solution.feasible
    
    def test_max_buffer_size(self):
        """Test maximum buffer size constraint"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [8]}),
            ADFGActor("a2", wcet=20, rates={"in": [8]})
        ]
        edges = [("a1", "out", "a2", "in")]
        
        # Limit individual buffer sizes
        config = BufferConfig(max_buffer_size=10)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert all(size <= 10 for size in solution.buffer_sizes.values())


class TestCSDFBufferSizing:
    """Test CSDF actor buffer sizing"""
    
    def test_csdf_edge(self):
        """Test buffer sizing for CSDF edge"""
        actors = [
            ADFGActor("csdf_prod", wcet=10, rates={"out": [1, 2, 3]}),  # Total: 6
            ADFGActor("csdf_cons", wcet=20, rates={"in": [3, 2, 1]})   # Total: 6
        ]
        edges = [("csdf_prod", "out", "csdf_cons", "in")]
        
        ilp = BufferSizingILP()
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # CSDF needs buffer for phase mismatches
        assert solution.buffer_sizes[("csdf_prod", "csdf_cons")] >= 3
    
    def test_mixed_sdf_csdf(self):
        """Test mixed SDF and CSDF actors"""
        actors = [
            ADFGActor("sdf", wcet=10, rates={"out": [4]}),           # SDF
            ADFGActor("csdf", wcet=20, rates={"in": [1, 1, 2]})     # CSDF
        ]
        edges = [("sdf", "out", "csdf", "in")]
        
        ilp = BufferSizingILP()
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # Need buffer for rate mismatch


class TestMultiBank:
    """Test multi-bank memory architecture"""
    
    def test_two_banks(self):
        """Test buffer allocation across two banks"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out1": [2], "out2": [3]}),
            ADFGActor("a2", wcet=20, rates={"in": [2]}),
            ADFGActor("a3", wcet=15, rates={"in": [3]})
        ]
        edges = [
            ("a1", "out1", "a2", "in"),
            ("a1", "out2", "a3", "in")
        ]
        
        config = BufferConfig(memory_banks=2)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # Each edge assigned to a bank
        assert len(solution.bank_assignment) == 2
        assert all(0 <= bank < 2 for bank in solution.bank_assignment.values())
    
    def test_bank_capacity(self):
        """Test bank capacity constraints"""
        actors = [
            ADFGActor("src", wcet=10, rates={f"out{i}": [5] for i in range(4)}),
        ] + [
            ADFGActor(f"sink{i}", wcet=20, rates={"in": [5]})
            for i in range(4)
        ]
        
        edges = [
            ("src", f"out{i}", f"sink{i}", "in")
            for i in range(4)
        ]
        
        # 2 banks, each can hold 10 tokens
        config = BufferConfig(memory_banks=2, bank_size=10)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        
        # Check bank capacities respected
        for bank in range(2):
            bank_edges = [e for e, b in solution.bank_assignment.items() if b == bank]
            bank_total = sum(solution.buffer_sizes[e] for e in bank_edges)
            assert bank_total <= 10


class TestInitialTokens:
    """Test initial token support"""
    
    def test_initial_tokens_allowed(self):
        """Test buffer sizing with initial tokens"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [2]}),
            ADFGActor("a2", wcet=20, rates={"in": [2]})
        ]
        edges = [("a1", "out", "a2", "in")]
        
        config = BufferConfig(allow_initial_tokens=True)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert ("a1", "a2") in solution.initial_tokens
        # Initial tokens should be >= 0
        assert solution.initial_tokens[("a1", "a2")] >= 0
    
    def test_initial_tokens_disabled(self):
        """Test buffer sizing without initial tokens"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [2]}),
            ADFGActor("a2", wcet=20, rates={"in": [2]})
        ]
        edges = [("a1", "out", "a2", "in")]
        
        config = BufferConfig(allow_initial_tokens=False)
        ilp = BufferSizingILP(config)
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        # All initial tokens should be 0
        assert all(tokens == 0 for tokens in solution.initial_tokens.values())


class TestBoundsAnalysis:
    """Test theoretical bounds analysis"""
    
    def test_analyze_bounds(self):
        """Test buffer bounds analysis"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [3]}),
            ADFGActor("a2", wcet=20, rates={"in": [2], "out": [4]}),
            ADFGActor("a3", wcet=15, rates={"in": [5]})
        ]
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        ilp = BufferSizingILP()
        bounds = ilp.analyze_bounds(actors, edges)
        
        assert "edges" in bounds
        assert "total_theoretical_min" in bounds
        assert "total_practical_min" in bounds
        
        # Check individual edge bounds
        assert ("a1", "a2") in bounds["edges"]
        edge_bound = bounds["edges"][("a1", "a2")]
        assert edge_bound["theoretical_min"] >= 3  # max(3, 2)
    
    def test_csdf_bounds(self):
        """Test bounds analysis for CSDF"""
        actors = [
            ADFGActor("csdf1", wcet=10, rates={"out": [1, 2, 3]}),
            ADFGActor("csdf2", wcet=20, rates={"in": [2, 2, 2]})
        ]
        edges = [("csdf1", "out", "csdf2", "in")]
        
        ilp = BufferSizingILP()
        bounds = ilp.analyze_bounds(actors, edges)
        
        assert bounds["csdf_edges"] == 1
        assert bounds["total_edges"] == 1
        
        # CSDF edge should have higher bound than simple max
        edge_bound = bounds["edges"][("csdf1", "csdf2")]
        assert edge_bound["theoretical_min"] >= 3  # Accounts for phase mismatch