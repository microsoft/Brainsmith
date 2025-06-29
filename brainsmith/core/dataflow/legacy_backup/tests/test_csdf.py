############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for CSDF support utilities"""

import pytest
from brainsmith.core.dataflow.adfg.csdf import (
    compute_phase_periods, csdf_hyperperiod, compute_cumulative_tokens,
    csdf_buffer_bounds, affine_relation, validate_rate_consistency,
    phase_schedule, compute_storage_distribution, analyze_throughput_bottleneck
)


class TestCSDFBasics:
    """Test basic CSDF computations"""
    
    def test_phase_periods(self):
        """Test phase period computation"""
        rates = [1, 2, 3]
        base_period = 10
        
        periods = compute_phase_periods(rates, base_period)
        assert periods == [10, 20, 30]
        
        # Empty rates
        assert compute_phase_periods([], 10) == []
    
    def test_hyperperiod(self):
        """Test hyperperiod calculation"""
        # Single actor
        assert csdf_hyperperiod([3]) == 3
        
        # Multiple actors
        assert csdf_hyperperiod([2, 3]) == 6      # LCM(2,3)
        assert csdf_hyperperiod([4, 6]) == 12     # LCM(4,6)
        assert csdf_hyperperiod([2, 3, 4]) == 12  # LCM(2,3,4)
        
        # Empty
        assert csdf_hyperperiod([]) == 1
    
    def test_cumulative_tokens(self):
        """Test cumulative token calculation"""
        rates = [2, 3, 1]
        
        # Test various firing counts
        assert compute_cumulative_tokens(rates, 1) == [2]
        assert compute_cumulative_tokens(rates, 2) == [2, 5]
        assert compute_cumulative_tokens(rates, 3) == [2, 5, 6]
        assert compute_cumulative_tokens(rates, 4) == [2, 5, 6, 8]  # Wraps to phase 0
        assert compute_cumulative_tokens(rates, 6) == [2, 5, 6, 8, 11, 12]
        
        # Empty rates
        assert compute_cumulative_tokens([], 3) == [0, 0, 0]


class TestBufferSizing:
    """Test CSDF buffer sizing"""
    
    def test_simple_buffer_bounds(self):
        """Test buffer sizing for simple CSDF"""
        # Same total rates, different patterns
        prod_rates = [3, 1]  # Total: 4
        cons_rates = [2, 2]  # Total: 4
        
        size, evolution = csdf_buffer_bounds(
            prod_rates, cons_rates,
            prod_period=10, cons_period=10
        )
        
        # Should need buffer for maximum accumulation
        assert size >= 2  # At least 2 tokens accumulate
    
    def test_unbalanced_rates(self):
        """Test buffer sizing with unbalanced rates"""
        # Producer faster in first phase
        prod_rates = [4, 1]  # Burst then slow
        cons_rates = [1, 1, 1, 1, 1]  # Steady consumption
        
        size, evolution = csdf_buffer_bounds(
            prod_rates, cons_rates,
            prod_period=5, cons_period=5
        )
        
        # Buffer must handle burst
        assert size >= 3
    
    def test_initial_tokens(self):
        """Test buffer sizing with initial tokens"""
        prod_rates = [2]
        cons_rates = [2]
        
        # With initial tokens
        size, evolution = csdf_buffer_bounds(
            prod_rates, cons_rates,
            prod_period=10, cons_period=10,
            initial_tokens=5
        )
        
        assert evolution[0] == 5  # Starts with initial tokens
    
    def test_affine_relation(self):
        """Test affine relation computation"""
        # Simple case
        n, d, phi = affine_relation(6, 4, 0)
        assert n == 3  # 6/gcd(6,4) = 6/2 = 3
        assert d == 2  # 4/gcd(6,4) = 4/2 = 2
        assert phi == 0
        
        # With initial delay
        n, d, phi = affine_relation(10, 15, 5)
        assert n == 2   # 10/gcd(10,15) = 10/5 = 2
        assert d == 3   # 15/gcd(10,15) = 15/5 = 3
        assert phi == 5
        
        # Coprime rates
        n, d, phi = affine_relation(7, 5, 0)
        assert n == 7
        assert d == 5


class TestRateConsistency:
    """Test rate consistency validation"""
    
    def test_consistent_rates(self):
        """Test validation of consistent rates"""
        edges = [
            ("a1", [2, 3], "a2", [5]),  # 5 tokens each way with reps (1,1)
            ("a2", [3], "a3", [1, 1, 1])  # 3 tokens each way with reps (1,1)
        ]
        
        repetitions = {"a1": 1, "a2": 1, "a3": 1}
        
        assert validate_rate_consistency(edges, repetitions) == True
    
    def test_inconsistent_rates(self):
        """Test detection of inconsistent rates"""
        edges = [
            ("a1", [2], "a2", [3])  # 2 != 3 with reps (1,1)
        ]
        
        repetitions = {"a1": 1, "a2": 1}
        
        assert validate_rate_consistency(edges, repetitions) == False
    
    def test_consistent_with_repetitions(self):
        """Test consistency with repetition vector"""
        edges = [
            ("a1", [3], "a2", [2])  # 3*2 = 2*3 = 6
        ]
        
        repetitions = {"a1": 2, "a2": 3}  # Balanced
        
        assert validate_rate_consistency(edges, repetitions) == True


class TestPhaseScheduling:
    """Test phase-aware scheduling"""
    
    def test_simple_phase_schedule(self):
        """Test phase scheduling for simple graph"""
        actors = ["a1", "a2", "a3"]
        phases = {"a1": 2, "a2": 1, "a3": 3}
        dependencies = [("a1", "a2"), ("a2", "a3")]
        
        schedule = phase_schedule(actors, phases, dependencies)
        
        # Check all phases are scheduled
        a1_phases = [(a, p) for a, p in schedule if a == "a1"]
        assert len(a1_phases) == 2
        assert a1_phases == [("a1", 0), ("a1", 1)]
        
        # Check dependencies are respected
        # All a1 phases before a2
        a1_end = max(i for i, (a, p) in enumerate(schedule) if a == "a1")
        a2_idx = next(i for i, (a, p) in enumerate(schedule) if a == "a2")
        assert a1_end < a2_idx
        
        # a2 before all a3 phases
        a2_idx = next(i for i, (a, p) in enumerate(schedule) if a == "a2")
        a3_start = min(i for i, (a, p) in enumerate(schedule) if a == "a3")
        assert a2_idx < a3_start
    
    def test_parallel_phases(self):
        """Test scheduling with parallel actors"""
        actors = ["a1", "a2", "a3"]
        phases = {"a1": 2, "a2": 2, "a3": 1}
        dependencies = [("a1", "a3"), ("a2", "a3")]  # a1 and a2 parallel
        
        schedule = phase_schedule(actors, phases, dependencies)
        
        # a1 and a2 can be interleaved
        # But both must complete before a3
        a3_idx = next(i for i, (a, p) in enumerate(schedule) if a == "a3")
        
        a1_done = all(i < a3_idx for i, (a, p) in enumerate(schedule) if a == "a1")
        a2_done = all(i < a3_idx for i, (a, p) in enumerate(schedule) if a == "a2")
        
        assert a1_done and a2_done


class TestMemoryAllocation:
    """Test memory allocation for buffers"""
    
    def test_storage_distribution(self):
        """Test buffer distribution across memory banks"""
        buffer_sizes = {
            ("a1", "a2"): 1000,
            ("a2", "a3"): 500,
            ("a1", "a3"): 750,
            ("a3", "a4"): 250
        }
        
        memory_limit = 1500
        
        allocation = compute_storage_distribution(buffer_sizes, memory_limit)
        
        # Check all buffers are allocated
        assert len(allocation) == len(buffer_sizes)
        
        # Check memory limits are respected
        banks = {}
        for edge, bank in allocation.items():
            if bank not in banks:
                banks[bank] = 0
            banks[bank] += buffer_sizes[edge]
        
        for bank, used in banks.items():
            assert used <= memory_limit
    
    def test_minimal_banks(self):
        """Test that allocation uses minimal banks"""
        buffer_sizes = {
            ("a1", "a2"): 500,
            ("a2", "a3"): 500,
            ("a3", "a4"): 500
        }
        
        memory_limit = 1500
        
        allocation = compute_storage_distribution(buffer_sizes, memory_limit)
        
        # All should fit in one bank
        assert len(set(allocation.values())) == 1


class TestBottleneckAnalysis:
    """Test throughput bottleneck analysis"""
    
    def test_bottleneck_identification(self):
        """Test identifying throughput bottleneck"""
        actors = {
            "a1": (100, 200),  # wcet=100, period=200, util=0.5
            "a2": (150, 200),  # wcet=150, period=200, util=0.75
            "a3": (50, 100),   # wcet=50, period=100, util=0.5
        }
        
        edges = [("a1", "a2"), ("a2", "a3")]
        
        bottleneck = analyze_throughput_bottleneck(actors, edges)
        
        # a2 has highest utilization
        assert bottleneck == "a2"
    
    def test_bottleneck_with_zero_period(self):
        """Test bottleneck with invalid period"""
        actors = {
            "a1": (100, 200),
            "a2": (100, 0),  # Invalid period
        }
        
        bottleneck = analyze_throughput_bottleneck(actors, [])
        
        # a2 has infinite utilization
        assert bottleneck == "a2"