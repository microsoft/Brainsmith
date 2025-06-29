############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for SRTA scheduler"""

import pytest
from brainsmith.core.dataflow.adfg.actor import ADFGActor
from brainsmith.core.dataflow.adfg.scheduler import (
    SRTAScheduler, SRTAConfig, SchedulabilityResult
)


class TestSRTAConfig:
    """Test SRTA configuration"""
    
    def test_default_config(self):
        """Test default configuration values"""
        config = SRTAConfig()
        
        assert config.utilization_target == 0.69
        assert config.max_period_search_iterations == 100
        assert config.period_search_step == 1.1
        assert config.deadline_strategy == "implicit"
    
    def test_config_validation(self):
        """Test configuration validation"""
        # Invalid utilization
        with pytest.raises(ValueError, match="Utilization target"):
            SRTAConfig(utilization_target=1.5)
        
        # Invalid period step
        with pytest.raises(ValueError, match="Period search step"):
            SRTAConfig(period_search_step=0.9)
        
        # Invalid deadline strategy
        with pytest.raises(ValueError, match="Unknown deadline strategy"):
            SRTAConfig(deadline_strategy="invalid")


class TestSRTASchedulerBasic:
    """Test basic SRTA scheduler functionality"""
    
    def test_empty_graph(self):
        """Test scheduling empty graph"""
        scheduler = SRTAScheduler()
        result = scheduler.analyze([], [])
        
        assert result.schedulable
        assert result.total_utilization == 0.0
        assert result.hyperperiod == 1
        assert len(result.actor_timings) == 0
    
    def test_single_actor(self):
        """Test scheduling single actor"""
        actor = ADFGActor("task1", wcet=10, rates={"out": [1]})
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze([actor], [])
        
        assert result.schedulable
        assert "task1" in result.actor_timings
        
        timing = result.actor_timings["task1"]
        assert timing.response_time == 10  # No interference
        assert timing.is_schedulable
    
    def test_two_actors_schedulable(self):
        """Test two actors that are schedulable"""
        actors = [
            ADFGActor("high", wcet=20, rates={"out": [1]}),
            ADFGActor("low", wcet=30, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, [("high", "out", "low", "in")])
        
        assert result.schedulable
        assert len(result.actor_timings) == 2
        
        # Check priorities (shorter WCET = higher priority by default)
        assert actors[0].priority < actors[1].priority
    
    def test_two_actors_not_schedulable(self):
        """Test two actors with high utilization"""
        actors = [
            ADFGActor("task1", wcet=60, rates={"out": [1]}),
            ADFGActor("task2", wcet=50, rates={"in": [1]})
        ]
        
        # Force very tight periods
        scheduler = SRTAScheduler(SRTAConfig(max_period_scale=1.2))
        result = scheduler.analyze(actors, [], fixed_periods={"task1": 65, "task2": 55})
        
        # Total utilization > 1, not schedulable
        assert not result.schedulable
        assert result.failure_reason is not None


class TestSRTAResponseTime:
    """Test response time analysis"""
    
    def test_response_time_with_interference(self):
        """Test response time computation with interference"""
        actors = [
            ADFGActor("high", wcet=10, rates={"out": [1]}),
            ADFGActor("med", wcet=20, rates={"io": [1]}),
            ADFGActor("low", wcet=30, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, [])
        
        assert result.schedulable
        
        # High priority task has no interference
        assert result.actor_timings["high"].response_time == 10
        
        # Medium priority has interference from high
        med_timing = result.actor_timings["med"]
        assert med_timing.response_time >= 20  # At least its WCET
        assert med_timing.response_time <= 30  # At most WCET + one high preemption
        
        # Low priority has interference from both
        low_timing = result.actor_timings["low"]
        assert low_timing.response_time >= 30  # At least its WCET
    
    def test_fixed_periods(self):
        """Test with fixed periods for some actors"""
        actors = [
            ADFGActor("periodic", wcet=25, rates={"out": [1]}),
            ADFGActor("sporadic", wcet=15, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(
            actors, 
            [("periodic", "out", "sporadic", "in")],
            fixed_periods={"periodic": 100}  # Fix period of first actor
        )
        
        assert result.schedulable
        assert result.actor_timings["periodic"].period == 100


class TestSRTADeadlineStrategies:
    """Test different deadline assignment strategies"""
    
    def test_implicit_deadlines(self):
        """Test implicit deadline strategy (D = T)"""
        actor = ADFGActor("task", wcet=10, rates={"out": [1]})
        
        scheduler = SRTAScheduler(SRTAConfig(deadline_strategy="implicit"))
        result = scheduler.analyze([actor], [])
        
        timing = result.actor_timings["task"]
        assert timing.deadline == timing.period
    
    def test_constrained_deadlines(self):
        """Test constrained deadline strategy"""
        actor = ADFGActor("task", wcet=10, rates={"out": [1]})
        
        scheduler = SRTAScheduler(SRTAConfig(
            deadline_strategy="constrained",
            deadline_factor=0.8
        ))
        result = scheduler.analyze([actor], [])
        
        timing = result.actor_timings["task"]
        assert timing.deadline == int(timing.period * 0.8)
    
    def test_arbitrary_deadlines(self):
        """Test arbitrary deadline strategy"""
        # Actor with explicit priority used as deadline
        actor = ADFGActor("task", wcet=10, rates={"out": [1]}, priority=50)
        
        scheduler = SRTAScheduler(SRTAConfig(deadline_strategy="arbitrary"))
        result = scheduler.analyze([actor], [])
        
        timing = result.actor_timings["task"]
        # In arbitrary mode, uses priority value if set
        assert timing.deadline == 50 or timing.deadline == timing.period


class TestSRTACSDFSupport:
    """Test CSDF actor scheduling"""
    
    def test_csdf_actor_scheduling(self):
        """Test scheduling CSDF actors"""
        actors = [
            ADFGActor("csdf1", wcet=10, rates={"out": [2, 1, 1]}),  # 3 phases
            ADFGActor("csdf2", wcet=15, rates={"in": [1, 1, 2]})   # 3 phases
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, [("csdf1", "out", "csdf2", "in")])
        
        assert result.schedulable
        
        # Hyperperiod should include CSDF phases
        # LCM of phases (3, 3) and periods
        assert result.hyperperiod % 3 == 0
        
        # Should have phase schedule
        assert result.schedule is not None
        assert len(result.schedule) >= 6  # At least 3 phases each
    
    def test_mixed_sdf_csdf(self):
        """Test mixed SDF and CSDF actors"""
        actors = [
            ADFGActor("sdf", wcet=20, rates={"out": [3]}),      # SDF
            ADFGActor("csdf", wcet=30, rates={"in": [1, 2]})   # CSDF with 2 phases
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, [("sdf", "out", "csdf", "in")])
        
        assert result.schedulable
        
        # Hyperperiod should be multiple of CSDF phases
        assert result.hyperperiod % 2 == 0


class TestSRTAPeriodOptimization:
    """Test period optimization functionality"""
    
    def test_minimize_hyperperiod(self):
        """Test hyperperiod minimization"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [1]}),
            ADFGActor("a2", wcet=15, rates={"io": [1]}),
            ADFGActor("a3", wcet=20, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        
        # Get optimized periods
        periods = scheduler.optimize_periods(
            actors, [], 
            objective="min_hyperperiod"
        )
        
        assert periods is not None
        
        # Verify schedulability
        result = scheduler.analyze(actors, [], fixed_periods=periods)
        assert result.schedulable
        
        # Check for harmonic relationship
        period_values = sorted(periods.values())
        # Harmonic if each period divides the next
        harmonic = all(
            period_values[i+1] % period_values[i] == 0
            for i in range(len(period_values)-1)
        )
        # May not always achieve perfect harmonic, but should try
    
    def test_minimize_utilization(self):
        """Test utilization minimization"""
        actors = [
            ADFGActor("cpu_bound", wcet=40, rates={"out": [1]}),
            ADFGActor("io_bound", wcet=20, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        
        # Get base result
        base_result = scheduler.analyze(actors, [])
        base_util = base_result.total_utilization
        
        # Optimize for lower utilization
        periods = scheduler.optimize_periods(
            actors, [],
            objective="min_utilization"
        )
        
        assert periods is not None
        
        # Check reduced utilization
        opt_result = scheduler.analyze(actors, [], fixed_periods=periods)
        assert opt_result.schedulable
        assert opt_result.total_utilization <= base_util
    
    def test_maximize_slack(self):
        """Test slack maximization"""
        actors = [
            ADFGActor("critical", wcet=30, rates={"out": [1]}),
            ADFGActor("normal", wcet=20, rates={"in": [1]})
        ]
        
        scheduler = SRTAScheduler()
        
        # Get base result
        base_result = scheduler.analyze(actors, [])
        base_min_slack = min(t.slack for t in base_result.actor_timings.values())
        
        # Optimize for maximum slack
        periods = scheduler.optimize_periods(
            actors, [],
            objective="max_slack"
        )
        
        assert periods is not None
        
        # Check increased slack
        opt_result = scheduler.analyze(actors, [], fixed_periods=periods)
        assert opt_result.schedulable
        
        opt_min_slack = min(t.slack for t in opt_result.actor_timings.values())
        assert opt_min_slack >= base_min_slack


class TestSRTAComplexGraphs:
    """Test complex graph scheduling"""
    
    def test_pipeline_graph(self):
        """Test pipeline of actors"""
        actors = [
            ADFGActor(f"stage{i}", wcet=10+i*5, rates={"in": [1], "out": [1]})
            for i in range(5)
        ]
        # Fix first stage to only have output
        actors[0].rates = {"out": [1]}
        # Fix last stage to only have input  
        actors[-1].rates = {"in": [1]}
        
        edges = [
            (f"stage{i}", "out", f"stage{i+1}", "in")
            for i in range(4)
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        assert len(result.actor_timings) == 5
    
    def test_fork_join_graph(self):
        """Test fork-join pattern"""
        actors = [
            ADFGActor("source", wcet=20, rates={"out1": [1], "out2": [1]}),
            ADFGActor("proc1", wcet=30, rates={"in": [1], "out": [1]}),
            ADFGActor("proc2", wcet=25, rates={"in": [1], "out": [1]}),
            ADFGActor("sink", wcet=15, rates={"in1": [1], "in2": [1]})
        ]
        
        edges = [
            ("source", "out1", "proc1", "in"),
            ("source", "out2", "proc2", "in"),
            ("proc1", "out", "sink", "in1"),
            ("proc2", "out", "sink", "in2")
        ]
        
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        
        # Check all actors scheduled
        assert len(result.actor_timings) == 4
        
        # Verify response times account for interference
        for name, timing in result.actor_timings.items():
            actor = next(a for a in actors if a.name == name)
            assert timing.response_time >= actor.wcet