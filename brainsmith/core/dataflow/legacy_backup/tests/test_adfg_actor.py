############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Tests for ADFG actor abstraction"""

import pytest
from brainsmith.core.dataflow.types import InterfaceDirection, INT16
from brainsmith.core.dataflow.interface import Interface
from brainsmith.core.dataflow.kernel import Kernel
from brainsmith.core.dataflow.adfg.actor import (
    ADFGActor, ActorTiming, validate_actor_graph, compute_repetition_vector
)


class TestADFGActorCreation:
    """Test ADFG actor creation and validation"""
    
    def test_basic_actor_creation(self):
        """Test basic actor creation"""
        actor = ADFGActor(
            name="test",
            wcet=100,
            rates={"in": [1], "out": [1]},
            priority=10
        )
        
        assert actor.name == "test"
        assert actor.wcet == 100
        assert actor.rates["in"] == [1]
        assert actor.priority == 10
        assert not actor.is_csdf
    
    def test_csdf_actor_creation(self):
        """Test CSDF actor creation"""
        actor = ADFGActor(
            name="csdf_test",
            wcet=50,
            rates={
                "in": [2, 1, 1],  # CSDF pattern
                "out": [1, 1, 2]
            }
        )
        
        assert actor.is_csdf
        assert actor.n_phases == 3
        assert actor.is_consistent  # Same phases for all interfaces
    
    def test_validation_errors(self):
        """Test actor validation"""
        # Invalid WCET
        with pytest.raises(ValueError, match="WCET must be positive"):
            ADFGActor("bad", wcet=0, rates={"in": [1]})
        
        # Empty rate pattern
        with pytest.raises(ValueError, match="Empty rate pattern"):
            ADFGActor("bad", wcet=10, rates={"in": []})
        
        # Negative rate
        with pytest.raises(ValueError, match="Negative rate"):
            ADFGActor("bad", wcet=10, rates={"in": [1, -1]})
    
    def test_from_kernel_conversion(self):
        """Test conversion from Kernel to ADFGActor"""
        kernel = Kernel(
            name="matmul",
            interfaces=[
                Interface("vec", InterfaceDirection.INPUT, INT16, (512,), (64,), (16,)),
                Interface("mat", InterfaceDirection.WEIGHT, INT16, (256, 512), (32, 64), (8, 16)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (256,), (32,), (8,)),
                Interface("cfg", InterfaceDirection.CONFIG, INT16, (4,), (4,), (4,))
            ],
            latency_cycles=(1000, 800),
            priming_cycles=50,
            flush_cycles=25
        )
        
        # Convert without config
        actor = ADFGActor.from_kernel(kernel)
        
        assert actor.name == "matmul"
        assert actor.wcet == 1000  # Worst-case
        assert actor.priming_cycles == 50
        assert actor.flush_cycles == 25
        assert "cfg" not in actor.rates  # Config excluded
        assert "vec" in actor.rates
        
        # Convert with config
        actor_with_cfg = ADFGActor.from_kernel(kernel, include_config=True)
        assert "cfg" in actor_with_cfg.rates


class TestADFGActorProperties:
    """Test ADFG actor properties and methods"""
    
    def test_phase_consistency(self):
        """Test phase consistency checking"""
        # Consistent phases
        actor = ADFGActor(
            name="consistent",
            wcet=100,
            rates={
                "in1": [1, 2, 3],
                "in2": [3, 2, 1],
                "out": [2, 2, 2]
            }
        )
        assert actor.is_consistent
        assert actor.n_phases == 3
        
        # Inconsistent phases
        actor = ADFGActor(
            name="inconsistent",
            wcet=100,
            rates={
                "in": [1, 2],      # 2 phases
                "out": [1, 1, 1]   # 3 phases
            }
        )
        assert not actor.is_consistent
        assert actor.n_phases == 3  # Maximum
    
    def test_get_rate(self):
        """Test rate retrieval for specific phases"""
        actor = ADFGActor(
            name="test",
            wcet=100,
            rates={
                "in": [2, 4, 6],
                "out": [3]
            }
        )
        
        # Normal phase access
        assert actor.get_rate("in", 0) == 2
        assert actor.get_rate("in", 1) == 4
        assert actor.get_rate("in", 2) == 6
        
        # Wrap-around for shorter patterns
        assert actor.get_rate("out", 0) == 3
        assert actor.get_rate("out", 1) == 3  # Wraps to phase 0
        assert actor.get_rate("out", 5) == 3  # Wraps to phase 2 % 1 = 0
        
        # Non-existent interface
        assert actor.get_rate("missing", 0) == 0
    
    def test_total_tokens(self):
        """Test total token calculation"""
        actor = ADFGActor(
            name="test",
            wcet=100,
            rates={"data": [2, 3, 1]}  # Pattern sums to 6
        )
        
        # Complete cycles
        assert actor.total_tokens("data", 3) == 6    # One complete cycle
        assert actor.total_tokens("data", 6) == 12   # Two complete cycles
        
        # Partial cycles
        assert actor.total_tokens("data", 1) == 2    # Just first phase
        assert actor.total_tokens("data", 2) == 5    # First two phases
        assert actor.total_tokens("data", 4) == 8    # One cycle + first phase
        
        # Non-existent interface
        assert actor.total_tokens("missing", 10) == 0
    
    def test_repetition_count(self):
        """Test repetition count calculation"""
        # Non-CSDF actor
        actor = ADFGActor("simple", wcet=10, rates={"in": [1]})
        assert actor.repetition_count(5) == 5
        
        # CSDF actor
        actor = ADFGActor("csdf", wcet=10, rates={"in": [1, 2, 3]})
        assert actor.repetition_count(1) == 3   # LCM(1, 3) = 3
        assert actor.repetition_count(2) == 6   # LCM(2, 3) = 6
        assert actor.repetition_count(3) == 3   # LCM(3, 3) = 3
        assert actor.repetition_count(4) == 12  # LCM(4, 3) = 12
    
    def test_utilization(self):
        """Test utilization calculation"""
        actor = ADFGActor("test", wcet=100, rates={"in": [1]})
        
        assert actor.utilization(100) == 1.0   # 100%
        assert actor.utilization(200) == 0.5   # 50%
        assert actor.utilization(1000) == 0.1  # 10%
        
        # Edge case
        assert actor.utilization(0) == float('inf')


class TestActorTiming:
    """Test actor timing analysis"""
    
    def test_timing_creation(self):
        """Test ActorTiming creation"""
        timing = ActorTiming(
            actor_name="test",
            period=100,
            deadline=100,
            response_time=80,
            start_time=0,
            utilization=0.8
        )
        
        assert timing.is_schedulable  # 80 <= 100
        assert timing.slack == 20      # 100 - 80
    
    def test_schedulability(self):
        """Test schedulability checking"""
        # Schedulable
        timing = ActorTiming("test", 100, 100, 90, 0, 0.9)
        assert timing.is_schedulable
        
        # Not schedulable
        timing = ActorTiming("test", 100, 100, 110, 0, 1.1)
        assert not timing.is_schedulable
        assert timing.slack == -10


class TestActorGraphValidation:
    """Test actor graph validation"""
    
    def test_valid_graph(self):
        """Test validation of valid actor graph"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [2]}),
            ADFGActor("a2", wcet=20, rates={"in": [1], "out": [1]}),
            ADFGActor("a3", wcet=15, rates={"in": [2]})
        ]
        
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        # Should not raise
        validate_actor_graph(actors, edges)
    
    def test_invalid_actor_reference(self):
        """Test validation with invalid actor references"""
        actors = [ADFGActor("a1", wcet=10, rates={"out": [1]})]
        
        # Non-existent producer
        with pytest.raises(ValueError, match="Producer actor 'missing' not found"):
            validate_actor_graph(actors, [("missing", "out", "a1", "in")])
        
        # Non-existent consumer
        with pytest.raises(ValueError, match="Consumer actor 'missing' not found"):
            validate_actor_graph(actors, [("a1", "out", "missing", "in")])
    
    def test_invalid_interface_reference(self):
        """Test validation with invalid interface references"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [1]}),
            ADFGActor("a2", wcet=10, rates={"in": [1]})
        ]
        
        # Non-existent producer interface
        with pytest.raises(ValueError, match="Interface 'wrong' not found"):
            validate_actor_graph(actors, [("a1", "wrong", "a2", "in")])
        
        # Non-existent consumer interface
        with pytest.raises(ValueError, match="Interface 'wrong' not found"):
            validate_actor_graph(actors, [("a1", "out", "a2", "wrong")])
    
    def test_inconsistent_csdf_phases(self):
        """Test validation of inconsistent CSDF phases"""
        actors = [
            ADFGActor("bad", wcet=10, rates={
                "in": [1, 2],    # 2 phases
                "out": [1, 1, 1] # 3 phases - inconsistent!
            })
        ]
        
        with pytest.raises(ValueError, match="inconsistent CSDF phases"):
            validate_actor_graph(actors, [])


class TestRepetitionVector:
    """Test repetition vector computation"""
    
    def test_simple_repetition_vector(self):
        """Test repetition vector for simple graph"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [3]}),
            ADFGActor("a2", wcet=20, rates={"in": [2]})
        ]
        
        edges = [("a1", "out", "a2", "in")]
        
        # a1 fires 2 times (produces 6), a2 fires 3 times (consumes 6)
        reps = compute_repetition_vector(actors, edges)
        assert reps["a1"] == 2
        assert reps["a2"] == 3
    
    def test_csdf_repetition_vector(self):
        """Test repetition vector for CSDF graph"""
        actors = [
            ADFGActor("prod", wcet=10, rates={"out": [1, 2]}),  # Total: 3
            ADFGActor("cons", wcet=20, rates={"in": [2, 1]})   # Total: 3
        ]
        
        edges = [("prod", "out", "cons", "in")]
        
        # Both actors have same total rate, so 1:1
        reps = compute_repetition_vector(actors, edges)
        assert reps["prod"] == 1
        assert reps["cons"] == 1
    
    def test_complex_repetition_vector(self):
        """Test repetition vector for complex graph"""
        actors = [
            ADFGActor("a1", wcet=10, rates={"out": [4]}),
            ADFGActor("a2", wcet=20, rates={"in": [3], "out": [2]}),
            ADFGActor("a3", wcet=30, rates={"in": [5]})
        ]
        
        edges = [
            ("a1", "out", "a2", "in"),
            ("a2", "out", "a3", "in")
        ]
        
        reps = compute_repetition_vector(actors, edges)
        
        # Verify rate balance
        # a1 -> a2: 4 * reps[a1] = 3 * reps[a2]
        assert 4 * reps["a1"] == 3 * reps["a2"]
        
        # a2 -> a3: 2 * reps[a2] = 5 * reps[a3]
        assert 2 * reps["a2"] == 5 * reps["a3"]