############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Integration tests for ADFG components"""

import pytest
from brainsmith.core.dataflow.core.types import INT16, InterfaceDirection
from brainsmith.core.dataflow.core.interface import Interface
from brainsmith.core.dataflow.core.kernel import Kernel
from brainsmith.core.dataflow.core.pragma import TiePragma, ConstrPragma
from brainsmith.core.dataflow.core.graph import DataflowGraph, DataflowEdge

from brainsmith.core.dataflow.adfg import (
    ADFGActor, compute_repetition_vector, SRTAScheduler, 
    csdf_buffer_bounds, phase_schedule
)

# Try importing buffer ILP components
try:
    from brainsmith.core.dataflow.adfg import BufferSizingILP, BufferConfig
    from brainsmith.core.dataflow.adfg.buffer_ilp import HAS_PULP
    HAS_BUFFER_ILP = HAS_PULP
except ImportError:
    HAS_BUFFER_ILP = False


class TestKernelToADFGWorkflow:
    """Test complete workflow from kernel definition to ADFG scheduling"""
    
    def test_simple_pipeline(self):
        """Test simple pipeline from kernels to scheduling"""
        # Define kernels
        k1 = Kernel(
            name="producer",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, (256,), (32,))
            ],
            latency_cycles=(100, 80)
        )
        
        k2 = Kernel(
            name="processor",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (256,), (32,)),
                Interface("out", InterfaceDirection.OUTPUT, INT16, (128,), (16,))
            ],
            latency_cycles=(200, 150)
        )
        
        k3 = Kernel(
            name="consumer",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16, (128,), (16,))
            ],
            latency_cycles=(50, 40)
        )
        
        # Create dataflow graph
        graph = DataflowGraph()
        
        # Need to rename kernels before adding
        k1.name = "prod"
        k2.name = "proc"
        k3.name = "cons"
        
        graph.add_kernel(k1)
        graph.add_kernel(k2)
        graph.add_kernel(k3)
        
        graph.add_edge(
            producer_kernel="prod",
            producer_intf="out",
            consumer_kernel="proc",
            consumer_intf="in"
        )
        
        graph.add_edge(
            producer_kernel="proc",
            producer_intf="out",
            consumer_kernel="cons",
            consumer_intf="in"
        )
        
        # Validate graph
        graph.validate()
        
        # Convert to ADFG actors
        actors = []
        for name, kernel in graph.kernels.items():
            actor = ADFGActor.from_kernel(kernel)
            actor.name = name  # Use instance name
            actors.append(actor)
        
        # Extract edges in ADFG format
        edges = []
        for edge in graph.edges.values():
            edges.append((
                edge.producer_kernel,
                edge.producer_intf,
                edge.consumer_kernel,
                edge.consumer_intf
            ))
        
        # Schedule with SRTA
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        assert len(result.actor_timings) == 3
        assert result.total_utilization <= 1.0  # May be exactly 1.0 if tightly scheduled
    
    def test_csdf_pipeline(self):
        """Test CSDF pipeline scheduling"""
        # Define CSDF kernels
        k1 = Kernel(
            name="csdf_source",
            interfaces=[
                Interface("out", InterfaceDirection.OUTPUT, INT16, 
                         (256,), [(32,), (64,), (32,)])  # CSDF pattern
            ],
            latency_cycles=(100, 80)
        )
        
        k2 = Kernel(
            name="csdf_sink",
            interfaces=[
                Interface("in", InterfaceDirection.INPUT, INT16,
                         (256,), [(64,), (32,), (32,)])  # Matching CSDF
            ],
            latency_cycles=(150, 120)
        )
        
        # Convert to actors
        a1 = ADFGActor.from_kernel(k1)
        a2 = ADFGActor.from_kernel(k2)
        
        actors = [a1, a2]
        edges = [("csdf_source", "out", "csdf_sink", "in")]
        
        # Check CSDF properties
        assert a1.is_csdf
        assert a2.is_csdf
        assert a1.n_phases == 3
        
        # Compute repetition vector
        reps = compute_repetition_vector(actors, edges)
        
        # Both actors produce/consume same total tokens
        # a1: [1, 2, 1] total=4, a2: [2, 1, 1] total=4
        assert reps[a1.name] == reps[a2.name]
        
        # Schedule
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        assert result.schedule is not None  # Has phase schedule
        
        # Check phase schedule
        phases = {a.name: a.n_phases for a in actors}
        dependencies = [("csdf_source", "csdf_sink")]
        schedule = phase_schedule([a.name for a in actors], phases, dependencies)
        
        # Should have 6 entries (3 phases each)
        assert len(schedule) == 6


class TestMatrixMultiplyExample:
    """Test realistic matrix multiply accelerator"""
    
    def create_matmul_kernel(self, pe_count: int = 16):
        """Create parameterized matrix multiply kernel"""
        return Kernel(
            name="matmul_pe",
            interfaces=[
                Interface("vec_in", InterfaceDirection.INPUT, INT16, 
                         (512,), (64,), (pe_count,)),
                Interface("mat_in", InterfaceDirection.WEIGHT, INT16,
                         (256, 512), (pe_count, 64), (pe_count, pe_count)),
                Interface("vec_out", InterfaceDirection.OUTPUT, INT16,
                         (256,), (pe_count,), (pe_count,))
            ],
            latency_cycles=(1000, 800),
            pragmas=[
                ConstrPragma(expr="vec_in[2]", op="%", value="PE"),
                ConstrPragma(expr="mat_in[1]", op="%", value="PE"),
                ConstrPragma(expr="vec_out[1]", op="%", value="PE")
            ],
            resources={"DSP": pe_count * 4, "BRAM": pe_count},
            pragma_env={"PE": pe_count}
        )
    
    def test_matmul_scheduling(self):
        """Test scheduling of matrix multiply pipeline"""
        # Create pipeline: Load -> MatMul -> Store
        load_kernel = Kernel(
            name="dma_load",
            interfaces=[
                Interface("dram", InterfaceDirection.INPUT, INT16, (512,), (512,)),
                Interface("stream", InterfaceDirection.OUTPUT, INT16, (512,), (64,))
            ],
            latency_cycles=(64, 64)  # One cycle per element
        )
        
        matmul = self.create_matmul_kernel(16)
        
        store_kernel = Kernel(
            name="dma_store", 
            interfaces=[
                Interface("stream", InterfaceDirection.INPUT, INT16, (256,), (16,)),
                Interface("dram", InterfaceDirection.OUTPUT, INT16, (256,), (256,))
            ],
            latency_cycles=(16, 16)
        )
        
        # Convert to actors
        actors = [
            ADFGActor.from_kernel(load_kernel),
            ADFGActor.from_kernel(matmul),
            ADFGActor.from_kernel(store_kernel)
        ]
        
        edges = [
            ("dma_load", "stream", "matmul_pe", "vec_in"),
            ("matmul_pe", "vec_out", "dma_store", "stream")
        ]
        
        # Schedule
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        
        # Optimize for minimum hyperperiod
        opt_periods = scheduler.optimize_periods(actors, edges, "min_hyperperiod")
        assert opt_periods is not None
        
        # Verify optimized solution
        opt_result = scheduler.analyze(actors, edges, opt_periods)
        assert opt_result.schedulable
    
    @pytest.mark.skipif(not HAS_BUFFER_ILP, reason="PuLP not installed")
    def test_matmul_buffer_sizing(self):
        """Test buffer sizing for matrix multiply"""
        # Simplified version for buffer sizing
        actors = [
            ADFGActor("load", wcet=64, rates={"out": [8]}),    # Load 8 elements
            ADFGActor("compute", wcet=100, rates={"in": [8], "out": [4]}),  # Process 8, produce 4
            ADFGActor("store", wcet=16, rates={"in": [4]})     # Store 4 elements
        ]
        
        edges = [
            ("load", "out", "compute", "in"),
            ("compute", "out", "store", "in")
        ]
        
        # Size buffers
        ilp = BufferSizingILP(BufferConfig(objective="min_total"))
        solution = ilp.solve(actors, edges)
        
        assert solution.feasible
        assert solution.total_memory >= 12  # At least 8 + 4
        
        # Check repetition vector balance
        reps = compute_repetition_vector(actors, edges)
        
        # Verify rate consistency
        assert 8 * reps["load"] == 8 * reps["compute"]
        assert 4 * reps["compute"] == 4 * reps["store"]


class TestStreamJoinPattern:
    """Test stream join/merge patterns"""
    
    def test_two_stream_join(self):
        """Test joining two streams"""
        # Two producers with different rates
        prod1 = ADFGActor("prod1", wcet=50, rates={"out": [3]})
        prod2 = ADFGActor("prod2", wcet=75, rates={"out": [2]})
        
        # Consumer that needs both
        consumer = ADFGActor("join", wcet=100, rates={"in1": [3], "in2": [2]})
        
        actors = [prod1, prod2, consumer]
        edges = [
            ("prod1", "out", "join", "in1"),
            ("prod2", "out", "join", "in2")
        ]
        
        # Schedule
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        
        # Check repetition vector
        reps = compute_repetition_vector(actors, edges)
        
        # Each producer fires once per period
        assert reps["prod1"] == 1
        assert reps["prod2"] == 1
        assert reps["join"] == 1
    
    def test_asymmetric_fork_join(self):
        """Test asymmetric fork-join pattern"""
        # Source splits data unevenly
        source = ADFGActor("split", wcet=100, rates={"out1": [5], "out2": [3]})
        
        # Two paths with different processing
        path1 = ADFGActor("path1", wcet=200, rates={"in": [5], "out": [5]})
        path2 = ADFGActor("path2", wcet=150, rates={"in": [3], "out": [3]})
        
        # Sink joins them back
        sink = ADFGActor("merge", wcet=80, rates={"in1": [5], "in2": [3]})
        
        actors = [source, path1, path2, sink]
        edges = [
            ("split", "out1", "path1", "in"),
            ("split", "out2", "path2", "in"),
            ("path1", "out", "merge", "in1"),
            ("path2", "out", "merge", "in2")
        ]
        
        # This should be schedulable
        scheduler = SRTAScheduler()
        result = scheduler.analyze(actors, edges)
        
        assert result.schedulable
        
        # All actors should fire once per period
        reps = compute_repetition_vector(actors, edges)
        assert all(r == 1 for r in reps.values())


class TestCSDFBufferAnalysis:
    """Test CSDF-specific buffer analysis"""
    
    def test_ragged_tiling_buffer(self):
        """Test buffer sizing for ragged tiling pattern"""
        # Ragged producer: alternates between small and large tiles
        prod_rates = [16, 48, 16, 48]  # Total: 128
        cons_rates = [32, 32, 32, 32]  # Total: 128, steady
        
        # Calculate buffer bounds
        buffer_size, evolution = csdf_buffer_bounds(
            prod_rates, cons_rates,
            prod_period=10, cons_period=10
        )
        
        # Buffer must handle accumulation from uneven production
        assert buffer_size >= 32  # At least one large burst
        
        # Check token evolution makes sense
        assert evolution[0] == 0  # Starts empty
        assert max(evolution) - min(evolution) == buffer_size
    
    def test_phased_processing(self):
        """Test phased processing pattern"""
        # Three-phase processing: load, compute, store
        actors = [
            ADFGActor("phased", wcet=100, rates={
                "in": [64, 0, 0],    # Load phase only
                "out": [0, 0, 32]    # Store phase only
            })
        ]
        
        # This actor loads in phase 0, processes in phase 1, stores in phase 2
        assert actors[0].is_csdf
        assert actors[0].n_phases == 3
        
        # Phase-specific rates
        assert actors[0].get_rate("in", 0) == 64   # Load phase
        assert actors[0].get_rate("in", 1) == 0    # Compute phase
        assert actors[0].get_rate("out", 2) == 32  # Store phase


class TestPerformanceBottlenecks:
    """Test identification of performance bottlenecks"""
    
    def test_utilization_bottleneck(self):
        """Test finding utilization bottleneck"""
        from brainsmith.core.dataflow.adfg import analyze_throughput_bottleneck
        
        actors_timing = {
            "dma": (100, 200),    # 50% utilization
            "compute": (180, 200), # 90% utilization - bottleneck
            "store": (50, 200)     # 25% utilization
        }
        
        bottleneck = analyze_throughput_bottleneck(actors_timing, [])
        assert bottleneck == "compute"
    
    def test_schedule_with_bottleneck(self):
        """Test scheduling with known bottleneck"""
        # Create pipeline with obvious bottleneck
        actors = [
            ADFGActor("fast1", wcet=10, rates={"out": [1]}),
            ADFGActor("slow", wcet=90, rates={"in": [1], "out": [1]}),  # Bottleneck
            ADFGActor("fast2", wcet=10, rates={"in": [1]})
        ]
        
        edges = [
            ("fast1", "out", "slow", "in"),
            ("slow", "out", "fast2", "in")
        ]
        
        scheduler = SRTAScheduler()
        
        # Try different optimization objectives
        for objective in ["min_utilization", "max_slack"]:
            periods = scheduler.optimize_periods(actors, edges, objective)
            assert periods is not None
            
            result = scheduler.analyze(actors, edges, periods)
            assert result.schedulable
            
            # Slow actor should have reasonable utilization
            slow_timing = result.actor_timings["slow"]
            assert slow_timing.utilization < 0.95  # Not overloaded


class TestMemoryBankAllocation:
    """Test memory bank allocation strategies"""
    
    @pytest.mark.skipif(not HAS_BUFFER_ILP, reason="PuLP not installed")
    def test_dual_bank_allocation(self):
        """Test allocation across dual memory banks"""
        from brainsmith.core.dataflow.adfg import compute_storage_distribution
        
        # Multiple buffers of varying sizes
        buffer_sizes = {
            ("a1", "a2"): 1024,
            ("a2", "a3"): 512,
            ("a1", "a4"): 768,
            ("a3", "a5"): 256,
            ("a4", "a5"): 384
        }
        
        # Each bank can hold 1536 bytes
        allocation = compute_storage_distribution(buffer_sizes, memory_limit=1536)
        
        # Verify all buffers allocated
        assert len(allocation) == len(buffer_sizes)
        
        # Verify bank limits respected
        banks_used = {}
        for edge, bank in allocation.items():
            if bank not in banks_used:
                banks_used[bank] = 0
            banks_used[bank] += buffer_sizes[edge]
        
        for bank, used in banks_used.items():
            assert used <= 1536
        
        # Should need at least 2 banks (total = 2920)
        assert len(banks_used) >= 2