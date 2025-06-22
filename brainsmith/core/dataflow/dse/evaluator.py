############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Performance evaluation for dataflow configurations"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math
from ..core.graph import DataflowGraph
from ..core.kernel import Kernel
from ..core.types import InterfaceDirection, prod
from ..adfg.actor import ADFGActor
from ..adfg.scheduler import SchedulabilityResult


@dataclass
class PerformanceMetrics:
    """Performance metrics for a dataflow configuration
    
    Captures key performance indicators including throughput,
    latency, and resource utilization.
    """
    
    # Core performance
    throughput: float  # inferences/sec
    latency: int  # cycles
    fps: float  # frames/sec (accounting for sparsity)
    
    # Resource usage
    resource_usage: Dict[str, float] = field(default_factory=dict)
    
    # Power estimate
    power_estimate: float = 0.0  # watts
    
    # Utilization metrics
    processor_utilization: float = 0.0  # [0, 1]
    memory_bandwidth_utilization: float = 0.0  # [0, 1]
    
    # Scheduling details
    hyperperiod: Optional[int] = None
    actor_periods: Dict[str, int] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return (f"PerformanceMetrics(throughput={self.throughput:.1f} inf/s, "
                f"latency={self.latency} cycles, fps={self.fps:.1f})")


class PerformanceEvaluator:
    """Evaluate performance of dataflow configurations
    
    Computes performance metrics for configured dataflow graphs,
    accounting for scheduling, pipelining, and sparsity effects.
    """
    
    def __init__(self, frequency_mhz: float = 200.0):
        """Initialize evaluator
        
        Args:
            frequency_mhz: Target clock frequency in MHz
        """
        self.frequency_mhz = frequency_mhz
        self.frequency_hz = frequency_mhz * 1e6
    
    def evaluate(self, graph: DataflowGraph,
                schedule: Optional[SchedulabilityResult] = None,
                batch_size: int = 1,
                input_sparsity: Optional[Dict[str, float]] = None) -> PerformanceMetrics:
        """Compute performance metrics for configuration
        
        Args:
            graph: Configured dataflow graph
            schedule: Optional scheduling result (will compute if not provided)
            batch_size: Batch size for throughput calculation
            input_sparsity: Optional sparsity per input interface
            
        Returns:
            Performance metrics
        """
        # Get scheduling result if not provided
        if schedule is None:
            from ..adfg.scheduler import SRTAScheduler
            actors = self._create_actors(graph)
            edges = self._extract_edges(graph)
            scheduler = SRTAScheduler()
            schedule = scheduler.analyze(actors, edges)
            
            if not schedule.schedulable:
                # Return worst-case metrics for unschedulable config
                return PerformanceMetrics(
                    throughput=0.0,
                    latency=float('inf'),
                    fps=0.0,
                    resource_usage=self._estimate_resources(graph),
                    processor_utilization=2.0,  # > 1 indicates overload
                    power_estimate=self._estimate_power(graph, schedule, self._estimate_resources(graph))
                )
        
        # Check if provided schedule is unschedulable
        if schedule and not schedule.schedulable:
            return PerformanceMetrics(
                throughput=0.0,
                latency=float('inf'),
                fps=0.0,
                resource_usage=self._estimate_resources(graph),
                processor_utilization=schedule.total_utilization,
                power_estimate=self._estimate_power(graph, schedule, self._estimate_resources(graph))
            )
        
        # Core performance metrics
        throughput = self._compute_throughput(schedule, batch_size)
        latency = self._compute_latency(graph, schedule, batch_size)
        
        # Apply sparsity effects
        effective_throughput = self._apply_sparsity(graph, throughput, input_sparsity)
        
        # Resource usage
        resources = self._estimate_resources(graph)
        
        # Power estimate
        power = self._estimate_power(graph, schedule, resources)
        
        # Utilization metrics
        proc_util = schedule.total_utilization if schedule else 0.0
        mem_bw_util = self._compute_bandwidth_utilization(graph, schedule)
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            fps=effective_throughput,
            resource_usage=resources,
            power_estimate=power,
            processor_utilization=proc_util,
            memory_bandwidth_utilization=mem_bw_util,
            hyperperiod=schedule.hyperperiod if schedule else None,
            actor_periods={name: timing.period 
                          for name, timing in schedule.actor_timings.items()} if schedule else {}
        )
    
    def _create_actors(self, graph: DataflowGraph) -> List[ADFGActor]:
        """Create ADFG actors from graph kernels"""
        actors = []
        for kernel_name, kernel in graph.kernels.items():
            actor = ADFGActor.from_kernel(kernel)
            actor.name = kernel_name  # Use instance name
            actors.append(actor)
        return actors
    
    def _extract_edges(self, graph: DataflowGraph) -> List[Tuple[str, str, str, str]]:
        """Extract edges in ADFG format"""
        edges = []
        for edge in graph.edges.values():
            edges.append((
                edge.producer_kernel,
                edge.producer_intf,
                edge.consumer_kernel,
                edge.consumer_intf
            ))
        return edges
    
    def _compute_throughput(self, schedule: SchedulabilityResult, 
                           batch_size: int) -> float:
        """Calculate throughput in inferences/second
        
        Throughput = batch_size * frequency / hyperperiod
        """
        if schedule.hyperperiod == 0:
            return 0.0
        
        cycles_per_batch = schedule.hyperperiod
        batches_per_second = self.frequency_hz / cycles_per_batch
        
        return batches_per_second * batch_size
    
    def _compute_latency(self, graph: DataflowGraph,
                        schedule: SchedulabilityResult,
                        batch_size: int) -> int:
        """Calculate end-to-end latency in cycles
        
        Includes:
        - Priming cycles for pipeline fill
        - Execution cycles for batch
        - Flush cycles for pipeline drain
        """
        total_cycles = 0
        
        # Add priming cycles (pipeline fill)
        for kernel in graph.kernels.values():
            total_cycles += kernel.priming_cycles
        
        # Add execution cycles
        # For pipelined execution, latency ≈ critical path + (batch-1) * II
        critical_path_cycles = self._compute_critical_path_latency(graph)
        
        if batch_size > 1:
            # Pipelined execution
            ii_cycles = schedule.hyperperiod  # Initiation interval
            total_cycles += critical_path_cycles + (batch_size - 1) * ii_cycles
        else:
            # Single inference
            total_cycles += critical_path_cycles
        
        # Add flush cycles (pipeline drain)
        for kernel in graph.kernels.values():
            total_cycles += kernel.flush_cycles
        
        return total_cycles
    
    def _compute_critical_path_latency(self, graph: DataflowGraph) -> int:
        """Compute critical path latency through graph"""
        # Use graph's critical path analysis
        path, latency = graph.get_critical_path()
        return latency
    
    def _apply_sparsity(self, graph: DataflowGraph,
                       base_throughput: float,
                       input_sparsity: Optional[Dict[str, float]]) -> float:
        """Apply sparsity effects to throughput
        
        If inputs have sparsity (skip probability), effective throughput
        increases as some computations are skipped.
        """
        if not input_sparsity:
            return base_throughput
        
        # Find minimum density (1 - sparsity) across critical inputs
        min_density = 1.0
        
        for kernel in graph.kernels.values():
            for intf in kernel.input_interfaces:
                if intf.name in input_sparsity:
                    density = 1.0 - input_sparsity[intf.name]
                    min_density = min(min_density, density)
                elif intf.skip_prob:
                    # Use interface's inherent skip probability
                    avg_density = 1.0 - sum(intf.skip_prob) / len(intf.skip_prob)
                    min_density = min(min_density, avg_density)
        
        # Effective throughput increases with sparsity
        if min_density > 0:
            return base_throughput / min_density
        else:
            return float('inf')  # All computation skipped
    
    def _estimate_resources(self, graph: DataflowGraph) -> Dict[str, float]:
        """Estimate total resource usage
        
        Aggregates resources across all kernels, accounting for
        parallelism and replication.
        """
        total_resources = {}
        
        for kernel_name, kernel in graph.kernels.items():
            # Get kernel resources
            kernel_resources = kernel.estimate_resources()
            
            # Aggregate
            for resource, amount in kernel_resources.items():
                if resource not in total_resources:
                    total_resources[resource] = 0
                total_resources[resource] += amount
        
        # Add graph-level resources
        
        # Total bandwidth across all edges
        total_bandwidth_bits = 0
        for kernel in graph.kernels.values():
            bw_reqs = kernel.bandwidth_requirements()
            total_bandwidth_bits += sum(bw_reqs.values())
        
        total_resources["bandwidth_gbps"] = (
            total_bandwidth_bits * self.frequency_mhz * 1e6 / 1e9
        )
        
        return total_resources
    
    def _estimate_power(self, graph: DataflowGraph,
                       schedule: SchedulabilityResult,
                       resources: Dict[str, float]) -> float:
        """Estimate power consumption
        
        Simple model based on resource usage and utilization.
        """
        # Base power consumption
        static_power = 5.0  # watts
        
        # Dynamic power based on resources
        dsp_power = resources.get("DSP", 0) * 0.02  # 20mW per DSP
        bram_power = resources.get("BRAM", 0) * 0.05  # 50mW per BRAM
        lut_power = resources.get("LUT", 0) * 0.00001  # 10uW per LUT
        
        # Activity factor based on utilization
        activity_factor = schedule.total_utilization if schedule else 1.0
        
        dynamic_power = (dsp_power + bram_power + lut_power) * activity_factor
        
        # Memory bandwidth power
        bandwidth_power = resources.get("bandwidth_gbps", 0) * 0.5  # 0.5W per GB/s
        
        return static_power + dynamic_power + bandwidth_power
    
    def _compute_bandwidth_utilization(self, graph: DataflowGraph,
                                     schedule: SchedulabilityResult) -> float:
        """Compute memory bandwidth utilization
        
        Compares required bandwidth to typical FPGA limits.
        """
        # Get total bandwidth requirement
        total_bw_gbps = 0
        resources = self._estimate_resources(graph)
        total_bw_gbps = resources.get("bandwidth_gbps", 0)
        
        # Typical FPGA memory bandwidth limits
        # (conservative estimate for mid-range FPGA)
        max_bandwidth_gbps = 100.0  # 100 GB/s
        
        return min(1.0, total_bw_gbps / max_bandwidth_gbps)
    
    def compare_configurations(self, configs: List[Tuple[DataflowGraph, PerformanceMetrics]]) -> Dict[str, any]:
        """Compare multiple configurations
        
        Args:
            configs: List of (graph, metrics) tuples
            
        Returns:
            Comparison summary
        """
        if not configs:
            return {}
        
        # Find best for each metric
        best_throughput_idx = max(range(len(configs)), 
                                 key=lambda i: configs[i][1].throughput)
        best_latency_idx = min(range(len(configs)),
                              key=lambda i: configs[i][1].latency)
        best_power_idx = min(range(len(configs)),
                            key=lambda i: configs[i][1].power_estimate)
        
        # Compute ranges
        throughputs = [m.throughput for _, m in configs]
        latencies = [m.latency for _, m in configs]
        powers = [m.power_estimate for _, m in configs]
        
        return {
            "n_configs": len(configs),
            "throughput_range": (min(throughputs), max(throughputs)),
            "latency_range": (min(latencies), max(latencies)),
            "power_range": (min(powers), max(powers)),
            "best_throughput_idx": best_throughput_idx,
            "best_latency_idx": best_latency_idx,
            "best_power_idx": best_power_idx,
            "pareto_optimal": self._find_pareto_optimal(configs)
        }
    
    def _find_pareto_optimal(self, configs: List[Tuple[DataflowGraph, PerformanceMetrics]]) -> List[int]:
        """Find Pareto-optimal configurations
        
        A configuration is Pareto-optimal if no other configuration
        is better in all metrics.
        """
        n = len(configs)
        pareto = []
        
        for i in range(n):
            _, metrics_i = configs[i]
            dominated = False
            
            for j in range(n):
                if i == j:
                    continue
                
                _, metrics_j = configs[j]
                
                # Check if j dominates i
                if (metrics_j.throughput >= metrics_i.throughput and
                    metrics_j.latency <= metrics_i.latency and
                    metrics_j.power_estimate <= metrics_i.power_estimate and
                    (metrics_j.throughput > metrics_i.throughput or
                     metrics_j.latency < metrics_i.latency or
                     metrics_j.power_estimate < metrics_i.power_estimate)):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(i)
        
        return pareto