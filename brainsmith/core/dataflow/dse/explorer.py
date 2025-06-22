############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Design space exploration engine for dataflow optimization"""

from dataclasses import dataclass, field, replace
from typing import List, Dict, Optional, Tuple, Callable
from itertools import product
import logging

from ..core.graph import DataflowGraph
from ..core.kernel import Kernel
from ..adfg.actor import ADFGActor
from ..adfg.scheduler import SRTAScheduler, SchedulabilityResult
from .config import ParallelismConfig, DSEConstraints, ConfigurationSpace
from .evaluator import PerformanceEvaluator, PerformanceMetrics


logger = logging.getLogger(__name__)


@dataclass
class DSEResult:
    """Result of design space exploration
    
    Contains the configuration, performance metrics, and feasibility status.
    """
    
    config: ParallelismConfig
    metrics: PerformanceMetrics
    feasible: bool
    violation_reasons: List[str] = field(default_factory=list)
    
    # Optional detailed results
    configured_graph: Optional[DataflowGraph] = None
    schedule: Optional[SchedulabilityResult] = None
    
    def __repr__(self) -> str:
        status = "feasible" if self.feasible else "infeasible"
        return (f"DSEResult({status}, throughput={self.metrics.throughput:.1f}, "
                f"latency={self.metrics.latency}, power={self.metrics.power_estimate:.1f}W)")


class DesignSpaceExplorer:
    """Explore parallelism configurations for dataflow graphs
    
    Systematically explores different parallelization strategies to find
    optimal configurations based on performance, power, and resource constraints.
    """
    
    def __init__(self, graph: DataflowGraph, 
                 constraints: DSEConstraints,
                 evaluator: Optional[PerformanceEvaluator] = None):
        """Initialize explorer
        
        Args:
            graph: Base dataflow graph to optimize
            constraints: Design constraints
            evaluator: Performance evaluator (creates default if None)
        """
        self.base_graph = graph
        self.constraints = constraints
        self.evaluator = evaluator or PerformanceEvaluator(
            frequency_mhz=constraints.target_frequency_mhz
        )
        
        # Cache for evaluated configurations
        self._cache = {}
    
    def explore(self, config_space: Optional[ConfigurationSpace] = None,
                batch_size: int = 1,
                progress_callback: Optional[Callable[[int, int], None]] = None) -> List[DSEResult]:
        """Explore design space
        
        Args:
            config_space: Configuration space to explore (generates default if None)
            batch_size: Batch size for performance evaluation
            progress_callback: Optional callback(current, total) for progress updates
            
        Returns:
            List of DSE results, sorted by feasibility then throughput
        """
        # Generate configuration space if not provided
        if config_space is None:
            config_space = self._generate_default_space()
        
        # Generate candidate configurations
        configs = config_space.generate_configs()
        logger.info(f"Exploring {len(configs)} configurations")
        
        results = []
        
        for i, config in enumerate(configs):
            if progress_callback:
                progress_callback(i, len(configs))
            
            # Check cache
            cache_key = self._config_cache_key(config)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue
            
            # Evaluate configuration
            result = self._evaluate_config(config, batch_size)
            
            # Cache result
            self._cache[cache_key] = result
            results.append(result)
        
        # Sort results: feasible first, then by throughput
        results.sort(key=lambda r: (-r.feasible, -r.metrics.throughput))
        
        # Log summary
        n_feasible = sum(1 for r in results if r.feasible)
        logger.info(f"Found {n_feasible} feasible configurations out of {len(results)}")
        
        return results
    
    def _generate_default_space(self) -> ConfigurationSpace:
        """Generate default configuration space based on graph
        
        Creates a reasonable search space based on the graph structure
        and constraint ranges.
        """
        space = ConfigurationSpace(
            global_options=self.constraints.get_parallelism_range()
        )
        
        # Add key interfaces to explore
        for kernel_name, kernel in self.base_graph.kernels.items():
            # Explore parallelism for input/output interfaces
            for intf in kernel.interfaces:
                if intf.direction.value in ["input", "output"]:
                    # Use constraint range
                    options = self.constraints.get_parallelism_range()
                    
                    # Filter based on interface size
                    max_par = min(intf.stream_dims[0] if intf.stream_dims else 1,
                                 self.constraints.max_parallelism)
                    options = [p for p in options if p <= max_par]
                    
                    if options:
                        space.add_interface(kernel_name, intf.name, options)
        
        # Add coupling for interfaces that should match
        # (e.g., connected interfaces often benefit from same parallelism)
        for edge in self.base_graph.edges.values():
            space.add_coupling([
                (edge.producer_kernel, edge.producer_intf),
                (edge.consumer_kernel, edge.consumer_intf)
            ])
        
        return space
    
    def _evaluate_config(self, config: ParallelismConfig, batch_size: int) -> DSEResult:
        """Evaluate a single configuration
        
        Args:
            config: Parallelism configuration
            batch_size: Batch size for evaluation
            
        Returns:
            DSE result with metrics and feasibility
        """
        violations = []
        
        # Apply configuration to graph
        try:
            configured_graph = self._apply_config(config)
        except Exception as e:
            # Configuration failed to apply
            return DSEResult(
                config=config,
                metrics=PerformanceMetrics(0.0, float('inf'), 0.0),
                feasible=False,
                violation_reasons=[f"Configuration error: {str(e)}"]
            )
        
        # Validate kernel constraints (pragmas)
        for kernel_name, kernel in configured_graph.kernels.items():
            valid, msg = config.validate_kernel(kernel, kernel_name)
            if not valid:
                violations.append(f"Kernel {kernel_name}: {msg}")
        
        # Early exit if configuration violates kernel constraints
        if violations:
            return DSEResult(
                config=config,
                metrics=PerformanceMetrics(0.0, float('inf'), 0.0),
                feasible=False,
                violation_reasons=violations,
                configured_graph=configured_graph
            )
        
        # Schedule and evaluate performance
        schedule = self._schedule_graph(configured_graph)
        metrics = self.evaluator.evaluate(configured_graph, schedule, batch_size)
        
        # Check resource constraints
        resource_ok, resource_violations = self.constraints.check_resources(
            metrics.resource_usage
        )
        if not resource_ok:
            violations.extend(resource_violations)
        
        # Check performance constraints
        perf_ok, perf_violations = self.constraints.check_performance({
            "throughput": metrics.throughput,
            "latency": metrics.latency,
            "fps": metrics.fps
        })
        if not perf_ok:
            violations.extend(perf_violations)
        
        # Check schedulability
        if schedule and not schedule.schedulable:
            violations.append(f"Not schedulable: {schedule.failure_reason or 'Unknown'}")
        
        return DSEResult(
            config=config,
            metrics=metrics,
            feasible=len(violations) == 0,
            violation_reasons=violations,
            configured_graph=configured_graph,
            schedule=schedule
        )
    
    def _apply_config(self, config: ParallelismConfig) -> DataflowGraph:
        """Apply parallelism configuration to graph
        
        Creates a new graph with configured kernels.
        """
        # Create new graph
        configured_graph = DataflowGraph()
        
        # Apply config to each kernel
        for kernel_name, kernel in self.base_graph.kernels.items():
            configured_kernel = config.apply_to_kernel(kernel, kernel_name)
            configured_graph.add_kernel(configured_kernel)
        
        # Copy edges
        for edge in self.base_graph.edges.values():
            configured_graph.add_edge(
                edge.producer_kernel,
                edge.producer_intf,
                edge.consumer_kernel,
                edge.consumer_intf,
                edge.buffer_depth
            )
        
        return configured_graph
    
    def _schedule_graph(self, graph: DataflowGraph) -> SchedulabilityResult:
        """Schedule the graph using SRTA
        
        Args:
            graph: Configured dataflow graph
            
        Returns:
            Scheduling result
        """
        # Create ADFG actors
        actors = []
        for kernel_name, kernel in graph.kernels.items():
            actor = ADFGActor.from_kernel(kernel)
            actor.name = kernel_name
            actors.append(actor)
        
        # Extract edges
        edges = []
        for edge in graph.edges.values():
            edges.append((
                edge.producer_kernel,
                edge.producer_intf,
                edge.consumer_kernel,
                edge.consumer_intf
            ))
        
        # Schedule
        scheduler = SRTAScheduler()
        return scheduler.analyze(actors, edges)
    
    def _config_cache_key(self, config: ParallelismConfig) -> str:
        """Generate cache key for configuration"""
        # Sort interface parallelisms for consistent key
        items = sorted(config.interface_pars.items())
        return f"{items}_{config.global_par}"
    
    def find_pareto_optimal(self, results: List[DSEResult],
                           objectives: Optional[List[str]] = None) -> List[DSEResult]:
        """Find Pareto-optimal configurations
        
        Args:
            results: List of DSE results
            objectives: Metrics to optimize (default: throughput, latency, power)
            
        Returns:
            List of Pareto-optimal results
        """
        if not results:
            return []
        
        if objectives is None:
            objectives = ["throughput", "latency", "power_estimate"]
        
        # Filter to feasible only
        feasible_results = [r for r in results if r.feasible]
        if not feasible_results:
            return []
        
        pareto = []
        
        for i, result_i in enumerate(feasible_results):
            dominated = False
            
            for j, result_j in enumerate(feasible_results):
                if i == j:
                    continue
                
                # Check if j dominates i
                better_count = 0
                equal_count = 0
                
                for obj in objectives:
                    val_i = getattr(result_i.metrics, obj)
                    val_j = getattr(result_j.metrics, obj)
                    
                    # Handle minimization (latency, power) vs maximization (throughput)
                    if obj in ["latency", "power_estimate"]:
                        if val_j < val_i:
                            better_count += 1
                        elif val_j == val_i:
                            equal_count += 1
                    else:  # maximization
                        if val_j > val_i:
                            better_count += 1
                        elif val_j == val_i:
                            equal_count += 1
                
                # j dominates i if better in at least one and not worse in any
                if better_count > 0 and better_count + equal_count == len(objectives):
                    dominated = True
                    break
            
            if not dominated:
                pareto.append(result_i)
        
        return pareto
    
    def summarize_results(self, results: List[DSEResult]) -> Dict[str, any]:
        """Generate summary of exploration results
        
        Args:
            results: List of DSE results
            
        Returns:
            Summary dictionary
        """
        if not results:
            return {"n_configs": 0, "n_feasible": 0}
        
        feasible = [r for r in results if r.feasible]
        pareto = self.find_pareto_optimal(results)
        
        summary = {
            "n_configs": len(results),
            "n_feasible": len(feasible),
            "n_pareto": len(pareto),
            "feasibility_rate": len(feasible) / len(results)
        }
        
        if feasible:
            # Performance ranges
            throughputs = [r.metrics.throughput for r in feasible]
            latencies = [r.metrics.latency for r in feasible]
            powers = [r.metrics.power_estimate for r in feasible]
            
            summary.update({
                "throughput_range": (min(throughputs), max(throughputs)),
                "latency_range": (min(latencies), max(latencies)),
                "power_range": (min(powers), max(powers)),
                "best_throughput": max(throughputs),
                "best_latency": min(latencies),
                "best_power": min(powers)
            })
            
            # Resource usage ranges
            dsp_usage = [r.metrics.resource_usage.get("DSP", 0) for r in feasible]
            bram_usage = [r.metrics.resource_usage.get("BRAM", 0) for r in feasible]
            
            summary.update({
                "dsp_range": (min(dsp_usage), max(dsp_usage)) if dsp_usage else (0, 0),
                "bram_range": (min(bram_usage), max(bram_usage)) if bram_usage else (0, 0)
            })
        
        # Common violation reasons
        all_violations = []
        for r in results:
            if not r.feasible:
                all_violations.extend(r.violation_reasons)
        
        # Count violations
        violation_counts = {}
        for v in all_violations:
            # Extract violation type
            vtype = v.split(":")[0] if ":" in v else v
            violation_counts[vtype] = violation_counts.get(vtype, 0) + 1
        
        summary["common_violations"] = sorted(
            violation_counts.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5
        
        return summary