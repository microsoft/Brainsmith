############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""SRTA (Shortest Remaining Time Adjustment) scheduler for ADFG"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
import math
from .actor import ADFGActor, ActorTiming
from .csdf import csdf_hyperperiod, phase_schedule


@dataclass
class SRTAConfig:
    """Configuration for SRTA scheduler"""
    utilization_target: float = 0.69  # Liu & Layland bound for n->inf
    max_period_search_iterations: int = 100
    period_search_step: float = 1.1  # Growth factor for period search
    min_period_scale: float = 1.0  # Minimum period scaling factor
    max_period_scale: float = 10.0  # Maximum period scaling factor
    
    # Deadline assignment strategies
    deadline_strategy: str = "implicit"  # "implicit", "constrained", "arbitrary"
    deadline_factor: float = 1.0  # For constrained deadlines (D = factor * T)
    
    def __post_init__(self):
        """Validate configuration"""
        if not 0 < self.utilization_target <= 1:
            raise ValueError("Utilization target must be in (0, 1]")
        if self.period_search_step <= 1:
            raise ValueError("Period search step must be > 1")
        if self.deadline_strategy not in ["implicit", "constrained", "arbitrary"]:
            raise ValueError(f"Unknown deadline strategy: {self.deadline_strategy}")


@dataclass
class SchedulabilityResult:
    """Result of schedulability analysis"""
    schedulable: bool
    actor_timings: Dict[str, ActorTiming]
    total_utilization: float
    hyperperiod: int
    schedule: Optional[List[Tuple[str, int]]] = None  # (actor, phase) pairs
    failure_reason: Optional[str] = None
    
    def summary(self) -> str:
        """Get human-readable summary"""
        if self.schedulable:
            return (f"Schedulable with utilization {self.total_utilization:.2%}, "
                   f"hyperperiod {self.hyperperiod}")
        else:
            return f"Not schedulable: {self.failure_reason}"


class SRTAScheduler:
    """Shortest Remaining Time Adjustment scheduler
    
    Implements SRTA scheduling with:
    - Deadline-monotonic priority assignment
    - Response time analysis with interference
    - Period adjustment for schedulability
    - CSDF phase handling
    """
    
    def __init__(self, config: Optional[SRTAConfig] = None):
        """Initialize scheduler with configuration
        
        Args:
            config: Scheduler configuration (uses defaults if None)
        """
        self.config = config or SRTAConfig()
    
    def analyze(self, actors: List[ADFGActor], 
                edges: List[Tuple[str, str, str, str]],
                fixed_periods: Optional[Dict[str, int]] = None) -> SchedulabilityResult:
        """Analyze schedulability of actor graph
        
        Args:
            actors: List of ADFG actors
            edges: List of (producer, prod_intf, consumer, cons_intf) tuples
            fixed_periods: Optional fixed periods for specific actors
            
        Returns:
            SchedulabilityResult with analysis details
        """
        if not actors:
            return SchedulabilityResult(
                schedulable=True,
                actor_timings={},
                total_utilization=0.0,
                hyperperiod=1
            )
        
        fixed_periods = fixed_periods or {}
        
        # Assign priorities (lower value = higher priority)
        actors_by_priority = self._assign_priorities(actors, fixed_periods)
        
        # Find schedulable periods
        periods = self._find_schedulable_periods(actors_by_priority, fixed_periods)
        
        if periods is None:
            return SchedulabilityResult(
                schedulable=False,
                actor_timings={},
                total_utilization=self._compute_utilization(actors, {a.name: a.wcet for a in actors}),
                hyperperiod=0,
                failure_reason="Could not find schedulable periods"
            )
        
        # Compute response times
        timings = self._compute_response_times(actors_by_priority, periods)
        
        # Check schedulability
        schedulable = all(t.is_schedulable for t in timings.values())
        
        # Compute hyperperiod and schedule if schedulable
        hyperperiod = 1
        schedule = None
        
        if schedulable:
            # Include CSDF phases in hyperperiod
            phase_counts = [a.n_phases for a in actors]
            period_values = list(periods.values())
            hyperperiod = math.lcm(*phase_counts, *period_values)
            
            # Generate phase-aware schedule
            if any(a.is_csdf for a in actors):
                actor_phases = {a.name: a.n_phases for a in actors}
                dependencies = [(e[0], e[2]) for e in edges]
                schedule = phase_schedule(
                    [a.name for a in actors],
                    actor_phases,
                    dependencies
                )
        
        return SchedulabilityResult(
            schedulable=schedulable,
            actor_timings=timings,
            total_utilization=self._compute_utilization(actors, periods),
            hyperperiod=hyperperiod,
            schedule=schedule,
            failure_reason=None if schedulable else "Response time exceeds deadline"
        )
    
    def _assign_priorities(self, actors: List[ADFGActor], 
                          fixed_periods: Dict[str, int]) -> List[ADFGActor]:
        """Assign priorities using deadline-monotonic policy
        
        Args:
            actors: List of actors
            fixed_periods: Fixed periods for some actors
            
        Returns:
            Actors sorted by priority (highest priority first)
        """
        # Estimate initial periods
        estimated_periods = {}
        for actor in actors:
            if actor.name in fixed_periods:
                estimated_periods[actor.name] = fixed_periods[actor.name]
            else:
                # Initial estimate based on WCET and target utilization
                estimated_periods[actor.name] = int(
                    actor.wcet / self.config.utilization_target
                )
        
        # Compute deadlines
        deadlines = {}
        for actor in actors:
            period = estimated_periods[actor.name]
            
            if self.config.deadline_strategy == "implicit":
                deadlines[actor.name] = period
            elif self.config.deadline_strategy == "constrained":
                deadlines[actor.name] = int(period * self.config.deadline_factor)
            else:  # arbitrary
                deadlines[actor.name] = actor.priority or period
        
        # Sort by deadline (shorter deadline = higher priority)
        sorted_actors = sorted(actors, key=lambda a: deadlines[a.name])
        
        # Assign priority values
        for i, actor in enumerate(sorted_actors):
            actor.priority = i
        
        return sorted_actors
    
    def _find_schedulable_periods(self, actors: List[ADFGActor],
                                 fixed_periods: Dict[str, int]) -> Optional[Dict[str, int]]:
        """Find schedulable periods using binary search
        
        Args:
            actors: Actors sorted by priority
            fixed_periods: Fixed periods for some actors
            
        Returns:
            Dict of actor names to periods, or None if not schedulable
        """
        # Start with minimum feasible periods
        min_periods = {}
        for actor in actors:
            if actor.name in fixed_periods:
                min_periods[actor.name] = fixed_periods[actor.name]
            else:
                min_periods[actor.name] = actor.wcet
        
        # Check if minimum periods are schedulable
        timings = self._compute_response_times(actors, min_periods)
        if all(t.is_schedulable for t in timings.values()):
            return min_periods
        
        # Binary search for schedulable periods
        scale_low = self.config.min_period_scale
        scale_high = self.config.max_period_scale
        
        for _ in range(self.config.max_period_search_iterations):
            scale = (scale_low + scale_high) / 2
            
            # Scale non-fixed periods
            test_periods = {}
            for actor in actors:
                if actor.name in fixed_periods:
                    test_periods[actor.name] = fixed_periods[actor.name]
                else:
                    test_periods[actor.name] = int(min_periods[actor.name] * scale)
            
            # Test schedulability
            timings = self._compute_response_times(actors, test_periods)
            
            if all(t.is_schedulable for t in timings.values()):
                # Schedulable - try smaller periods
                scale_high = scale
                
                # Check if we've converged
                if scale_high - scale_low < 0.01:
                    return test_periods
            else:
                # Not schedulable - need larger periods
                scale_low = scale
        
        return None
    
    def _compute_response_times(self, actors: List[ADFGActor],
                               periods: Dict[str, int]) -> Dict[str, ActorTiming]:
        """Compute worst-case response times
        
        Uses fixed-point iteration to compute response times
        accounting for interference from higher-priority actors.
        
        Args:
            actors: Actors sorted by priority
            periods: Actor periods
            
        Returns:
            Dict mapping actor names to timing analysis results
        """
        timings = {}
        
        for i, actor in enumerate(actors):
            # Higher priority actors (indices 0 to i-1)
            hp_actors = actors[:i]
            
            # Initial response time estimate (no interference)
            response_time = actor.wcet
            
            # Fixed-point iteration
            converged = False
            max_iterations = 100
            
            for _ in range(max_iterations):
                # Compute interference from higher priority actors
                interference = 0
                
                for hp in hp_actors:
                    # Number of preemptions in response time window
                    n_preemptions = math.ceil(response_time / periods[hp.name])
                    interference += n_preemptions * hp.wcet
                
                # New response time
                new_response = actor.wcet + interference
                
                # Check convergence
                if new_response == response_time:
                    converged = True
                    break
                
                # Check if exceeds deadline
                deadline = self._get_deadline(actor, periods[actor.name])
                if new_response > deadline:
                    response_time = new_response
                    break
                
                response_time = new_response
            
            # Create timing result
            timings[actor.name] = ActorTiming(
                actor_name=actor.name,
                period=periods[actor.name],
                deadline=self._get_deadline(actor, periods[actor.name]),
                response_time=response_time,
                start_time=0,  # Could be refined with offset analysis
                utilization=actor.wcet / periods[actor.name]
            )
        
        return timings
    
    def _get_deadline(self, actor: ADFGActor, period: int) -> int:
        """Get deadline for actor based on configuration
        
        Args:
            actor: The actor
            period: Actor period
            
        Returns:
            Deadline value
        """
        if self.config.deadline_strategy == "implicit":
            return period
        elif self.config.deadline_strategy == "constrained":
            return int(period * self.config.deadline_factor)
        else:  # arbitrary
            return actor.priority or period
    
    def _compute_utilization(self, actors: List[ADFGActor],
                           periods: Dict[str, int]) -> float:
        """Compute total processor utilization
        
        Args:
            actors: List of actors
            periods: Actor periods
            
        Returns:
            Total utilization in [0, 1]
        """
        total = 0.0
        for actor in actors:
            if actor.name in periods and periods[actor.name] > 0:
                total += actor.wcet / periods[actor.name]
        return total
    
    def optimize_periods(self, actors: List[ADFGActor],
                        edges: List[Tuple[str, str, str, str]],
                        objective: str = "min_hyperperiod") -> Optional[Dict[str, int]]:
        """Optimize periods for specific objective
        
        Args:
            actors: List of actors
            edges: Graph edges
            objective: Optimization objective
                - "min_hyperperiod": Minimize hyperperiod
                - "min_utilization": Minimize processor utilization
                - "max_slack": Maximize scheduling slack
                
        Returns:
            Optimized periods or None if not schedulable
        """
        # First find any schedulable solution
        result = self.analyze(actors, edges)
        if not result.schedulable:
            return None
        
        base_periods = {name: t.period for name, t in result.actor_timings.items()}
        
        if objective == "min_hyperperiod":
            # Try to find periods with small LCM
            # Use harmonic periods when possible
            base = min(base_periods.values())
            harmonic_periods = {}
            
            for actor in actors:
                # Find smallest harmonic period >= minimum
                min_period = actor.wcet
                factor = math.ceil(min_period / base)
                harmonic_periods[actor.name] = base * factor
            
            # Verify schedulability
            test_result = self.analyze(actors, edges, harmonic_periods)
            if test_result.schedulable:
                return harmonic_periods
        
        elif objective == "min_utilization":
            # Gradually increase periods to reduce utilization
            scale = 1.0
            best_periods = base_periods.copy()
            
            for _ in range(20):
                scale *= 1.1
                test_periods = {
                    name: int(period * scale)
                    for name, period in base_periods.items()
                }
                
                test_result = self.analyze(actors, edges, test_periods)
                if test_result.schedulable:
                    best_periods = test_periods
                    
                    # Stop if utilization is low enough
                    if test_result.total_utilization < 0.5:
                        break
            
            return best_periods
        
        elif objective == "max_slack":
            # Maximize minimum slack across all actors
            # This provides robustness against execution time variations
            best_periods = base_periods.copy()
            best_min_slack = min(t.slack for t in result.actor_timings.values())
            
            # Try scaling periods
            for scale in [1.1, 1.2, 1.5, 2.0]:
                test_periods = {
                    name: int(period * scale)
                    for name, period in base_periods.items()
                }
                
                test_result = self.analyze(actors, edges, test_periods)
                if test_result.schedulable:
                    min_slack = min(t.slack for t in test_result.actor_timings.values())
                    if min_slack > best_min_slack:
                        best_min_slack = min_slack
                        best_periods = test_periods
            
            return best_periods
        
        # Default: return base solution
        return base_periods