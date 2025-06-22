############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""ADFG actor representation"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import math
from ..core.kernel import Kernel
from ..core.types import InterfaceDirection


@dataclass
class ADFGActor:
    """Actor in affine dataflow graph
    
    Represents a kernel as an ADFG actor with:
    - Worst-case execution time (WCET)
    - Cyclo-static production/consumption rates
    - Priority for scheduling
    """
    name: str
    wcet: int  # Worst-case execution time in cycles
    rates: Dict[str, List[int]]  # Interface name -> CSDF rate pattern
    priority: Optional[int] = None  # Lower value = higher priority
    
    # Additional timing info
    priming_cycles: int = 0
    flush_cycles: int = 0
    
    # Resource requirements
    resources: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        """Validate actor"""
        if self.wcet <= 0:
            raise ValueError(f"WCET must be positive, got {self.wcet}")
        
        # Validate rates
        for intf_name, rate_pattern in self.rates.items():
            if not rate_pattern:
                raise ValueError(f"Empty rate pattern for interface {intf_name}")
            for rate in rate_pattern:
                if rate < 0:
                    raise ValueError(f"Negative rate {rate} for interface {intf_name}")
    
    @classmethod
    def from_kernel(cls, kernel: Kernel, include_config: bool = False) -> "ADFGActor":
        """Convert kernel to ADFG actor
        
        Args:
            kernel: Kernel to convert
            include_config: Whether to include config interfaces (usually False)
            
        Returns:
            ADFGActor representation
        """
        rates = {}
        
        for intf in kernel.interfaces:
            # Skip config interfaces unless requested
            if intf.direction == InterfaceDirection.CONFIG and not include_config:
                continue
            
            # Get CSDF rate pattern
            rates[intf.name] = intf.rate_pattern
        
        return cls(
            name=kernel.name,
            wcet=kernel.latency_cycles[0],  # Use worst-case
            rates=rates,
            priority=kernel.latency_cycles[0],  # Default: DM priority
            priming_cycles=kernel.priming_cycles,
            flush_cycles=kernel.flush_cycles,
            resources=kernel.resources.copy()
        )
    
    @property
    def is_csdf(self) -> bool:
        """Check if actor has cyclo-static behavior"""
        return any(len(pattern) > 1 for pattern in self.rates.values())
    
    @property
    def n_phases(self) -> int:
        """Get number of CSDF phases
        
        Returns maximum phase count across all interfaces.
        For consistency, all interfaces should have same phase count.
        """
        if not self.rates:
            return 1
        return max(len(pattern) for pattern in self.rates.values())
    
    @property
    def is_consistent(self) -> bool:
        """Check if all interfaces have same number of phases"""
        if not self.rates:
            return True
        
        phase_counts = [len(pattern) for pattern in self.rates.values()]
        return len(set(phase_counts)) == 1
    
    def get_rate(self, interface: str, phase: int = 0) -> int:
        """Get rate for specific interface and phase
        
        Args:
            interface: Interface name
            phase: CSDF phase (0-indexed)
            
        Returns:
            Rate for that phase, or 0 if interface not found
        """
        if interface not in self.rates:
            return 0
        
        pattern = self.rates[interface]
        if phase >= len(pattern):
            # Wrap around for shorter patterns
            phase = phase % len(pattern)
        
        return pattern[phase]
    
    def get_phase_wcet(self, phase: int = 0) -> int:
        """Get WCET for specific phase
        
        For basic actors, WCET is same for all phases.
        This can be overridden for phase-specific timing.
        """
        return self.wcet
    
    def total_tokens(self, interface: str, n_firings: int = 1) -> int:
        """Calculate total tokens produced/consumed over n firings
        
        Args:
            interface: Interface name
            n_firings: Number of actor firings
            
        Returns:
            Total tokens transferred
        """
        if interface not in self.rates:
            return 0
        
        pattern = self.rates[interface]
        full_cycles = n_firings // len(pattern)
        remainder = n_firings % len(pattern)
        
        total = full_cycles * sum(pattern)
        total += sum(pattern[:remainder])
        
        return total
    
    def repetition_count(self, base_period: int = 1) -> int:
        """Calculate repetitions needed in one graph period
        
        For CSDF actors, this ensures all phases complete.
        
        Args:
            base_period: Base repetition count
            
        Returns:
            Actual repetitions needed
        """
        if not self.is_csdf:
            return base_period
        
        # LCM with phase count ensures complete cycles
        return math.lcm(base_period, self.n_phases)
    
    def utilization(self, period: int) -> float:
        """Calculate processor utilization
        
        Args:
            period: Actor period in cycles
            
        Returns:
            Utilization in [0, 1]
        """
        if period <= 0:
            return float('inf')
        
        return self.wcet / period
    
    def __repr__(self) -> str:
        csdf_str = " (CSDF)" if self.is_csdf else ""
        return f"ADFGActor('{self.name}', wcet={self.wcet}, rates={len(self.rates)} interfaces{csdf_str})"


@dataclass
class ActorTiming:
    """Timing analysis results for an actor"""
    actor_name: str
    period: int  # Assigned period
    deadline: int  # Relative deadline
    response_time: int  # Worst-case response time
    start_time: int  # Earliest start time
    utilization: float  # Processor utilization
    
    @property
    def is_schedulable(self) -> bool:
        """Check if actor meets deadline"""
        return self.response_time <= self.deadline
    
    @property
    def slack(self) -> int:
        """Calculate scheduling slack"""
        return self.deadline - self.response_time


def validate_actor_graph(actors: List[ADFGActor], 
                        edges: List[Tuple[str, str, str, str]]) -> None:
    """Validate actor graph for ADFG scheduling
    
    Args:
        actors: List of ADFG actors
        edges: List of (producer, prod_intf, consumer, cons_intf) tuples
        
    Raises:
        ValueError: If validation fails
    """
    actor_dict = {a.name: a for a in actors}
    
    # Check all edges reference valid actors and interfaces
    for prod, prod_intf, cons, cons_intf in edges:
        if prod not in actor_dict:
            raise ValueError(f"Producer actor '{prod}' not found")
        if cons not in actor_dict:
            raise ValueError(f"Consumer actor '{cons}' not found")
        
        prod_actor = actor_dict[prod]
        cons_actor = actor_dict[cons]
        
        if prod_intf not in prod_actor.rates:
            raise ValueError(f"Interface '{prod_intf}' not found in actor '{prod}'")
        if cons_intf not in cons_actor.rates:
            raise ValueError(f"Interface '{cons_intf}' not found in actor '{cons}'")
    
    # Check CSDF consistency
    for actor in actors:
        if not actor.is_consistent:
            raise ValueError(
                f"Actor '{actor.name}' has inconsistent CSDF phases across interfaces"
            )


def compute_repetition_vector(actors: List[ADFGActor],
                             edges: List[Tuple[str, str, str, str]]) -> Dict[str, int]:
    """Compute repetition vector for actor graph
    
    The repetition vector ensures rate consistency:
    For each edge, tokens produced = tokens consumed
    
    Args:
        actors: List of ADFG actors
        edges: List of (producer, prod_intf, consumer, cons_intf) tuples
        
    Returns:
        Dict mapping actor names to repetition counts
    """
    import math
    from fractions import Fraction
    
    if not actors:
        return {}
    
    actor_dict = {a.name: a for a in actors}
    
    # Use fractions for exact arithmetic
    repetitions = {a.name: Fraction(1) for a in actors}
    
    # Iterate until all edges are balanced
    max_iterations = 100
    for iteration in range(max_iterations):
        changed = False
        
        for prod, prod_intf, cons, cons_intf in edges:
            prod_actor = actor_dict[prod]
            cons_actor = actor_dict[cons]
            
            # Get total rates per cycle
            prod_rate = sum(prod_actor.rates.get(prod_intf, [0]))
            cons_rate = sum(cons_actor.rates.get(cons_intf, [0]))
            
            if prod_rate == 0 or cons_rate == 0:
                continue
            
            # Current tokens produced/consumed
            prod_tokens = repetitions[prod] * prod_rate
            cons_tokens = repetitions[cons] * cons_rate
            
            # Balance if needed
            if prod_tokens != cons_tokens:
                if prod_tokens < cons_tokens:
                    # Increase producer repetitions
                    repetitions[prod] = cons_tokens / prod_rate
                else:
                    # Increase consumer repetitions
                    repetitions[cons] = prod_tokens / cons_rate
                changed = True
        
        if not changed:
            break
    
    # Convert to minimal integer form
    # Find LCD of all denominators
    denominators = [r.denominator for r in repetitions.values()]
    lcm_denom = math.lcm(*denominators) if denominators else 1
    
    # Convert to integers
    result = {}
    for name, frac in repetitions.items():
        result[name] = int(frac * lcm_denom)
    
    # Minimize by dividing by GCD
    if result:
        gcd = math.gcd(*result.values())
        for name in result:
            result[name] //= gcd
    
    return result