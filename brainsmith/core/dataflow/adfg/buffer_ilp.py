############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Buffer sizing using Integer Linear Programming (ILP)"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
import math
try:
    import pulp
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

from .actor import ADFGActor, compute_repetition_vector
from .csdf import csdf_buffer_bounds, validate_rate_consistency


@dataclass
class BufferConfig:
    """Configuration for buffer sizing"""
    memory_limit: Optional[int] = None  # Total memory budget
    min_buffer_size: int = 1  # Minimum buffer size
    max_buffer_size: Optional[int] = None  # Maximum buffer size per edge
    
    # Optimization objectives
    objective: str = "min_total"  # "min_total", "min_max", "balanced"
    
    # Memory architecture
    memory_banks: int = 1  # Number of memory banks
    bank_size: Optional[int] = None  # Size of each bank
    
    # Constraints
    enforce_back_pressure: bool = True  # Ensure no buffer overflow
    allow_initial_tokens: bool = True  # Allow non-zero initial tokens
    
    def __post_init__(self):
        """Validate configuration"""
        if self.min_buffer_size < 1:
            raise ValueError("Minimum buffer size must be >= 1")
        if self.max_buffer_size and self.max_buffer_size < self.min_buffer_size:
            raise ValueError("Maximum buffer size must be >= minimum")
        if self.objective not in ["min_total", "min_max", "balanced"]:
            raise ValueError(f"Unknown objective: {self.objective}")
        if self.memory_banks < 1:
            raise ValueError("Must have at least one memory bank")


@dataclass
class BufferSolution:
    """Solution from buffer sizing"""
    feasible: bool
    buffer_sizes: Dict[Tuple[str, str], int]  # (producer, consumer) -> size
    initial_tokens: Dict[Tuple[str, str], int]  # (producer, consumer) -> tokens
    bank_assignment: Dict[Tuple[str, str], int]  # (producer, consumer) -> bank
    total_memory: int
    max_buffer: int
    objective_value: float
    
    # Additional metrics
    memory_utilization: float = 0.0  # Fraction of memory budget used
    bank_utilization: Dict[int, float] = field(default_factory=dict)  # Per-bank usage
    
    def summary(self) -> str:
        """Get human-readable summary"""
        if not self.feasible:
            return "No feasible buffer sizing found"
        
        return (f"Buffer sizing: total={self.total_memory}, max={self.max_buffer}, "
                f"edges={len(self.buffer_sizes)}, banks={len(set(self.bank_assignment.values()))}")


class BufferSizingILP:
    """Buffer sizing using Integer Linear Programming
    
    Formulates buffer sizing as an ILP problem to:
    - Minimize total buffer memory
    - Ensure deadlock-free execution
    - Respect memory constraints
    - Support CSDF actors
    """
    
    def __init__(self, config: Optional[BufferConfig] = None):
        """Initialize buffer sizing with configuration
        
        Args:
            config: Buffer configuration (uses defaults if None)
        """
        if not HAS_PULP:
            raise ImportError("PuLP is required for ILP buffer sizing. Install with: pip install pulp")
        
        self.config = config or BufferConfig()
    
    def solve(self, actors: List[ADFGActor],
              edges: List[Tuple[str, str, str, str]],
              periods: Optional[Dict[str, int]] = None) -> BufferSolution:
        """Solve buffer sizing problem
        
        Args:
            actors: List of ADFG actors
            edges: List of (producer, prod_intf, consumer, cons_intf) tuples
            periods: Optional actor periods for tighter bounds
            
        Returns:
            BufferSolution with sizing results
        """
        if not edges:
            return BufferSolution(
                feasible=True,
                buffer_sizes={},
                initial_tokens={},
                bank_assignment={},
                total_memory=0,
                max_buffer=0,
                objective_value=0.0
            )
        
        # Compute repetition vector
        repetitions = compute_repetition_vector(actors, edges)
        
        # Create actor lookup
        actor_dict = {a.name: a for a in actors}
        
        # Create edge lookup: (prod, cons) -> (prod_intf, cons_intf)
        edge_dict = {}
        for prod, prod_intf, cons, cons_intf in edges:
            edge_dict[(prod, cons)] = (prod_intf, cons_intf)
        
        # Formulate ILP
        prob = pulp.LpProblem("BufferSizing", pulp.LpMinimize)
        
        # Decision variables
        buffer_vars = {}  # Buffer size for each edge
        token_vars = {}   # Initial tokens for each edge
        bank_vars = {}    # Bank assignment (if multi-bank)
        
        # Create variables for each edge
        for (prod, cons), (prod_intf, cons_intf) in edge_dict.items():
            # Buffer size variable
            max_size = self.config.max_buffer_size or 10000
            buffer_vars[(prod, cons)] = pulp.LpVariable(
                f"buf_{prod}_{cons}",
                lowBound=self.config.min_buffer_size,
                upBound=max_size,
                cat='Integer'
            )
            
            # Initial tokens (if allowed)
            if self.config.allow_initial_tokens:
                token_vars[(prod, cons)] = pulp.LpVariable(
                    f"tok_{prod}_{cons}",
                    lowBound=0,
                    upBound=max_size,
                    cat='Integer'
                )
            else:
                token_vars[(prod, cons)] = 0
            
            # Bank assignment (if multi-bank)
            if self.config.memory_banks > 1:
                for b in range(self.config.memory_banks):
                    bank_vars[(prod, cons, b)] = pulp.LpVariable(
                        f"bank_{prod}_{cons}_{b}",
                        cat='Binary'
                    )
        
        # Objective function
        if self.config.objective == "min_total":
            # Minimize total buffer memory
            prob += pulp.lpSum(buffer_vars.values())
        
        elif self.config.objective == "min_max":
            # Minimize maximum buffer size
            max_buf = pulp.LpVariable("max_buffer", lowBound=0)
            prob += max_buf
            
            # Constraints: each buffer <= max_buf
            for buf_var in buffer_vars.values():
                prob += buf_var <= max_buf
        
        elif self.config.objective == "balanced":
            # Minimize variance (approximated by range)
            max_buf = pulp.LpVariable("max_buffer", lowBound=0)
            min_buf = pulp.LpVariable("min_buffer", lowBound=0)
            prob += max_buf - min_buf
            
            for buf_var in buffer_vars.values():
                prob += buf_var <= max_buf
                prob += buf_var >= min_buf
        
        # Constraints
        
        # 1. Liveness constraint: ensure deadlock-free execution
        # For CSDF, we need to ensure buffer can hold maximum accumulation
        for (prod, cons), (prod_intf, cons_intf) in edge_dict.items():
            prod_actor = actor_dict[prod]
            cons_actor = actor_dict[cons]
            
            # Get rate patterns
            prod_rates = prod_actor.rates[prod_intf]
            cons_rates = cons_actor.rates[cons_intf]
            
            # Compute minimum buffer size for liveness
            if len(prod_rates) > 1 or len(cons_rates) > 1:
                # CSDF edge - use analytical bounds
                min_size, _ = csdf_buffer_bounds(
                    prod_rates, cons_rates,
                    prod_period=100,  # Dummy period for sizing
                    cons_period=100,
                    initial_tokens=0
                )
                prob += buffer_vars[(prod, cons)] >= min_size
            else:
                # SDF edge - simple constraint
                min_size = max(prod_rates[0], cons_rates[0])
                prob += buffer_vars[(prod, cons)] >= min_size
        
        # 2. Memory constraints
        if self.config.memory_limit:
            prob += pulp.lpSum(buffer_vars.values()) <= self.config.memory_limit
        
        # 3. Bank assignment constraints (if multi-bank)
        if self.config.memory_banks > 1:
            # Each edge assigned to exactly one bank
            for (prod, cons) in edge_dict:
                prob += pulp.lpSum(
                    bank_vars[(prod, cons, b)] 
                    for b in range(self.config.memory_banks)
                ) == 1
            
            # Bank capacity constraints
            if self.config.bank_size:
                for b in range(self.config.memory_banks):
                    prob += pulp.lpSum(
                        buffer_vars[(prod, cons)] * bank_vars.get((prod, cons, b), 0)
                        for (prod, cons) in edge_dict
                    ) <= self.config.bank_size
        
        # 4. Back-pressure constraints (prevent overflow)
        if self.config.enforce_back_pressure:
            for (prod, cons), (prod_intf, cons_intf) in edge_dict.items():
                prod_actor = actor_dict[prod]
                cons_actor = actor_dict[cons]
                
                # Maximum tokens that can accumulate
                prod_rep = repetitions[prod]
                cons_rep = repetitions[cons]
                
                max_prod = sum(prod_actor.rates[prod_intf]) * prod_rep
                max_cons = sum(cons_actor.rates[cons_intf]) * cons_rep
                
                # Buffer must handle worst-case accumulation
                if self.config.allow_initial_tokens:
                    prob += buffer_vars[(prod, cons)] >= token_vars[(prod, cons)]
        
        # Solve ILP
        prob.solve(pulp.PULP_CBC_CMD(msg=0))  # Use CBC solver, suppress output
        
        # Extract solution
        if prob.status != pulp.LpStatusOptimal:
            return BufferSolution(
                feasible=False,
                buffer_sizes={},
                initial_tokens={},
                bank_assignment={},
                total_memory=0,
                max_buffer=0,
                objective_value=float('inf')
            )
        
        # Extract buffer sizes
        buffer_sizes = {
            edge: int(var.varValue)
            for edge, var in buffer_vars.items()
        }
        
        # Extract initial tokens
        initial_tokens = {}
        if self.config.allow_initial_tokens:
            for edge, var in token_vars.items():
                if isinstance(var, pulp.LpVariable):
                    initial_tokens[edge] = int(var.varValue)
                else:
                    initial_tokens[edge] = 0
        
        # Extract bank assignments
        bank_assignment = {}
        if self.config.memory_banks > 1:
            for (prod, cons) in edge_dict:
                for b in range(self.config.memory_banks):
                    if bank_vars[(prod, cons, b)].varValue > 0.5:
                        bank_assignment[(prod, cons)] = b
                        break
        else:
            # Single bank
            bank_assignment = {edge: 0 for edge in edge_dict}
        
        # Compute metrics
        total_memory = sum(buffer_sizes.values())
        max_buffer = max(buffer_sizes.values()) if buffer_sizes else 0
        
        # Memory utilization
        memory_util = 0.0
        if self.config.memory_limit:
            memory_util = total_memory / self.config.memory_limit
        
        # Bank utilization
        bank_util = {}
        for b in range(self.config.memory_banks):
            bank_buffers = [
                buffer_sizes[edge]
                for edge, bank in bank_assignment.items()
                if bank == b
            ]
            bank_total = sum(bank_buffers)
            
            if self.config.bank_size:
                bank_util[b] = bank_total / self.config.bank_size
            else:
                bank_util[b] = 1.0 if bank_buffers else 0.0
        
        return BufferSolution(
            feasible=True,
            buffer_sizes=buffer_sizes,
            initial_tokens=initial_tokens,
            bank_assignment=bank_assignment,
            total_memory=total_memory,
            max_buffer=max_buffer,
            objective_value=prob.objective.value(),
            memory_utilization=memory_util,
            bank_utilization=bank_util
        )
    
    def analyze_bounds(self, actors: List[ADFGActor],
                      edges: List[Tuple[str, str, str, str]]) -> Dict[str, any]:
        """Analyze theoretical bounds on buffer sizes
        
        Args:
            actors: List of actors
            edges: List of edges
            
        Returns:
            Dict with bound analysis
        """
        actor_dict = {a.name: a for a in actors}
        bounds = {}
        
        # Analyze each edge
        edge_bounds = {}
        for prod, prod_intf, cons, cons_intf in edges:
            prod_actor = actor_dict[prod]
            cons_actor = actor_dict[cons]
            
            # Get rates
            prod_rates = prod_actor.rates[prod_intf]
            cons_rates = cons_actor.rates[cons_intf]
            
            # Theoretical minimum (for liveness)
            if len(prod_rates) == 1 and len(cons_rates) == 1:
                # SDF edge
                min_bound = max(prod_rates[0], cons_rates[0])
            else:
                # CSDF edge
                min_bound, _ = csdf_buffer_bounds(
                    prod_rates, cons_rates, 100, 100, 0
                )
            
            # Practical minimum (with back-pressure)
            total_prod = sum(prod_rates)
            total_cons = sum(cons_rates)
            gcd = math.gcd(total_prod, total_cons)
            practical_min = max(total_prod, total_cons) // gcd
            
            edge_bounds[(prod, cons)] = {
                "theoretical_min": min_bound,
                "practical_min": practical_min,
                "rates": (prod_rates, cons_rates)
            }
        
        # Total bounds
        bounds["edges"] = edge_bounds
        bounds["total_theoretical_min"] = sum(
            eb["theoretical_min"] for eb in edge_bounds.values()
        )
        bounds["total_practical_min"] = sum(
            eb["practical_min"] for eb in edge_bounds.values()
        )
        
        # CSDF analysis
        csdf_edges = [
            edge for edge, eb in edge_bounds.items()
            if len(eb["rates"][0]) > 1 or len(eb["rates"][1]) > 1
        ]
        bounds["csdf_edges"] = len(csdf_edges)
        bounds["total_edges"] = len(edges)
        
        return bounds