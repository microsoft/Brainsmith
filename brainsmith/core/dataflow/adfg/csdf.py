############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Cyclo-static dataflow support utilities"""

from typing import List, Tuple, Dict, Optional
import math
import numpy as np


def compute_phase_periods(rates: List[int], base_period: int) -> List[int]:
    """Compute per-phase periods for CSDF pattern
    
    Args:
        rates: CSDF rate pattern [r0, r1, ..., rn]
        base_period: Base period for one token
        
    Returns:
        List of periods for each phase
    """
    if not rates:
        return []
    
    # Period for phase i = base_period * rate[i]
    return [base_period * rate for rate in rates]


def csdf_hyperperiod(phase_counts: List[int]) -> int:
    """Compute hyperperiod for CSDF actors
    
    The hyperperiod is the LCM of all phase counts,
    ensuring all actors complete full CSDF cycles.
    
    Args:
        phase_counts: Number of phases for each actor
        
    Returns:
        Hyperperiod (minimum period for full cycles)
    """
    if not phase_counts:
        return 1
    
    return math.lcm(*phase_counts)


def compute_cumulative_tokens(rates: List[int], n_firings: int) -> List[int]:
    """Compute cumulative token production/consumption
    
    Args:
        rates: CSDF rate pattern
        n_firings: Number of firings to compute
        
    Returns:
        List of cumulative tokens after each firing
    """
    if not rates:
        return [0] * n_firings
    
    cumulative = []
    total = 0
    
    for i in range(n_firings):
        phase = i % len(rates)
        total += rates[phase]
        cumulative.append(total)
    
    return cumulative


def csdf_buffer_bounds(prod_rates: List[int], cons_rates: List[int],
                      prod_period: int, cons_period: int,
                      initial_tokens: int = 0) -> Tuple[int, List[int]]:
    """Calculate buffer size for CSDF edge
    
    Computes minimum buffer size and token evolution over time.
    
    Args:
        prod_rates: Producer CSDF rate pattern
        cons_rates: Consumer CSDF rate pattern  
        prod_period: Producer actor period
        cons_period: Consumer actor period
        initial_tokens: Initial tokens in buffer
        
    Returns:
        Tuple of (max_tokens, token_evolution)
    """
    # Find hyperperiod
    prod_phases = len(prod_rates)
    cons_phases = len(cons_rates)
    
    # Total tokens must balance over repetition vector
    prod_total = sum(prod_rates)
    cons_total = sum(cons_rates)
    
    # Find repetitions that balance rates
    prod_reps = cons_total // math.gcd(prod_total, cons_total)
    cons_reps = prod_total // math.gcd(prod_total, cons_total)
    
    # Compute token evolution over hyperperiod
    total_time = max(prod_reps * prod_phases * prod_period,
                     cons_reps * cons_phases * cons_period)
    
    # Track tokens at each time point
    tokens = initial_tokens
    max_tokens = initial_tokens
    min_tokens = initial_tokens
    token_evolution = [initial_tokens]
    
    # Event list: (time, delta_tokens)
    events = []
    
    # Add producer events
    time = 0
    for rep in range(prod_reps):
        for phase in range(prod_phases):
            if prod_rates[phase] > 0:
                events.append((time, prod_rates[phase]))
            time += prod_period
    
    # Add consumer events
    time = 0
    for rep in range(cons_reps):
        for phase in range(cons_phases):
            if cons_rates[phase] > 0:
                events.append((time, -cons_rates[phase]))
            time += cons_period
    
    # Sort events by time
    events.sort()
    
    # Simulate token evolution
    for time, delta in events:
        tokens += delta
        max_tokens = max(max_tokens, tokens)
        min_tokens = min(min_tokens, tokens)
        token_evolution.append(tokens)
    
    # Buffer size must accommodate maximum tokens
    buffer_size = max_tokens - min_tokens
    
    return buffer_size, token_evolution


def affine_relation(prod_rate: int, cons_rate: int,
                   initial_delay: int = 0) -> Tuple[int, int, int]:
    """Compute affine relation (n, d, φ) between producer and consumer
    
    The affine relation is: d * T_prod = n * T_cons + φ
    
    Args:
        prod_rate: Producer rate
        cons_rate: Consumer rate
        initial_delay: Initial delay/offset
        
    Returns:
        Tuple of (n, d, phi)
    """
    # Reduce rates to lowest terms
    gcd = math.gcd(prod_rate, cons_rate)
    n = prod_rate // gcd
    d = cons_rate // gcd
    phi = initial_delay
    
    return n, d, phi


def validate_rate_consistency(edges: List[Tuple[str, List[int], str, List[int]]],
                             repetitions: Dict[str, int]) -> bool:
    """Validate that rates are consistent with repetition vector
    
    Args:
        edges: List of (producer, prod_rates, consumer, cons_rates)
        repetitions: Actor repetition counts
        
    Returns:
        True if rates are consistent
    """
    for prod, prod_rates, cons, cons_rates in edges:
        prod_reps = repetitions.get(prod, 1)
        cons_reps = repetitions.get(cons, 1)
        
        # Total tokens must match
        prod_total = sum(prod_rates) * prod_reps
        cons_total = sum(cons_rates) * cons_reps
        
        if prod_total != cons_total:
            return False
    
    return True


def phase_schedule(actors: List[str], phases: Dict[str, int],
                  dependencies: List[Tuple[str, str]]) -> List[Tuple[str, int]]:
    """Generate phase-aware schedule for CSDF actors
    
    Args:
        actors: Actor names
        phases: Number of phases per actor
        dependencies: List of (predecessor, successor) pairs
        
    Returns:
        Schedule as list of (actor, phase) tuples
    """
    from collections import defaultdict, deque
    
    # Build dependency graph
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for pred, succ in dependencies:
        graph[pred].append(succ)
        in_degree[succ] += 1
    
    # Initialize with actors that have no dependencies
    queue = deque()
    for actor in actors:
        if in_degree[actor] == 0:
            # Add first phase of independent actors
            queue.append((actor, 0))
    
    schedule = []
    completed_phases = defaultdict(int)
    
    while queue:
        actor, phase = queue.popleft()
        schedule.append((actor, phase))
        
        # Check if we should add next phase
        if phase + 1 < phases.get(actor, 1):
            # Add next phase of same actor
            queue.append((actor, phase + 1))
        
        # Increment completed phases
        completed_phases[actor] = phase + 1
        
        # Check if all phases of this actor are done
        if completed_phases[actor] >= phases.get(actor, 1):
            # Enable successors
            for succ in graph[actor]:
                in_degree[succ] -= 1
                if in_degree[succ] == 0:
                    # Add first phase of newly enabled actor
                    queue.append((succ, 0))
    
    return schedule


def compute_storage_distribution(buffer_sizes: Dict[Tuple[str, str], int],
                               memory_limit: int) -> Dict[str, int]:
    """Distribute buffers across memory banks
    
    Simple first-fit algorithm for memory allocation.
    
    Args:
        buffer_sizes: Dict mapping (producer, consumer) to buffer size
        memory_limit: Size of each memory bank
        
    Returns:
        Dict mapping edge to memory bank index
    """
    # Sort buffers by size (largest first)
    sorted_buffers = sorted(buffer_sizes.items(), 
                           key=lambda x: x[1], 
                           reverse=True)
    
    banks = []  # List of used space per bank
    allocation = {}
    
    for edge, size in sorted_buffers:
        # Try to fit in existing bank
        placed = False
        for i, used in enumerate(banks):
            if used + size <= memory_limit:
                banks[i] += size
                allocation[edge] = i
                placed = True
                break
        
        # Create new bank if needed
        if not placed:
            allocation[edge] = len(banks)
            banks.append(size)
    
    return allocation


def analyze_throughput_bottleneck(actors: Dict[str, Tuple[int, int]],
                                 edges: List[Tuple[str, str]]) -> str:
    """Identify throughput bottleneck in CSDF graph
    
    Args:
        actors: Dict mapping name to (wcet, period)
        edges: List of edges
        
    Returns:
        Name of bottleneck actor
    """
    # Find actor with highest utilization
    max_util = 0
    bottleneck = None
    
    for name, (wcet, period) in actors.items():
        util = wcet / period if period > 0 else float('inf')
        if util > max_util:
            max_util = util
            bottleneck = name
    
    return bottleneck