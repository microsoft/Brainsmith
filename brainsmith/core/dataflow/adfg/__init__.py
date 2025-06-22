############################################################################
# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#
# @author       Thomas Keller <thomaskeller@microsoft.com>
############################################################################

"""Affine Dataflow Graph (ADFG) scheduling components"""

from .actor import ADFGActor, ActorTiming, validate_actor_graph, compute_repetition_vector
from .csdf import (
    compute_phase_periods, csdf_hyperperiod, compute_cumulative_tokens,
    csdf_buffer_bounds, affine_relation, validate_rate_consistency,
    phase_schedule, compute_storage_distribution, analyze_throughput_bottleneck
)
from .scheduler import SRTAScheduler, SRTAConfig, SchedulabilityResult

# Buffer ILP is optional (requires PuLP)
try:
    from .buffer_ilp import BufferSizingILP, BufferConfig, BufferSolution
    HAS_BUFFER_ILP = True
except ImportError:
    HAS_BUFFER_ILP = False

__all__ = [
    # Actor abstraction
    "ADFGActor", "ActorTiming", "validate_actor_graph", "compute_repetition_vector",
    
    # CSDF utilities
    "compute_phase_periods", "csdf_hyperperiod", "compute_cumulative_tokens",
    "csdf_buffer_bounds", "affine_relation", "validate_rate_consistency", 
    "phase_schedule", "compute_storage_distribution", "analyze_throughput_bottleneck",
    
    # SRTA scheduler
    "SRTAScheduler", "SRTAConfig", "SchedulabilityResult"
]

# Add buffer ILP if available
if HAS_BUFFER_ILP:
    __all__.extend(["BufferSizingILP", "BufferConfig", "BufferSolution"])