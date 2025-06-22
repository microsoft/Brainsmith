# Unified Kernel Modeling Framework - Implementation Plan

## Overview

This document provides a detailed, step-by-step implementation plan for creating the unified kernel modeling framework in `brainsmith/core/dataflow/`. The implementation is divided into 3 main phases focusing on core functionality, ADFG integration, and DSE capabilities.

## Directory Structure

```
brainsmith/core/dataflow/
├── __init__.py
├── core/
│   ├── __init__.py
│   ├── types.py          # Basic types and shapes
│   ├── interface.py      # Interface definition
│   ├── kernel.py         # Kernel definition
│   ├── pragma.py         # Constraint system
│   └── graph.py          # Dataflow graph
├── adfg/
│   ├── __init__.py
│   ├── actor.py          # ADFG actor abstraction
│   ├── scheduler.py      # SRTA scheduler
│   ├── buffer_sizing.py  # ILP buffer sizing
│   └── csdf.py          # Cyclo-static support
├── dse/
│   ├── __init__.py
│   ├── config.py         # Parallelism configuration
│   ├── explorer.py       # Design space exploration
│   └── evaluator.py      # Performance evaluation
└── tests/
    ├── test_core.py
    ├── test_adfg.py
    └── test_dse.py
```

## Phase 1: Core Data Structures (Week 1-2)

### Step 1.1: Basic Types and Shapes
**File:** `brainsmith/core/dataflow/core/types.py`

```python
"""Basic types for kernel modeling"""

from typing import Tuple, Union, List, Optional
from dataclasses import dataclass
from enum import Enum

# Type aliases
Shape = Tuple[int, ...]
RaggedShape = Union[Shape, List[Shape]]

class InterfaceDirection(Enum):
    INPUT = "input"
    OUTPUT = "output" 
    WEIGHT = "weight"
    CONFIG = "config"

@dataclass
class DataType:
    """Data type with bit width"""
    name: str  # "INT8", "INT16", "FP16", etc.
    bits: int
    
    @classmethod
    def from_string(cls, dtype_str: str) -> "DataType":
        """Parse from FINN-style dtype string"""
        # Implementation here
```

**Tasks:**
- [ ] Define basic type aliases
- [ ] Create InterfaceDirection enum
- [ ] Implement DataType class with parsing
- [ ] Add shape manipulation utilities

### Step 1.2: Interface Implementation
**File:** `brainsmith/core/dataflow/core/interface.py`

```python
"""Unified interface definition"""

from dataclasses import dataclass, field
from typing import List, Optional, Union
from .types import Shape, RaggedShape, InterfaceDirection, DataType

@dataclass
class Interface:
    """Hardware interface with data hierarchy"""
    
    # Core properties
    name: str
    direction: InterfaceDirection
    dtype: DataType
    
    # Data hierarchy
    tensor_dims: Shape
    block_dims: RaggedShape
    stream_dims: Shape = (1,)
    
    # Advanced features
    skip_prob: List[float] = field(default_factory=list)
    optional: bool = False
    
    def __post_init__(self):
        """Validate and normalize inputs"""
        # Normalize block_dims to list format
        if isinstance(self.block_dims, tuple):
            self.block_dims = [self.block_dims]
            
    @property
    def ipar(self) -> int:
        """Interface parallelism"""
        return prod(self.stream_dims)
    
    @property
    def n_phases(self) -> int:
        """Number of CSDF phases"""
        return len(self.block_dims) if isinstance(self.block_dims, list) else 1
    
    @property
    def rate_pattern(self) -> List[int]:
        """CSDF rate pattern for ADFG"""
        return [prod(bd) // prod(self.stream_dims) 
                for bd in self.block_dims]
    
    @property
    def ii_pattern(self) -> List[int]:
        """Initiation interval per phase"""
        return [ceil(prod(bd) / prod(self.stream_dims))
                for bd in self.block_dims]
```

**Tasks:**
- [ ] Implement Interface dataclass
- [ ] Add validation in __post_init__
- [ ] Implement derived properties (ipar, rate_pattern, etc.)
- [ ] Add helper methods for shape calculations
- [ ] Write comprehensive unit tests

### Step 1.3: Pragma System
**File:** `brainsmith/core/dataflow/core/pragma.py`

```python
"""Constraint system for kernel modeling"""

import ast
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, List

class Pragma(ABC):
    """Base class for all pragmas"""
    
    @abstractmethod
    def evaluate(self, interfaces: Dict[str, Interface], 
                env: Dict[str, int]) -> bool:
        """Evaluate pragma given interfaces and environment"""
        pass
    
    @abstractmethod
    def to_string(self) -> str:
        """Convert to human-readable string"""
        pass

@dataclass
class TiePragma(Pragma):
    """Equality constraint between interface expressions"""
    left_expr: str
    right_expr: str
    
    def evaluate(self, interfaces: Dict[str, Interface], 
                env: Dict[str, int]) -> bool:
        left_val = evaluate_expr(self.left_expr, interfaces, env)
        right_val = evaluate_expr(self.right_expr, interfaces, env)
        return left_val == right_val
    
    def to_string(self) -> str:
        return f"TIE {self.left_expr} {self.right_expr}"

@dataclass
class ConstrPragma(Pragma):
    """Unary constraint on interface expression"""
    expr: str
    op: str  # "=", "<=", ">=", "%"
    value: Union[int, str]  # int or symbol name
    
    def evaluate(self, interfaces: Dict[str, Interface],
                env: Dict[str, int]) -> bool:
        expr_val = evaluate_expr(self.expr, interfaces, env)
        constraint_val = env.get(self.value, self.value) if isinstance(self.value, str) else self.value
        
        if self.op == "=":
            return expr_val == constraint_val
        elif self.op == "<=":
            return expr_val <= constraint_val
        elif self.op == ">=":
            return expr_val >= constraint_val
        elif self.op == "%":
            return expr_val % constraint_val == 0
        else:
            raise ValueError(f"Unknown operator: {self.op}")

def evaluate_expr(expr: str, interfaces: Dict[str, Interface], 
                 env: Dict[str, int]) -> int:
    """Evaluate expression with interface references"""
    # Parse expression AST
    # Handle interface[dim] references
    # Evaluate arithmetic
    # Return integer result
```

**Tasks:**
- [ ] Implement Pragma base class
- [ ] Implement TiePragma with expression evaluation
- [ ] Implement ConstrPragma with operators
- [ ] Create expression parser/evaluator
- [ ] Add pragma parsing from strings
- [ ] Write tests for various constraint types

### Step 1.4: Kernel Definition
**File:** `brainsmith/core/dataflow/core/kernel.py`

```python
"""Unified kernel definition"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from .interface import Interface
from .pragma import Pragma

@dataclass
class Kernel:
    """Hardware kernel with interfaces and constraints"""
    
    # Identity
    name: str
    hw_module: Optional[str] = None
    
    # Interfaces
    interfaces: List[Interface] = field(default_factory=list)
    
    # Timing characteristics
    latency_cycles: Tuple[int, int] = (1, 1)  # (worst_case, average)
    calculation_ii: Optional[int] = None
    execution_ii: Optional[int] = None
    
    # Pipeline costs
    priming_cycles: int = 0
    flush_cycles: int = 0
    
    # Constraints
    pragmas: List[Pragma] = field(default_factory=list)
    pragma_env: Dict[str, int] = field(default_factory=dict)
    
    # Resource estimates
    resources: Dict[str, float] = field(default_factory=dict)
    
    def validate(self) -> None:
        """Validate kernel configuration"""
        # Check interface names are unique
        names = [intf.name for intf in self.interfaces]
        if len(names) != len(set(names)):
            raise ValueError("Duplicate interface names")
        
        # Validate all pragmas
        intf_dict = {intf.name: intf for intf in self.interfaces}
        for pragma in self.pragmas:
            if not pragma.evaluate(intf_dict, self.pragma_env):
                raise ValueError(f"Pragma violation: {pragma.to_string()}")
    
    @property
    def input_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.direction == InterfaceDirection.INPUT]
    
    @property
    def output_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.direction == InterfaceDirection.OUTPUT]
    
    @property
    def weight_interfaces(self) -> List[Interface]:
        return [i for i in self.interfaces if i.direction == InterfaceDirection.WEIGHT]
```

**Tasks:**
- [ ] Implement Kernel dataclass
- [ ] Add validation method
- [ ] Implement interface filtering properties
- [ ] Add methods for calculating derived timing
- [ ] Write comprehensive tests

### Step 1.5: Dataflow Graph
**File:** `brainsmith/core/dataflow/core/graph.py`

```python
"""Dataflow graph representation"""

import networkx as nx
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

@dataclass
class DataflowEdge:
    """Edge between kernel interfaces"""
    producer_kernel: str
    producer_intf: str
    consumer_kernel: str
    consumer_intf: str
    buffer_depth: Optional[int] = None

@dataclass
class DataflowGraph:
    """Graph of connected kernels"""
    
    def __init__(self, kernels: List[Kernel]):
        self.kernels = {k.name: k for k in kernels}
        self.graph = nx.DiGraph()
        
        # Add nodes
        for kernel in kernels:
            self.graph.add_node(kernel.name, kernel=kernel)
    
    def add_edge(self, producer_kernel: str, producer_intf: str,
                 consumer_kernel: str, consumer_intf: str,
                 buffer_depth: Optional[int] = None):
        """Add connection between kernels"""
        edge = DataflowEdge(producer_kernel, producer_intf,
                           consumer_kernel, consumer_intf, buffer_depth)
        self.graph.add_edge(producer_kernel, consumer_kernel, 
                           edge=edge, 
                           producer_intf=producer_intf,
                           consumer_intf=consumer_intf)
    
    def validate(self):
        """Validate graph connectivity and interface compatibility"""
        # Check all edges connect compatible interfaces
        # Verify no dangling interfaces
        # Check for cycles if needed
```

**Tasks:**
- [ ] Implement DataflowEdge class
- [ ] Implement DataflowGraph with networkx
- [ ] Add edge validation
- [ ] Add graph analysis methods
- [ ] Write tests for graph construction

## Phase 2: ADFG Integration (Week 3-4)

### Step 2.1: ADFG Actor Abstraction
**File:** `brainsmith/core/dataflow/adfg/actor.py`

```python
"""ADFG actor representation"""

from dataclasses import dataclass
from typing import Dict, List, Tuple
from ..core.kernel import Kernel

@dataclass
class ADFGActor:
    """Actor in affine dataflow graph"""
    name: str
    wcet: int  # Worst-case execution time
    rates: Dict[str, List[int]]  # Interface name -> CSDF rates
    priority: Optional[int] = None
    
    @classmethod
    def from_kernel(cls, kernel: Kernel) -> "ADFGActor":
        """Convert kernel to ADFG actor"""
        rates = {}
        
        for intf in kernel.interfaces:
            rates[intf.name] = intf.rate_pattern
            
        return cls(
            name=kernel.name,
            wcet=kernel.latency_cycles[0],
            rates=rates,
            priority=kernel.latency_cycles[0]  # Default: DM priority
        )
    
    @property
    def is_csdf(self) -> bool:
        """Check if actor has cyclo-static behavior"""
        return any(len(r) > 1 for r in self.rates.values())
```

**Tasks:**
- [ ] Implement ADFGActor class
- [ ] Add conversion from Kernel
- [ ] Add CSDF detection
- [ ] Implement rate calculations

### Step 2.2: CSDF Support
**File:** `brainsmith/core/dataflow/adfg/csdf.py`

```python
"""Cyclo-static dataflow support"""

from typing import List, Tuple
import numpy as np

def compute_repetition_vector(actors: List[ADFGActor], 
                            edges: List[DataflowEdge]) -> Dict[str, int]:
    """Compute repetition vector for CSDF graph"""
    # Build topology matrix
    # Solve for repetition vector
    # Return minimal integer solution

def compute_phase_periods(rates: List[int], base_period: int) -> List[int]:
    """Compute per-phase periods for CSDF"""
    # Calculate period for each phase
    # Based on token production rate

def csdf_buffer_bounds(prod_rates: List[int], cons_rates: List[int],
                      prod_period: int, cons_period: int) -> int:
    """Calculate buffer size for CSDF edge"""
    # Implement phase-wise analysis
    # Return maximum buffer occupancy
```

**Tasks:**
- [ ] Implement repetition vector calculation
- [ ] Implement phase period computation
- [ ] Implement CSDF buffer sizing
- [ ] Add helper functions for rate analysis

### Step 2.3: SRTA Scheduler
**File:** `brainsmith/core/dataflow/adfg/scheduler.py`

```python
"""SRTA (Shortest Remaining Time Adjustment) scheduler"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math

@dataclass
class ScheduleResult:
    """Result of scheduling analysis"""
    periods: Dict[str, int]  # Actor name -> period
    utilization: Dict[str, float]  # Actor name -> utilization
    schedulable: bool
    hyperperiod: int

class SRTAScheduler:
    """Deadline-monotonic scheduler with SRTA"""
    
    def __init__(self, actors: List[ADFGActor]):
        self.actors = {a.name: a for a in actors}
        self._assign_priorities()
    
    def _assign_priorities(self):
        """Assign DM priorities (lower WCET = higher priority)"""
        sorted_actors = sorted(self.actors.values(), 
                             key=lambda a: a.wcet)
        for i, actor in enumerate(sorted_actors):
            actor.priority = i
    
    def schedule(self, target_throughput: Optional[float] = None) -> ScheduleResult:
        """Find minimal periods that satisfy timing constraints"""
        
        # Start with initial period estimates
        periods = self._initial_periods(target_throughput)
        
        # SRTA iteration
        converged = False
        iteration = 0
        max_iterations = 100
        
        while not converged and iteration < max_iterations:
            new_periods = {}
            schedulable = True
            
            for actor_name, actor in self.actors.items():
                # Calculate response time with interference
                response_time = self._calculate_response_time(
                    actor, periods
                )
                
                # Check schedulability
                if response_time > periods[actor_name]:
                    schedulable = False
                    # Adjust period
                    new_periods[actor_name] = math.ceil(response_time)
                else:
                    new_periods[actor_name] = periods[actor_name]
            
            # Check convergence
            converged = (new_periods == periods)
            periods = new_periods
            iteration += 1
        
        # Calculate utilization
        utilization = {
            name: actor.wcet / periods[name]
            for name, actor in self.actors.items()
        }
        
        # Calculate hyperperiod
        hyperperiod = math.lcm(*periods.values())
        
        return ScheduleResult(
            periods=periods,
            utilization=utilization,
            schedulable=schedulable and converged,
            hyperperiod=hyperperiod
        )
    
    def _calculate_response_time(self, actor: ADFGActor, 
                               periods: Dict[str, int]) -> int:
        """Calculate worst-case response time with interference"""
        # Implement response time analysis
        # Account for higher priority interference
```

**Tasks:**
- [ ] Implement SRTA scheduler class
- [ ] Add priority assignment (DM)
- [ ] Implement response time analysis
- [ ] Add convergence detection
- [ ] Write tests with known examples

### Step 2.4: Buffer Sizing ILP
**File:** `brainsmith/core/dataflow/adfg/buffer_sizing.py`

```python
"""ILP-based buffer sizing"""

from typing import Dict, List, Tuple, Optional
import pulp  # or ortools

@dataclass
class BufferSolution:
    """Solution to buffer sizing problem"""
    buffer_sizes: Dict[Tuple[str, str], int]  # (prod, cons) -> size
    total_memory: int
    solver_status: str

class BufferSizer:
    """ILP-based buffer sizing for ADFG"""
    
    def __init__(self, graph: DataflowGraph, schedule: ScheduleResult):
        self.graph = graph
        self.schedule = schedule
    
    def size_buffers(self, memory_limit: Optional[int] = None) -> BufferSolution:
        """Compute minimal buffer sizes using ILP"""
        
        # Create ILP problem
        prob = pulp.LpProblem("BufferSizing", pulp.LpMinimize)
        
        # Variables: buffer size for each edge
        buffer_vars = {}
        for edge in self.graph.edges():
            var_name = f"buf_{edge[0]}_{edge[1]}"
            buffer_vars[edge] = pulp.LpVariable(var_name, 
                                               lowBound=1, 
                                               cat='Integer')
        
        # Objective: minimize total buffer memory
        prob += pulp.lpSum(buffer_vars.values())
        
        # Constraints: affine relations
        for edge in self.graph.edges():
            # Get producer/consumer info
            prod_kernel = self.graph.nodes[edge[0]]['kernel']
            cons_kernel = self.graph.nodes[edge[1]]['kernel']
            edge_data = self.graph.edges[edge]
            
            # Add affine constraints
            self._add_affine_constraints(
                prob, buffer_vars[edge],
                prod_kernel, cons_kernel,
                edge_data['producer_intf'],
                edge_data['consumer_intf']
            )
        
        # Memory limit constraint
        if memory_limit:
            prob += pulp.lpSum(buffer_vars.values()) <= memory_limit
        
        # Solve
        prob.solve()
        
        # Extract solution
        buffer_sizes = {
            edge: int(var.varValue) 
            for edge, var in buffer_vars.items()
        }
        
        return BufferSolution(
            buffer_sizes=buffer_sizes,
            total_memory=sum(buffer_sizes.values()),
            solver_status=pulp.LpStatus[prob.status]
        )
```

**Tasks:**
- [ ] Implement BufferSizer class
- [ ] Add ILP formulation
- [ ] Implement affine constraints
- [ ] Add CSDF support
- [ ] Test with various graph topologies

## Phase 3: Design Space Exploration (Week 5-6)

### Step 3.1: Parallelism Configuration
**File:** `brainsmith/core/dataflow/dse/config.py`

```python
"""Configuration for design space exploration"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from ..core.kernel import Kernel

@dataclass
class ParallelismConfig:
    """Parallelism configuration for kernels"""
    
    # Per-interface parallelism (interface name -> value)
    interface_pars: Dict[str, int]
    
    # Global resource constraints
    total_bandwidth: Optional[int] = None  # bytes/sec
    total_dsp: Optional[int] = None
    total_bram: Optional[int] = None
    total_lut: Optional[int] = None
    
    def apply_to_kernel(self, kernel: Kernel) -> Kernel:
        """Apply parallelism config to kernel"""
        # Create new kernel with updated stream_dims
        new_interfaces = []
        
        for intf in kernel.interfaces:
            if intf.name in self.interface_pars:
                # Update stream dimensions
                new_stream_dims = self._compute_stream_dims(
                    intf, self.interface_pars[intf.name]
                )
                new_intf = replace(intf, stream_dims=new_stream_dims)
            else:
                new_intf = intf
            new_interfaces.append(new_intf)
        
        return replace(kernel, interfaces=new_interfaces)
    
    def validate(self, kernel: Kernel) -> bool:
        """Check if config satisfies kernel constraints"""
        # Apply config
        configured = self.apply_to_kernel(kernel)
        
        # Validate pragmas
        try:
            configured.validate()
            return True
        except ValueError:
            return False

@dataclass 
class DSEConstraints:
    """Constraints for design space exploration"""
    min_throughput: Optional[float] = None  # tokens/sec
    max_latency: Optional[int] = None  # cycles
    max_power: Optional[float] = None  # watts
    resource_limits: Dict[str, int] = field(default_factory=dict)
```

**Tasks:**
- [ ] Implement ParallelismConfig class
- [ ] Add config application logic
- [ ] Implement constraint validation
- [ ] Add resource estimation methods

### Step 3.2: Design Space Explorer
**File:** `brainsmith/core/dataflow/dse/explorer.py`

```python
"""Design space exploration engine"""

from itertools import product
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class DSEResult:
    """Result of design space exploration"""
    config: ParallelismConfig
    metrics: Dict[str, float]
    feasible: bool

class DesignSpaceExplorer:
    """Explore parallelism configurations"""
    
    def __init__(self, graph: DataflowGraph, constraints: DSEConstraints):
        self.graph = graph
        self.constraints = constraints
        self.scheduler = SRTAScheduler(self._create_actors())
        
    def explore(self, 
               min_par: int = 1,
               max_par: int = 64,
               par_step: int = 1) -> List[DSEResult]:
        """Explore design space"""
        
        results = []
        
        # Generate candidate configurations
        for config in self._generate_configs(min_par, max_par, par_step):
            # Apply configuration
            configured_graph = self._apply_config(config)
            
            # Check pragma constraints
            if not self._validate_pragmas(configured_graph):
                continue
            
            # Schedule
            schedule = self._schedule_graph(configured_graph)
            
            # Evaluate performance
            metrics = self._evaluate_config(configured_graph, schedule)
            
            # Check constraints
            feasible = self._check_constraints(metrics)
            
            results.append(DSEResult(
                config=config,
                metrics=metrics,
                feasible=feasible
            ))
        
        # Filter to Pareto frontier
        return self._pareto_filter(results)
    
    def _generate_configs(self, min_par: int, max_par: int, 
                         step: int) -> List[ParallelismConfig]:
        """Generate candidate configurations"""
        configs = []
        
        # Get all interfaces that need parallelism
        interfaces = []
        for kernel in self.graph.kernels.values():
            for intf in kernel.interfaces:
                if intf.direction in [InterfaceDirection.INPUT, 
                                    InterfaceDirection.WEIGHT]:
                    interfaces.append((kernel.name, intf.name))
        
        # Generate all combinations
        par_values = range(min_par, max_par + 1, step)
        
        for pars in product(par_values, repeat=len(interfaces)):
            interface_pars = {
                f"{k}_{i}": p 
                for (k, i), p in zip(interfaces, pars)
            }
            configs.append(ParallelismConfig(interface_pars))
        
        return configs
    
    def _pareto_filter(self, results: List[DSEResult]) -> List[DSEResult]:
        """Filter results to Pareto-optimal frontier"""
        # Multi-objective optimization
        # Keep non-dominated solutions
```

**Tasks:**
- [ ] Implement DesignSpaceExplorer class
- [ ] Add configuration generation
- [ ] Implement scheduling integration
- [ ] Add performance evaluation
- [ ] Implement Pareto filtering
- [ ] Write comprehensive tests

### Step 3.3: Performance Evaluator
**File:** `brainsmith/core/dataflow/dse/evaluator.py`

```python
"""Performance evaluation for configurations"""

from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class PerformanceMetrics:
    """Performance metrics for a configuration"""
    throughput: float  # inferences/sec
    latency: int  # cycles
    fps: float  # frames/sec
    resource_usage: Dict[str, float]
    power_estimate: float

class PerformanceEvaluator:
    """Evaluate performance of configurations"""
    
    def evaluate(self, graph: DataflowGraph, 
                schedule: ScheduleResult,
                batch_size: int = 1) -> PerformanceMetrics:
        """Compute performance metrics"""
        
        # Throughput from schedule
        throughput = self._compute_throughput(schedule)
        
        # Latency including pipeline effects
        latency = self._compute_latency(graph, schedule, batch_size)
        
        # Account for sparsity
        effective_throughput = self._apply_sparsity(graph, throughput)
        
        # Resource usage
        resources = self._estimate_resources(graph)
        
        # Power estimate
        power = self._estimate_power(graph, schedule)
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            fps=effective_throughput,
            resource_usage=resources,
            power_estimate=power
        )
    
    def _compute_latency(self, graph: DataflowGraph,
                        schedule: ScheduleResult,
                        batch_size: int) -> int:
        """Calculate end-to-end latency"""
        total = 0
        
        # Add priming cycles
        for kernel in graph.kernels.values():
            total += kernel.priming_cycles
        
        # Add execution cycles
        total += batch_size * schedule.hyperperiod
        
        # Add flush cycles
        for kernel in graph.kernels.values():
            total += kernel.flush_cycles
            
        return total
```

**Tasks:**
- [ ] Implement PerformanceEvaluator class
- [ ] Add throughput calculation
- [ ] Implement latency computation
- [ ] Add sparsity handling
- [ ] Implement resource estimation

## Testing Strategy

### Unit Tests (Continuous)
For each component, create corresponding test file:

```python
# tests/test_core.py
def test_interface_creation():
    """Test basic interface creation"""
    intf = Interface(
        name="input",
        direction=InterfaceDirection.INPUT,
        dtype=DataType("INT16", 16),
        tensor_dims=(32, 512),
        block_dims=(512,),
        stream_dims=(16,)
    )
    assert intf.ipar == 16
    assert intf.rate_pattern == [32]  # 512/16

def test_pragma_evaluation():
    """Test pragma constraint checking"""
    # Create interfaces
    # Create pragmas
    # Test evaluation

def test_kernel_validation():
    """Test kernel with pragmas"""
    # Create kernel
    # Add conflicting pragmas
    # Assert validation fails
```

### Integration Tests (End of each phase)

```python
# tests/test_integration.py
def test_simple_chain():
    """Test linear chain of kernels"""
    # Create 3 kernels in sequence
    # Connect them
    # Schedule
    # Check buffer sizes

def test_complex_graph():
    """Test graph with branches and merges"""
    # Create complex topology
    # Add pragmas
    # Run DSE
    # Verify results
```

### Validation Tests (Phase 3)

```python
# tests/test_validation.py
def test_against_known_example():
    """Compare with hand-calculated example"""
    # Use example from ADFG paper
    # Verify schedule matches
    # Verify buffer sizes match

def test_migration_from_old_model():
    """Test migration from interface-based model"""
    # Load old model definition
    # Convert to new format
    # Verify same results
```

## Implementation Timeline

### Week 1-2: Core Implementation ✅ COMPLETED
- [x] Day 1-2: Basic types and Interface
- [x] Day 3-4: Pragma system
- [x] Day 5-6: Kernel class
- [x] Day 7-8: Dataflow graph
- [x] Day 9-10: Unit tests and debugging
- **Result**: 78 tests passing, all core components implemented

### Week 3-4: ADFG Integration ✅ COMPLETED
- [x] Day 1-2: Actor abstraction and CSDF
- [x] Day 3-5: SRTA scheduler
- [x] Day 6-8: Buffer sizing ILP
- [x] Day 9-10: Integration tests
- **Result**: 52 tests passing (31 active, 21 skipped due to optional PuLP), all ADFG components implemented

### Week 5-6: DSE Framework
- [ ] Day 1-2: Configuration management
- [ ] Day 3-4: Design space explorer
- [ ] Day 5-6: Performance evaluator
- [ ] Day 7-8: Pareto optimization
- [ ] Day 9-10: End-to-end validation

## Success Criteria

1. **Correctness**: All unit tests pass, integration tests match expected results
2. **Performance**: DSE can explore 1000+ configurations in < 1 minute
3. **Accuracy**: Buffer sizes within 5% of theoretical minimum
4. **Usability**: Clean API that's easier than current approaches
5. **Compatibility**: Can represent all existing kernels

## Next Steps After Implementation

1. **Migration Tools**: Create converters from existing kernel definitions
2. **Integration**: Connect to RTL parser and FINN generation
3. **Optimization**: Add caching and parallel evaluation
4. **Extensions**: Add hierarchical memory, power models
5. **Documentation**: Create user guide and API reference