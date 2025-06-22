# Unified Kernel Modeling Framework: Technical Implementation Guide

## Overview

This document provides a comprehensive technical guide to the Unified Kernel Modeling Framework implementation. It covers the three-phase development, architectural decisions, key algorithms, extension points, and integration strategies for developers who want to understand, modify, or extend the framework.

## Architecture Overview

### System Architecture

The framework follows a layered architecture with clear separation of concerns:

```
┌─────────────────────────────────────────────────────────────┐
│                    Application Layer                        │
├─────────────────────┬─────────────────────┬─────────────────┤
│   Phase 3: DSE     │   Integration       │    Examples     │
│   - Explorer        │   - hw_kernel_gen   │   - CNN         │
│   - Evaluator       │   - ONNX Import     │   - BERT        │
│   - Configuration   │   - RTL Generation  │   - Custom      │
├─────────────────────┼─────────────────────┴─────────────────┤
│   Phase 2: ADFG    │                                       │
│   - Actor Model    │         Scheduling Layer              │
│   - SRTA Scheduler  │                                       │
│   - Buffer ILP      │                                       │
├─────────────────────┼─────────────────────┬─────────────────┤
│   Phase 1: Core    │    Validation       │    Utilities    │
│   - Interface       │    - Pragmas        │    - Types      │
│   - Kernel          │    - Constraints    │    - Shapes     │
│   - Graph           │    - Analysis       │    - DataTypes  │
└─────────────────────┴─────────────────────┴─────────────────┘
```

### Module Dependency Graph

```python
# Core dependencies (Phase 1)
brainsmith.core.dataflow.core.types
├── brainsmith.core.dataflow.core.interface
├── brainsmith.core.dataflow.core.pragma
├── brainsmith.core.dataflow.core.kernel
└── brainsmith.core.dataflow.core.graph

# ADFG dependencies (Phase 2)
brainsmith.core.dataflow.adfg.actor
├── brainsmith.core.dataflow.adfg.csdf
├── brainsmith.core.dataflow.adfg.scheduler
└── brainsmith.core.dataflow.adfg.buffer_ilp

# DSE dependencies (Phase 3)
brainsmith.core.dataflow.dse.config
├── brainsmith.core.dataflow.dse.evaluator
└── brainsmith.core.dataflow.dse.explorer
```

### Key Design Patterns

1. **Builder Pattern**: Used for constructing complex kernels and graphs
2. **Strategy Pattern**: Different scheduling algorithms and DSE objectives
3. **Observer Pattern**: Progress callbacks during DSE exploration
4. **Factory Pattern**: Creating actors from kernels and configurations
5. **Visitor Pattern**: AST traversal for pragma evaluation

## Phase 1: Core Data Structures

### Interface Implementation

The `Interface` class is the foundation of the framework:

```python
@dataclass
class Interface:
    name: str
    direction: InterfaceDirection
    dtype: DataType
    tensor_dims: Shape
    block_dims: Union[Shape, RaggedShape]  # SDF or CSDF
    stream_dims: Shape = field(default_factory=lambda: (1,))
    skip_prob: Optional[List[float]] = None
    
    def __post_init__(self):
        """Comprehensive validation on creation"""
        self._validate_dimensions()
        self._validate_tiling()
        self._validate_sparsity()
```

#### Key Algorithms

**Algorithm 1: Dimension Validation**
```python
def _validate_dimensions(self) -> None:
    """Validate dimensional consistency"""
    # Check tensor dimensions are positive
    if not all(d > 0 for d in self.tensor_dims):
        raise ValueError("All tensor dimensions must be positive")
    
    # Validate block dimensions
    if isinstance(self.block_dims, list):  # CSDF
        for phase_blocks in self.block_dims:
            if len(phase_blocks) != len(self.tensor_dims):
                raise ValueError("Block dims must match tensor dims length")
    else:  # SDF
        if len(self.block_dims) != len(self.tensor_dims):
            raise ValueError("Block dims must match tensor dims length")
    
    # Check stream dimensions
    if len(self.stream_dims) != len(self.tensor_dims):
        raise ValueError("Stream dims must match tensor dims length")
```

**Algorithm 2: Tiling Invariant Checking**
```python
def _validate_tiling(self) -> None:
    """Verify tiling invariants hold"""
    # Get effective block dimensions
    if isinstance(self.block_dims, list):
        block_dims = self.block_dims[0]  # Use first phase
    else:
        block_dims = self.block_dims
    
    # Check tensor-block tiling
    for i, (tensor_dim, block_dim) in enumerate(zip(self.tensor_dims, block_dims)):
        if tensor_dim % block_dim != 0:
            raise ValueError(
                f"Tensor dimension {i} ({tensor_dim}) not divisible "
                f"by block dimension ({block_dim})"
            )
    
    # Check block-stream tiling
    for i, (block_dim, stream_dim) in enumerate(zip(block_dims, self.stream_dims)):
        if block_dim % stream_dim != 0:
            raise ValueError(
                f"Block dimension {i} ({block_dim}) not divisible "
                f"by stream dimension ({stream_dim})"
            )
```

### Pragma System Architecture

The pragma system uses AST-based expression evaluation:

```python
class PragmaVisitor(ast.NodeVisitor):
    """AST visitor for evaluating interface expressions"""
    
    def __init__(self, interfaces: Dict[str, Interface], env: Dict[str, int]):
        self.interfaces = interfaces
        self.env = env
    
    def visit_Subscript(self, node: ast.Subscript) -> int:
        """Handle interface[dimension] expressions"""
        # Extract interface name
        if isinstance(node.value, ast.Name):
            intf_name = node.value.id
        else:
            raise ValueError("Only simple interface names supported")
        
        # Get interface
        if intf_name not in self.interfaces:
            raise ValueError(f"Unknown interface: {intf_name}")
        intf = self.interfaces[intf_name]
        
        # Extract dimension index
        if isinstance(node.slice, ast.Constant):
            idx = node.slice.value
        elif hasattr(node.slice, 'value'):  # Python < 3.9
            idx = node.slice.value.value
        else:
            raise ValueError("Only constant indices supported")
        
        # Return tensor dimension
        return intf.tensor_dims[idx]
```

#### Custom Pragma Development

To add new pragma types:

1. **Inherit from base Pragma class**:
```python
class CustomPragma(Pragma):
    def __init__(self, expression: str, operation: str, value: Union[str, int]):
        self.expression = expression
        self.operation = operation  
        self.value = value
    
    def evaluate(self, interfaces: Dict[str, Interface], 
                env: Dict[str, int]) -> bool:
        # Implement custom evaluation logic
        pass
    
    def to_string(self) -> str:
        return f"CUSTOM {self.expression} {self.operation} {self.value}"
```

2. **Register with parser**:
```python
def parse_pragma(pragma_str: str) -> Pragma:
    parts = pragma_str.strip().split()
    pragma_type = parts[0].upper()
    
    if pragma_type == "CUSTOM":
        return CustomPragma(parts[1], parts[2], parts[3])
    # ... existing parsers
```

### Kernel Implementation Details

#### Resource Estimation

The kernel's resource estimation uses a sophisticated model:

```python
def estimate_resources(self) -> Dict[str, float]:
    """Estimate resource usage based on interfaces and parallelism"""
    resources = self.resources.copy()
    
    # Scale DSP usage by parallelism
    total_parallelism = 1
    for intf in self.interfaces:
        if intf.direction in [InterfaceDirection.INPUT, InterfaceDirection.WEIGHT]:
            total_parallelism *= prod(intf.stream_dims)
    
    if "DSP" in resources:
        resources["DSP"] *= total_parallelism
    
    # Estimate memory usage
    total_memory_bits = 0
    for intf in self.interfaces:
        bits_per_element = intf.dtype.bitwidth() if hasattr(intf.dtype, 'bitwidth') else 16
        elements = prod(intf.tensor_dims)
        total_memory_bits += elements * bits_per_element
    
    # Convert to BRAM blocks (18Kb each)
    resources["BRAM"] = total_memory_bits / (18 * 1024)
    
    return resources
```

#### Bandwidth Requirements

```python
def bandwidth_requirements(self) -> Dict[str, float]:
    """Calculate bandwidth per interface in bits/cycle"""
    bandwidth = {}
    
    for intf in self.interfaces:
        if intf.direction in [InterfaceDirection.INPUT, InterfaceDirection.OUTPUT]:
            # Elements per cycle
            elements_per_cycle = prod(intf.stream_dims)
            
            # Bits per element
            bits_per_element = intf.dtype.bitwidth() if hasattr(intf.dtype, 'bitwidth') else 16
            
            # Total bandwidth
            bandwidth[intf.name] = elements_per_cycle * bits_per_element
    
    return bandwidth
```

## Phase 2: ADFG Integration

### Actor Model Implementation

The ADFG actor model bridges the gap between dataflow kernels and scheduling:

```python
@dataclass
class ADFGActor:
    name: str
    rates: Dict[str, List[int]]  # Interface -> rate pattern
    min_latency: int
    max_latency: int
    
    @classmethod
    def from_kernel(cls, kernel: Kernel) -> 'ADFGActor':
        """Convert kernel to ADFG actor"""
        rates = {}
        
        for intf in kernel.interfaces:
            # Calculate rate as elements per firing
            if isinstance(intf.block_dims, list):  # CSDF
                rate_pattern = []
                for phase_blocks in intf.block_dims:
                    rate_pattern.append(prod(phase_blocks))
                rates[intf.name] = rate_pattern
            else:  # SDF
                rates[intf.name] = [prod(intf.block_dims)]
        
        return cls(
            name=kernel.name,
            rates=rates,
            min_latency=kernel.latency_cycles[1],  # average
            max_latency=kernel.latency_cycles[0]   # worst case
        )
```

### CSDF Support Implementation

#### Phase Period Computation

```python
def compute_phase_periods(actors: List[ADFGActor], 
                         edges: List[Tuple[str, str, str, str]]) -> Dict[str, List[int]]:
    """Compute period for each phase of each actor"""
    # Build rate matrix
    interfaces = set()
    for actor in actors:
        interfaces.update(actor.rates.keys())
    
    # For each connected pair, solve rate equations
    phase_periods = {}
    
    for actor in actors:
        max_phases = max(len(rates) for rates in actor.rates.values())
        periods = [1] * max_phases
        
        # Find constraints from connected edges
        for edge in edges:
            if edge[0] == actor.name:  # This actor is producer
                prod_rates = actor.rates[edge[1]]
                # Find consumer rates and balance
                for consumer in actors:
                    if consumer.name == edge[2]:
                        cons_rates = consumer.rates[edge[3]]
                        
                        # Balance rates across phases
                        for phase in range(max_phases):
                            prod_rate = prod_rates[phase % len(prod_rates)]
                            cons_rate = cons_rates[phase % len(cons_rates)]
                            
                            # Period must ensure rate balance
                            periods[phase] = lcm(periods[phase], 
                                               cons_rate // gcd(prod_rate, cons_rate))
        
        phase_periods[actor.name] = periods
    
    return phase_periods
```

### SRTA Scheduler Implementation

#### Core Scheduling Algorithm

```python
class SRTAScheduler:
    def analyze(self, actors: List[ADFGActor], 
               edges: List[Tuple[str, str, str, str]]) -> SchedulabilityResult:
        """Analyze schedulability using SRTA"""
        
        # Step 1: Compute repetition vector
        repetitions = self._compute_repetition_vector(actors, edges)
        
        # Step 2: Assign periods (Rate Monotonic)
        periods = self._assign_periods(actors, repetitions)
        
        # Step 3: Response time analysis
        response_times = {}
        schedulable = True
        
        # Sort by period (deadline monotonic)
        sorted_actors = sorted(actors, key=lambda a: periods[a.name])
        
        for i, actor in enumerate(sorted_actors):
            # Compute response time with interference
            response_time = self._compute_response_time(
                actor, periods[actor.name], sorted_actors[:i], periods
            )
            
            if response_time > periods[actor.name]:
                schedulable = False
                break
                
            response_times[actor.name] = response_time
        
        # Step 4: Compute hyperperiod
        hyperperiod = 1
        for period in periods.values():
            hyperperiod = lcm(hyperperiod, period)
        
        return SchedulabilityResult(
            schedulable=schedulable,
            hyperperiod=hyperperiod,
            actor_periods=periods,
            response_times=response_times,
            total_utilization=sum(
                repetitions[a.name] * a.max_latency / periods[a.name] 
                for a in actors
            )
        )
```

#### Response Time Computation

```python
def _compute_response_time(self, actor: ADFGActor, period: int,
                          higher_priority: List[ADFGActor],
                          periods: Dict[str, int]) -> int:
    """Compute response time with interference from higher priority actors"""
    
    # Initial response time (no interference)
    response_time = actor.max_latency
    
    # Iterative fixed-point computation
    max_iterations = 100
    for iteration in range(max_iterations):
        new_response_time = actor.max_latency
        
        # Add interference from each higher priority actor
        for hp_actor in higher_priority:
            interference_jobs = math.ceil(response_time / periods[hp_actor.name])
            new_response_time += interference_jobs * hp_actor.max_latency
        
        # Check for convergence
        if new_response_time == response_time:
            return response_time
        
        # Check for deadline miss
        if new_response_time > period:
            return float('inf')  # Unschedulable
        
        response_time = new_response_time
    
    # Didn't converge - conservative estimate
    return period + 1
```

### Buffer Sizing ILP

The buffer sizing uses Integer Linear Programming when PuLP is available:

```python
class BufferSizeILP:
    def solve(self, actors: List[ADFGActor], edges: List[Tuple],
             objective: str = "minimize_total") -> Dict[str, int]:
        """Solve buffer sizing using ILP"""
        
        try:
            import pulp
        except ImportError:
            return self._fallback_solution(actors, edges)
        
        # Create problem
        prob = pulp.LpProblem("BufferSizing", pulp.LpMinimize)
        
        # Variables: buffer size for each edge
        buffer_vars = {}
        for i, edge in enumerate(edges):
            edge_name = f"{edge[0]}_{edge[1]}_to_{edge[2]}_{edge[3]}"
            buffer_vars[edge_name] = pulp.LpVariable(
                f"buffer_{i}", lowBound=1, cat='Integer'
            )
        
        # Constraints: ensure deadlock freedom
        for actor in actors:
            # Add constraints based on consumption/production patterns
            self._add_deadlock_constraints(prob, actor, buffer_vars, edges)
        
        # Objective function
        if objective == "minimize_total":
            prob += pulp.lpSum(buffer_vars.values())
        elif objective == "minimize_max":
            max_var = pulp.LpVariable("max_buffer", lowBound=1, cat='Integer')
            for var in buffer_vars.values():
                prob += var <= max_var
            prob += max_var
        
        # Solve
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        # Extract solution
        if prob.status == pulp.LpStatusOptimal:
            return {name: int(var.varValue) 
                   for name, var in buffer_vars.items()}
        else:
            return self._fallback_solution(actors, edges)
```

## Phase 3: Design Space Exploration

### Configuration Management

The configuration system manages parallelism across interfaces:

```python
class ParallelismConfig:
    def __init__(self, interface_pars: Dict[Tuple[str, str], int] = None,
                 global_par: int = 1):
        self.interface_pars = interface_pars or {}
        self.global_par = global_par
    
    def apply_to_kernel(self, kernel: Kernel, kernel_name: str) -> Kernel:
        """Apply configuration to create new kernel with modified parallelism"""
        new_interfaces = []
        
        for intf in kernel.interfaces:
            # Get parallelism for this interface
            key = (kernel_name, intf.name)
            parallelism = self.interface_pars.get(key, self.global_par)
            
            # Compute new stream dimensions
            new_stream_dims = self._compute_stream_dims(intf, parallelism)
            
            # Create modified interface
            new_intf = replace(intf, stream_dims=new_stream_dims)
            new_interfaces.append(new_intf)
        
        return replace(kernel, interfaces=new_interfaces)
```

#### Stream Dimension Computation

```python
def _compute_stream_dims(self, interface: Interface, parallelism: int) -> Shape:
    """Compute stream dimensions for given parallelism"""
    
    # Get current block dims (first phase for CSDF)
    if isinstance(interface.block_dims, list):
        block_dims = interface.block_dims[0]
    else:
        block_dims = interface.block_dims
    
    # Factor parallelism to match dimensions
    factors = self._factorize(parallelism)
    
    # Distribute factors across dimensions
    stream_dims = []
    remaining_par = parallelism
    
    for i, bdim in enumerate(block_dims):
        # Find best factor for this dimension
        best_factor = 1
        for f in factors:
            if f <= bdim and bdim % f == 0 and remaining_par % f == 0:
                best_factor = max(best_factor, f)
        
        stream_dims.append(best_factor)
        remaining_par //= best_factor
    
    # Put any remaining parallelism in the last dimension
    if remaining_par > 1:
        stream_dims[-1] *= remaining_par
    
    return tuple(stream_dims)

def _factorize(self, n: int) -> List[int]:
    """Get all factors of n in ascending order"""
    factors = []
    for i in range(1, int(math.sqrt(n)) + 1):
        if n % i == 0:
            factors.append(i)
            if i != n // i:
                factors.append(n // i)
    return sorted(factors)
```

### Performance Evaluator

#### Core Evaluation Algorithm

```python
class PerformanceEvaluator:
    def evaluate(self, graph: DataflowGraph,
                schedule: Optional[SchedulabilityResult] = None,
                batch_size: int = 1,
                input_sparsity: Optional[Dict[str, float]] = None) -> PerformanceMetrics:
        """Evaluate performance of configured graph"""
        
        # Get or compute schedule
        if schedule is None:
            schedule = self._schedule_graph(graph)
        
        if not schedule.schedulable:
            return PerformanceMetrics(
                throughput=0.0,
                latency=float('inf'),
                fps=0.0,
                resource_usage=self._estimate_resources(graph),
                processor_utilization=schedule.total_utilization
            )
        
        # Compute core metrics
        throughput = self._compute_throughput(schedule, batch_size)
        latency = self._compute_latency(graph, schedule, batch_size)
        
        # Apply sparsity effects
        effective_throughput = self._apply_sparsity(graph, throughput, input_sparsity)
        
        # Resource and power estimates
        resources = self._estimate_resources(graph)
        power = self._estimate_power(graph, schedule, resources)
        
        return PerformanceMetrics(
            throughput=throughput,
            latency=latency,
            fps=effective_throughput,
            resource_usage=resources,
            power_estimate=power,
            processor_utilization=schedule.total_utilization,
            memory_bandwidth_utilization=self._compute_bandwidth_utilization(graph, schedule)
        )
```

#### Sparsity-Aware Performance

```python
def _apply_sparsity(self, graph: DataflowGraph,
                   base_throughput: float,
                   input_sparsity: Optional[Dict[str, float]]) -> float:
    """Apply sparsity effects to throughput calculation"""
    
    if not input_sparsity:
        return base_throughput
    
    # Find minimum density across critical inputs
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
```

### Design Space Explorer

#### Exploration Algorithm

```python
class DesignSpaceExplorer:
    def explore(self, config_space: Optional[ConfigurationSpace] = None,
               batch_size: int = 1,
               progress_callback: Optional[Callable[[int, int], None]] = None) -> List[DSEResult]:
        """Explore design space systematically"""
        
        # Generate configuration space if not provided
        if config_space is None:
            config_space = self._generate_default_space()
        
        # Generate all candidate configurations
        configs = config_space.generate_configs()
        
        results = []
        for i, config in enumerate(configs):
            if progress_callback:
                progress_callback(i, len(configs))
            
            # Check cache first
            cache_key = self._config_cache_key(config)
            if cache_key in self._cache:
                results.append(self._cache[cache_key])
                continue
            
            # Evaluate configuration
            result = self._evaluate_config(config, batch_size)
            
            # Cache and store result
            self._cache[cache_key] = result
            results.append(result)
        
        # Sort by feasibility then throughput
        results.sort(key=lambda r: (-r.feasible, -r.metrics.throughput))
        
        return results
```

#### Pareto Optimization

```python
def find_pareto_optimal(self, results: List[DSEResult],
                       objectives: Optional[List[str]] = None) -> List[DSEResult]:
    """Find Pareto-optimal configurations"""
    
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
                
                # Handle minimization vs maximization
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
```

## Extension Points

### Adding Custom Scheduling Algorithms

To implement custom scheduling algorithms:

1. **Inherit from base scheduler**:
```python
class CustomScheduler:
    def analyze(self, actors: List[ADFGActor], 
               edges: List[Tuple]) -> SchedulabilityResult:
        # Implement custom scheduling logic
        pass
    
    def optimize_periods(self, actors: List[ADFGActor],
                        edges: List[Tuple],
                        objective: str = "hyperperiod") -> Dict[str, int]:
        # Implement period optimization
        pass
```

2. **Register with evaluator**:
```python
# In evaluator configuration
evaluator = PerformanceEvaluator()
evaluator.scheduler = CustomScheduler()
```

### Custom DSE Objectives

Add new optimization objectives:

```python
class CustomObjectiveEvaluator(PerformanceEvaluator):
    def evaluate(self, graph: DataflowGraph, **kwargs) -> PerformanceMetrics:
        metrics = super().evaluate(graph, **kwargs)
        
        # Add custom metrics
        metrics.custom_score = self._compute_custom_score(graph)
        
        return metrics
    
    def _compute_custom_score(self, graph: DataflowGraph) -> float:
        # Implement custom objective computation
        pass
```

### Hardware Backend Integration

Integrate with hardware generation tools:

```python
class RTLGenerator:
    def __init__(self, framework_config: DSEResult):
        self.config = framework_config
    
    def generate_rtl(self) -> Dict[str, str]:
        """Generate RTL files from optimized configuration"""
        rtl_files = {}
        
        for kernel_name, kernel in self.config.configured_graph.kernels.items():
            # Generate RTL for each kernel
            rtl_files[f"{kernel_name}.v"] = self._generate_kernel_rtl(kernel)
        
        # Generate top-level interconnect
        rtl_files["top.v"] = self._generate_top_level(self.config.configured_graph)
        
        return rtl_files
```

## Testing Strategy

### Unit Test Organization

```
tests/
├── core/
│   ├── test_types.py          # Basic type operations
│   ├── test_interface.py      # Interface validation
│   ├── test_pragma.py         # Pragma evaluation
│   ├── test_kernel.py         # Kernel functionality  
│   └── test_graph.py          # Graph operations
├── adfg/
│   ├── test_actor.py          # Actor conversion
│   ├── test_csdf.py          # CSDF utilities
│   ├── test_scheduler.py      # SRTA scheduling
│   └── test_buffer_ilp.py     # Buffer sizing
├── dse/
│   ├── test_config.py         # Configuration management
│   ├── test_evaluator.py      # Performance evaluation
│   ├── test_explorer.py       # Design space exploration
│   └── test_e2e.py           # End-to-end workflows
└── integration/
    ├── test_adfg_integration.py    # Phase 1+2 integration
    ├── test_dse_integration.py     # Phase 2+3 integration
    └── test_full_workflow.py       # Complete workflows
```

### Test Categories

1. **Unit Tests**: Test individual components in isolation
2. **Integration Tests**: Test component interactions
3. **End-to-End Tests**: Test complete workflows
4. **Performance Tests**: Measure algorithm performance
5. **Property Tests**: Use hypothesis for property-based testing

### Example Test Pattern

```python
class TestInterface:
    def test_valid_interface_creation(self):
        """Test creating valid interfaces"""
        interface = Interface(
            name="test",
            direction=InterfaceDirection.INPUT,
            dtype=INT16,
            tensor_dims=(256, 256),
            block_dims=(16, 16),
            stream_dims=(4, 4)
        )
        
        assert interface.repetition_count() == 256  # 256*256 / 16*16
        assert interface.calculate_cii() == 16      # 16*16 / 4*4
    
    def test_invalid_tiling_raises_error(self):
        """Test that invalid tiling raises ValueError"""
        with pytest.raises(ValueError, match="not divisible"):
            Interface(
                name="invalid",
                direction=InterfaceDirection.INPUT,
                dtype=INT16,
                tensor_dims=(255, 256),  # Not divisible by block
                block_dims=(16, 16),
                stream_dims=(4, 4)
            )
    
    @pytest.mark.parametrize("skip_prob,expected_density", [
        ([0.0, 0.0], 1.0),
        ([0.5, 0.5], 0.5),
        ([0.2, 0.8, 0.0], 0.667)
    ])
    def test_sparsity_calculation(self, skip_prob, expected_density):
        """Test sparsity calculation with different patterns"""
        interface = Interface(
            name="sparse",
            direction=InterfaceDirection.INPUT,
            dtype=INT8,
            tensor_dims=(128,),
            block_dims=[(32,), (64,), (32,)],  # CSDF
            stream_dims=(8,),
            skip_prob=skip_prob
        )
        
        density = interface.average_density()
        assert abs(density - expected_density) < 0.01
```

## Performance Benchmarking

### Benchmarking Framework

```python
class FrameworkBenchmark:
    def __init__(self):
        self.results = {}
    
    def benchmark_dse_scalability(self, max_interfaces: int = 20):
        """Benchmark DSE performance vs problem size"""
        for n_interfaces in range(2, max_interfaces + 1, 2):
            graph = self._create_test_graph(n_interfaces)
            
            start_time = time.time()
            explorer = DesignSpaceExplorer(graph, DSEConstraints())
            results = explorer.explore()
            end_time = time.time()
            
            self.results[f"dse_{n_interfaces}"] = {
                "time": end_time - start_time,
                "configs": len(results),
                "feasible": sum(1 for r in results if r.feasible)
            }
    
    def benchmark_scheduling_scalability(self, max_actors: int = 50):
        """Benchmark scheduling performance vs graph size"""
        for n_actors in range(5, max_actors + 1, 5):
            actors, edges = self._create_test_adfg(n_actors)
            
            start_time = time.time()
            scheduler = SRTAScheduler()
            result = scheduler.analyze(actors, edges)
            end_time = time.time()
            
            self.results[f"sched_{n_actors}"] = {
                "time": end_time - start_time,
                "schedulable": result.schedulable,
                "hyperperiod": result.hyperperiod
            }
```

## Integration Guide

### Connecting to hw_kernel_gen

The framework integrates with the existing hw_kernel_gen system:

```python
class HWKGIntegration:
    def __init__(self, dse_result: DSEResult):
        self.optimized_config = dse_result
    
    def generate_hwkg_metadata(self) -> Dict[str, Any]:
        """Generate metadata for hw_kernel_gen"""
        metadata = {}
        
        for kernel_name, kernel in self.optimized_config.configured_graph.kernels.items():
            kernel_metadata = {
                "name": kernel.name,
                "hw_module": kernel.hw_module,
                "interfaces": [],
                "resources": kernel.resources,
                "timing": {
                    "latency_cycles": kernel.latency_cycles,
                    "calculation_ii": kernel.calculation_ii,
                    "execution_ii": kernel.execution_ii
                }
            }
            
            # Convert interfaces to hw_kernel_gen format
            for intf in kernel.interfaces:
                intf_metadata = {
                    "name": intf.name,
                    "direction": intf.direction.value,
                    "datatype": str(intf.dtype),
                    "tensor_dims": list(intf.tensor_dims),
                    "block_dims": list(intf.block_dims) if not isinstance(intf.block_dims, list) else intf.block_dims,
                    "stream_dims": list(intf.stream_dims),
                    "parallelism": prod(intf.stream_dims)
                }
                kernel_metadata["interfaces"].append(intf_metadata)
            
            metadata[kernel_name] = kernel_metadata
        
        return metadata
```

### ONNX Model Import

Import ONNX models into the framework:

```python
class ONNXImporter:
    def __init__(self, onnx_model_path: str):
        import onnx
        self.model = onnx.load(onnx_model_path)
    
    def create_dataflow_graph(self) -> DataflowGraph:
        """Convert ONNX model to dataflow graph"""
        graph = DataflowGraph()
        
        # Map ONNX operations to kernels
        for node in self.model.graph.node:
            kernel = self._convert_onnx_node(node)
            if kernel:
                graph.add_kernel(kernel)
        
        # Connect based on ONNX value flows
        self._create_edges(graph)
        
        return graph
    
    def _convert_onnx_node(self, node) -> Optional[Kernel]:
        """Convert ONNX node to Kernel"""
        if node.op_type == "Conv":
            return self._create_conv_kernel(node)
        elif node.op_type == "MatMul":
            return self._create_matmul_kernel(node)
        elif node.op_type == "Relu":
            return self._create_activation_kernel(node)
        # ... other operators
        
        return None
```

### RTL Generation Workflow

Complete workflow from algorithm to hardware:

```python
def algorithm_to_hardware_workflow(onnx_model_path: str, 
                                 target_constraints: DSEConstraints) -> Dict[str, str]:
    """Complete workflow from ONNX model to optimized RTL"""
    
    # Step 1: Import ONNX model
    importer = ONNXImporter(onnx_model_path)
    graph = importer.create_dataflow_graph()
    
    # Step 2: Validate and analyze
    graph.validate()
    path, latency = graph.get_critical_path()
    print(f"Critical path: {' -> '.join(path)} ({latency} cycles)")
    
    # Step 3: Design space exploration
    explorer = DesignSpaceExplorer(graph, target_constraints)
    results = explorer.explore()
    
    # Step 4: Select best configuration
    feasible = [r for r in results if r.feasible]
    if not feasible:
        raise ValueError("No feasible configurations found")
    
    best = max(feasible, key=lambda r: r.metrics.throughput)
    print(f"Best config: {best.metrics.throughput:.1f} GOPS @ {best.metrics.power_estimate:.1f}W")
    
    # Step 5: Generate hardware
    hwkg_integration = HWKGIntegration(best)
    metadata = hwkg_integration.generate_hwkg_metadata()
    
    rtl_generator = RTLGenerator(best)
    rtl_files = rtl_generator.generate_rtl()
    
    # Step 6: Generate synthesis scripts
    synth_scripts = generate_synthesis_scripts(metadata, target_constraints)
    
    return {**rtl_files, **synth_scripts}
```

## Debugging and Profiling

### Debug Utilities

```python
class FrameworkDebugger:
    @staticmethod
    def visualize_graph(graph: DataflowGraph, filename: str = "graph.png"):
        """Generate visual representation of dataflow graph"""
        try:
            import graphviz
        except ImportError:
            print("Install graphviz for visualization: pip install graphviz")
            return
        
        dot = graphviz.Digraph(comment='Dataflow Graph')
        
        # Add nodes
        for name, kernel in graph.kernels.items():
            label = f"{name}\\n{kernel.latency_cycles}"
            dot.node(name, label)
        
        # Add edges
        for edge in graph.edges.values():
            dot.edge(edge.producer_kernel, edge.consumer_kernel,
                    label=f"{edge.producer_intf}→{edge.consumer_intf}")
        
        dot.render(filename, view=True)
    
    @staticmethod
    def analyze_config_failures(results: List[DSEResult]) -> Dict[str, int]:
        """Analyze why configurations failed"""
        failure_counts = {}
        
        for result in results:
            if not result.feasible:
                for reason in result.violation_reasons:
                    # Extract failure type
                    failure_type = reason.split(':')[0] if ':' in reason else reason
                    failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1
        
        return dict(sorted(failure_counts.items(), key=lambda x: x[1], reverse=True))
```

### Performance Profiling

```python
class PerformanceProfiler:
    def __init__(self):
        self.timings = {}
    
    def profile_dse_exploration(self, explorer: DesignSpaceExplorer,
                              config_space: ConfigurationSpace):
        """Profile DSE exploration performance"""
        
        import cProfile
        import pstats
        
        profiler = cProfile.Profile()
        profiler.enable()
        
        results = explorer.explore(config_space)
        
        profiler.disable()
        
        # Analyze results
        stats = pstats.Stats(profiler)
        stats.sort_stats('cumulative')
        stats.print_stats(20)  # Top 20 functions
        
        return results
```

## Conclusion

This technical implementation guide provides a comprehensive overview of the Unified Kernel Modeling Framework's architecture, algorithms, and extension points. The three-phase development approach ensures a solid foundation while enabling advanced optimization capabilities.

Key implementation highlights:

1. **Rigorous Validation**: Mathematical constraints are enforced at every level
2. **Modular Design**: Clean separation between phases enables independent development
3. **Performance Focus**: Algorithms are designed for efficiency and scalability
4. **Extensibility**: Clear extension points for custom algorithms and objectives
5. **Integration Ready**: APIs designed for seamless integration with existing tools

The framework provides a solid foundation for automated FPGA accelerator design while maintaining the flexibility needed for research and development in emerging applications.