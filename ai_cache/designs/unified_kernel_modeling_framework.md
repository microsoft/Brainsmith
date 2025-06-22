# Unified Kernel Modeling Framework - Design Document

## Executive Summary

This document presents a unified kernel modeling framework that combines:
1. The **Interface-Based Dataflow Model's** intuitive interface abstractions and data hierarchy
2. The **Affine Dataflow Graph's** rigorous mathematical foundations for scheduling and buffer sizing
3. The **Kernel Modeling Framework's** pragmatic constraint system and real-world extensions

The result is a comprehensive system that provides both ease of use and mathematical rigor for FPGA AI accelerator design.

## 1. Architecture Overview

### 1.1 Core Philosophy

The unified framework maintains a clear separation of concerns:

```
User Interface Layer (Intuitive API)
         ↓
Constraint & Validation Layer (Pragmas)
         ↓
Mathematical Modeling Layer (ADFG)
         ↓
Hardware Generation Layer (RTL/FINN)
```

### 1.2 Key Design Principles

1. **Progressive Refinement**: Start with simple interface definitions, progressively add constraints
2. **Mathematical Rigor**: All timing and buffer calculations backed by ADFG theory
3. **Hardware Realism**: Account for pipeline latency, sparsity, and FPGA-specific constraints
4. **Declarative Constraints**: Express requirements, not implementation details

## 2. Data Model

### 2.1 Unified Data Hierarchy

Combining the best of both interface-based and ADFG approaches:

```python
@dataclass
class DataHierarchy:
    """Unified data representation across all interfaces"""
    
    # From Interface-Based Model
    tensor_dims: Shape      # Full inference payload (TDIM)
    block_dims: Shape       # Kernel firing granularity (BDIM)
    stream_dims: Shape      # Per-cycle transfer (SDIM)
    
    # From ADFG (as derived properties)
    @property
    def tokens_per_inference(self) -> int:
        """Number of ADFG tokens (blocks) per tensor"""
        return prod(tensor_dims) // prod(block_dims)
    
    @property
    def cycles_per_token(self) -> int:
        """Initiation interval for one block"""
        return ceil(prod(block_dims) / prod(stream_dims))
```

### 2.2 Enhanced Interface Definition

```python
@dataclass
class Interface:
    """Unified interface combining all three approaches"""
    
    # Basic properties
    name: str
    direction: Literal["input", "output", "weight", "config"]
    dtype: DataType
    
    # Data hierarchy (from Interface-Based)
    tensor_dims: Shape
    block_dims: Union[Shape, List[Shape]]  # Can be ragged (from KM)
    stream_dims: Shape
    
    # Advanced features (from KM)
    skip_prob: List[float] = field(default_factory=list)  # Sparsity
    optional: bool = False  # For conditional interfaces
    
    # Derived ADFG properties
    @property
    def rate_pattern(self) -> List[int]:
        """CSDF rate pattern for ADFG analysis"""
        if isinstance(self.block_dims, list):
            return [prod(bd) // prod(self.stream_dims) for bd in self.block_dims]
        return [prod(self.block_dims) // prod(self.stream_dims)]
    
    @property
    def ipar(self) -> int:
        """Interface parallelism (flattened stream width)"""
        return prod(self.stream_dims)
```

### 2.3 Kernel Definition

```python
@dataclass
class Kernel:
    """Hardware kernel with full modeling capabilities"""
    
    # Identity
    name: str
    hw_module: str  # SystemVerilog module name
    
    # Interfaces
    interfaces: List[Interface]
    
    # Timing (combining both approaches)
    latency_cycles: Tuple[int, int]  # (worst_case, average)
    calculation_ii: Optional[int] = None  # From Interface-Based cII
    execution_ii: Optional[int] = None    # From Interface-Based eII
    
    # Pipeline costs (from KM)
    priming_cycles: int = 0
    flush_cycles: int = 0
    
    # Constraints (from KM)
    pragmas: List[Pragma] = field(default_factory=list)
    pragma_env: Dict[str, int] = field(default_factory=dict)
    
    # Resource estimates
    resources: Dict[str, float] = field(default_factory=dict)
    
    # ADFG integration
    def to_adfg_actor(self) -> ADFGActor:
        """Convert to ADFG actor for scheduling"""
        return ADFGActor(
            name=self.name,
            wcet=self.latency_cycles[0],
            rates={intf.name: intf.rate_pattern for intf in self.interfaces}
        )
```

## 3. Constraint System

### 3.1 Unified Pragma System

Building on KM's pragmas with Interface-Based concepts:

```python
# Interface relationship constraints (from KM)
TIE mat[1] vec          # Dimension equality
TIE input.block_dims output.block_dims  # Block size matching

# Value constraints (from KM)
CONSTR vec % BURST      # Alignment requirements
CONSTR mat[0] >= 16     # Minimum parallelism

# New unified constraints
RATIO input output 2:1  # Enforces specific production/consumption ratios
LATENCY <= 1000        # Maximum kernel latency constraint
THROUGHPUT >= 100      # Minimum throughput in tokens/cycle
```

### 3.2 Parallelism Parameters

Unifying Interface-Based (iPar, wPar) with KM's approach:

```python
@dataclass
class ParallelismConfig:
    """Design space parameters"""
    
    # Per-interface parallelism
    interface_pars: Dict[str, int]  # {"vec": 16, "mat": 64}
    
    # Global constraints
    total_bandwidth: int  # DDR/HBM limit
    total_dsp: int       # FPGA resource limit
    
    def validate(self, kernel: Kernel) -> bool:
        """Check if configuration satisfies all constraints"""
        # Validate pragmas
        for pragma in kernel.pragmas:
            if not pragma.evaluate(self):
                return False
        
        # Check resource limits
        estimated = kernel.estimate_resources(self)
        return (estimated["bandwidth"] <= self.total_bandwidth and
                estimated["dsp"] <= self.total_dsp)
```

## 4. Scheduling Integration

### 4.1 ADFG Scheduling with Extensions

```python
class UnifiedScheduler:
    """Combines ADFG theory with practical extensions"""
    
    def schedule(self, kernels: List[Kernel], config: ParallelismConfig):
        # Step 1: Convert to ADFG representation
        actors = [k.to_adfg_actor() for k in kernels]
        
        # Step 2: Apply SRTA for period calculation
        periods = self.srta_search(actors)
        
        # Step 3: Size buffers with ILP
        buffers = self.size_buffers_ilp(actors, periods)
        
        # Step 4: Apply sparsity adjustments
        if any(intf.skip_prob for k in kernels for intf in k.interfaces):
            periods = self.adjust_for_sparsity(periods, kernels)
        
        # Step 5: Calculate total latency with pipeline costs
        latency = self.calculate_total_latency(kernels, periods)
        
        return ScheduleResult(periods, buffers, latency)
```

### 4.2 Buffer Sizing with Ragged Tiles

```python
def size_buffer_unified(producer: Interface, consumer: Interface, 
                       prod_period: int, cons_period: int) -> int:
    """Unified buffer sizing accounting for ragged tiles"""
    
    if isinstance(producer.block_dims, list):
        # Handle ragged tiling (CSDF)
        return size_buffer_csdf(
            prod_rates=producer.rate_pattern,
            cons_rates=consumer.rate_pattern,
            prod_period=prod_period,
            cons_period=cons_period
        )
    else:
        # Simple case
        prod_rate = prod(producer.block_dims) // prod(producer.stream_dims)
        cons_rate = prod(consumer.block_dims) // prod(consumer.stream_dims)
        return ceil(lcm(prod_rate, cons_rate) * max(prod_period, cons_period) / 
                   min(prod_period, cons_period))
```

## 5. Design Space Exploration

### 5.1 Unified DSE Framework

```python
class UnifiedDSE:
    """Design space exploration combining all approaches"""
    
    def explore(self, model: DataflowModel, constraints: DSEConstraints):
        # Generate candidate configurations
        candidates = []
        
        for kernel in model.kernels:
            # Use Interface-Based parameter ranges
            ipar_range = range(1, kernel.max_ipar())
            wpar_range = range(1, kernel.max_wpar()) if kernel.has_weights() else [1]
            
            for ipar, wpar in product(ipar_range, wpar_range):
                config = self.generate_config(kernel, ipar, wpar)
                
                # Filter by pragmas (KM approach)
                if config.validate(kernel):
                    candidates.append(config)
        
        # Evaluate each candidate
        results = []
        for config in candidates:
            # Schedule with ADFG
            schedule = self.scheduler.schedule(model.kernels, config)
            
            # Evaluate metrics
            metrics = {
                "throughput": 1.0 / max(schedule.periods.values()),
                "latency": schedule.total_latency,
                "resources": self.estimate_resources(model, config),
                "power": self.estimate_power(model, config, schedule)
            }
            
            results.append((config, metrics))
        
        # Return Pareto frontier
        return self.pareto_filter(results)
```

## 6. Implementation Architecture

### 6.1 Module Structure

```
brainsmith/kernel_modeling/
├── core/
│   ├── interface.py      # Unified Interface class
│   ├── kernel.py         # Unified Kernel class
│   ├── pragma.py         # Constraint system
│   └── hierarchy.py      # Data hierarchy helpers
├── adfg/
│   ├── actor.py          # ADFG actor conversion
│   ├── scheduler.py      # SRTA implementation
│   ├── buffer_sizing.py  # ILP formulation
│   └── csdf.py          # Cyclo-static extensions
├── dse/
│   ├── explorer.py       # Design space exploration
│   ├── evaluator.py      # Performance evaluation
│   └── optimizer.py      # Pareto optimization
├── codegen/
│   ├── hw_custom_op.py   # FINN integration
│   ├── rtl_wrapper.py    # RTL generation
│   └── metadata.py       # Export to KernelMetadata
└── analysis/
    ├── performance.py    # Throughput/latency analysis
    ├── resources.py      # Resource estimation
    └── power.py         # Power modeling
```

### 6.2 Integration Points

1. **RTL Parser Integration**:
   ```python
   def parse_rtl_with_modeling(rtl_file: Path) -> Kernel:
       """Parse RTL and create Kernel with full modeling info"""
       # Parse interfaces and pragmas
       metadata = parse_rtl(rtl_file)
       
       # Create unified kernel
       return Kernel(
           name=metadata.name,
           interfaces=[create_interface(intf) for intf in metadata.interfaces],
           pragmas=parse_pragmas(metadata.pragmas),
           latency_cycles=extract_latency(metadata)
       )
   ```

2. **FINN Export**:
   ```python
   def export_to_finn(kernel: Kernel, config: ParallelismConfig) -> HWCustomOp:
       """Generate FINN HWCustomOp from unified model"""
       # Apply parallelism configuration
       configured = kernel.apply_config(config)
       
       # Generate code
       return generate_hw_custom_op(configured)
   ```

## 7. Example Usage

### 7.1 Simple MatMul Kernel

```python
# Define interfaces
vec_in = Interface(
    name="vec",
    direction="input",
    dtype=DataType("INT16"),
    tensor_dims=(1, 512),
    block_dims=(512,),
    stream_dims=(16,)  # 16 elements per cycle
)

mat_in = Interface(
    name="mat",
    direction="weight",
    dtype=DataType("INT16"),
    tensor_dims=(512, 512),
    block_dims=(512, 512),
    stream_dims=(16, 32)  # 16x32 tile per cycle
)

vec_out = Interface(
    name="out",
    direction="output",
    dtype=DataType("INT32"),
    tensor_dims=(512,),
    block_dims=(512,),
    stream_dims=(32,)  # 32 elements per cycle
)

# Create kernel with constraints
matmul = Kernel(
    name="MatMul",
    hw_module="matmul_unit",
    interfaces=[vec_in, mat_in, vec_out],
    latency_cycles=(1000, 800),  # worst-case, average
    priming_cycles=64,
    flush_cycles=32,
    pragmas=[
        TiePragma("mat[1]", "vec"),  # Matrix columns match vector size
        ConstrPragma("vec", "%", "BURST"),  # Align to burst size
        ConstrPragma("out", ">=", "32")  # Minimum output parallelism
    ],
    pragma_env={"BURST": 64, "SIMD": 16}
)

# Explore design space
dse = UnifiedDSE()
results = dse.explore(
    model=DataflowModel([matmul]),
    constraints=DSEConstraints(
        max_dsp=2000,
        max_bandwidth=25.6e9,  # 25.6 GB/s
        target_fps=100
    )
)

# Select best configuration
best_config, metrics = results[0]
print(f"Best configuration achieves {metrics['throughput']} tokens/s")
print(f"Latency: {metrics['latency']} cycles")
print(f"Resources: {metrics['resources']}")
```

### 7.2 Complex Multi-Kernel Graph

```python
# Build a transformer attention block
kernels = [
    create_kernel("QKV_Proj", ...),
    create_kernel("Attention", ...),
    create_kernel("Output_Proj", ...)
]

# Add inter-kernel constraints
graph = DataflowGraph(kernels)
graph.add_edge("QKV_Proj", "Attention", buffer_depth=1024)
graph.add_edge("Attention", "Output_Proj", buffer_depth=512)

# Schedule entire graph
scheduler = UnifiedScheduler()
schedule = scheduler.schedule(graph, config)

print(f"Total inference latency: {schedule.total_latency} cycles")
print(f"Throughput: {schedule.throughput} inferences/s")
```

## 8. Migration Path

### 8.1 From Interface-Based Dataflow

```python
# Old Interface-Based code
old_kernel = {
    "interfaces": {
        "input": {"tensor_dims": (B, M), "block_dims": (M,), "stream_dims": (S,)},
        "weight": {"tensor_dims": (N, M), "block_dims": (N, M), "stream_dims": (S, P)},
        "output": {"tensor_dims": (B, N), "block_dims": (N,), "stream_dims": (P,)}
    },
    "cII": calculate_cii(...),
    "eII": calculate_eii(...)
}

# Migrated to unified framework
new_kernel = Kernel(
    name="MyKernel",
    interfaces=[
        Interface("input", "input", dtype, (B, M), (M,), (S,)),
        Interface("weight", "weight", dtype, (N, M), (N, M), (S, P)),
        Interface("output", "output", dtype, (B, N), (N,), (P,))
    ],
    calculation_ii=old_kernel["cII"],
    execution_ii=old_kernel["eII"],
    latency_cycles=(calculate_latency(...), None)
)
```

### 8.2 From Current Kernel Modeling

```python
# Current KM code
km_kernel = Kernel(
    name="kernel",
    interfaces=[...],
    latency_cycles=(900, 600),
    pragmas=[TiePragma(...), ConstrPragma(...)]
)

# Already compatible! Just ensure interfaces have all hierarchy levels:
for intf in km_kernel.interfaces:
    if not hasattr(intf, 'tensor_dims'):
        intf.tensor_dims = infer_tensor_dims(intf)
```

## 9. Benefits of Unified Approach

1. **Complete Modeling**: From high-level tensor operations down to cycle-accurate RTL
2. **Mathematical Rigor**: ADFG theory ensures correctness of scheduling and buffering
3. **Practical Extensions**: Handles real-world effects like sparsity and pipeline latency
4. **Declarative Constraints**: Express requirements naturally without implementation details
5. **Progressive Refinement**: Start simple, add complexity as needed
6. **Tool Integration**: Clean interfaces to RTL parser, FINN, and analysis tools

## 10. Future Extensions

1. **Hierarchical Memories**: Model DRAM→BRAM→Register hierarchies
2. **Dynamic Scheduling**: Support data-dependent execution patterns
3. **Multi-Clock Domains**: Handle kernels with different clock frequencies
4. **Power Optimization**: Integrate dynamic voltage/frequency scaling
5. **Automated Pragma Inference**: Derive constraints from RTL analysis

## Conclusion

This unified framework combines the intuitive interface abstractions from the Interface-Based Dataflow Model, the mathematical rigor of Affine Dataflow Graphs, and the practical constraint system from the Kernel Modeling framework. The result is a comprehensive system that can model FPGA AI accelerators from high-level specifications down to cycle-accurate implementations, while maintaining both ease of use and formal correctness guarantees.