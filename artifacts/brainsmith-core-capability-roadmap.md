# BrainSmith Core: Capability Roadmap

## Current Foundation (What We Have)

### 🏗️ Infrastructure Layer
- **Plugin Registry**: Unified system supporting transforms, kernels, backends, steps
- **Framework Integration**: 243 pre-registered QONNX/FINN components
- **Metadata System**: Rich attribution and discovery capabilities
- **Namespace Support**: Multi-framework coexistence (brainsmith:, qonnx:, finn:)

### 🌳 Execution Layer  
- **Segment-Based Trees**: Efficient prefix-sharing execution model
- **Artifact Management**: Smart caching and sharing between branches
- **Blueprint Inheritance**: Composable configuration system
- **FINN Isolation**: Clean adapter pattern for framework quirks

### 🔧 Extensibility Points
- **Plugin Decorators**: Simple @transform, @kernel, @backend, @step registration
- **Kernel Inference Hook**: Ready for custom inference strategies
- **Tree Statistics**: Analytics-ready execution metrics
- **Multi-Backend Support**: Hardware implementation diversity

## Near-Term Capabilities (What It Enables Now)

### 1. Domain-Specific Optimizations
```python
@transform(
    name="FuseGELU", 
    stage="optimization",
    domain="transformer",
    hardware_aware=True
)
class FuseGELU(Transformation):
    """Custom transform leveraging metadata for discovery"""
```

### 2. Automated Design Space Exploration
```yaml
# Blueprint using multi-backend exploration
kernels:
  - Attention: [AttentionHLS, AttentionSystolic, AttentionSpatial]
  - FFN: [FFNHLS, FFNPipelined]
# Automatically explores 6 combinations
```

### 3. Framework Comparison Studies
```yaml
steps:
  - ["qonnx:Cleanup", "finn:Cleanup", "brainsmith:Cleanup"]
  - "quantize"
  - ["qonnx:Optimize", "finn:Optimize"]
# Compare framework implementations
```

### 4. Custom Hardware Backends
```python
@backend(
    name="ChiselMatMul",
    kernel="MatMul",
    language="chisel",
    target="ASIC"
)
class ChiselMatMul(MatMul, ChiselBackend):
    """New backend slots into existing system"""
```

## Medium-Term Vision (Next 6-12 Months)

### 1. Intelligent DSE with Metadata
```python
# Find optimal transforms for target
transforms = registry.find(
    'transform',
    stage='quantization',
    hardware_aware=True,
    target_latency='<10ms'
)
```

### 2. Multi-Framework Pipelines
```yaml
# Mix best-of-breed components
steps:
  - "qonnx:PrepareModel"         # QONNX's strength
  - "brainsmith:CustomQuantize"   # Our innovation
  - "finn:DataflowPartition"      # FINN's expertise
  - "mlir:OptimizeLoops"         # Future framework
```

### 3. Distributed Execution
```python
# Segments enable distributed exploration
executor = DistributedExecutor(
    worker_nodes=["gpu-0", "gpu-1", "cpu-cluster"],
    scheduler="ray"
)
results = executor.explore(tree)
```

### 4. Hardware Cost Models
```python
# Tree statistics feed cost estimation
stats = get_tree_stats(tree)
cost_model = HardwareCostModel(stats)
estimated_resources = cost_model.predict(
    target="ZCU104",
    frequency="200MHz"
)
```

## Long-Term Platform (2+ Years)

### 1. ML-Guided Optimization
```python
# Learn from execution patterns
optimizer = MLOptimizer()
optimizer.train(historical_executions)
suggested_pipeline = optimizer.recommend(
    model=bert,
    constraints={"latency": "<5ms", "power": "<10W"}
)
```

### 2. Universal Hardware IR
```python
# Kernels target multiple backends transparently
@kernel(name="UniversalConv")
class UniversalConv(HWCustomOp):
    def to_hls(self): ...
    def to_verilog(self): ...
    def to_spatial(self): ...
    def to_cgra(self): ...
```

### 3. Collaborative Research Platform
```python
# Researchers contribute optimizations
registry.publish(
    transform=MyNovelQuantization,
    paper="arxiv:2024.12345",
    reproducible=True
)

# Others can discover and use
novel_transforms = registry.find(
    'transform',
    tags=['published', 'quantization'],
    citations='>10'
)
```

### 4. Hardware/Software Co-Design
```yaml
# Unified DSE across stack
design_space:
  software:
    - compilers: [gcc, clang, icc]
    - optimizations: [O2, O3, Ofast]
  hardware:
    - kernels: [MatMul, Conv2D]
    - frequencies: [100MHz, 200MHz, 300MHz]
  system:
    - memory: [DDR4, HBM2]
    - interconnect: [AXI4, CHI]
```

## Why Current "Unused" Features Matter

| Feature | Current State | Future Value |
|---------|--------------|--------------|
| 243 Plugins | 20 used | Each is a research opportunity |
| Metadata | Rarely queried | Enables intelligent selection |
| Namespacing | Single framework | Multi-framework workflows |
| Tree Stats | Just logged | Feed ML optimizers |
| Multi-Backend | Single backend used | Hardware DSE |

## Development Priorities

### Phase 1: Developer Experience (Now)
- [ ] Plugin development guide
- [ ] Blueprint cookbook
- [ ] Extension examples
- [ ] API documentation

### Phase 2: Core Capabilities (Q1 2024)
- [ ] Kernel inference implementation
- [ ] Progress tracking API
- [ ] Result analysis tools
- [ ] Distributed execution support

### Phase 3: Intelligence Layer (Q2-Q3 2024)
- [ ] Cost models
- [ ] ML-guided optimization
- [ ] Performance prediction
- [ ] Automated tuning

### Phase 4: Ecosystem (Q4 2024+)
- [ ] Plugin marketplace
- [ ] Research integration
- [ ] Cloud execution
- [ ] Hardware vendor SDKs

## Success Metrics

1. **Adoption**: Number of research papers using BrainSmith
2. **Ecosystem**: Third-party plugins published
3. **Performance**: DSE speedup vs manual exploration
4. **Coverage**: Hardware targets supported

## Conclusion

BrainSmith Core's current architecture isn't overbuilt - it's **strategically built** for a future where:
- Every optimization is a plugin
- Frameworks cooperate instead of compete
- Hardware DSE is automated
- Research directly feeds production

The "unused" features are the **seeds** of this future platform.