# Plugin System Architecture

## Zero-Overhead Plugin Registry (ASCII Art)

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            PLUGIN REGISTRY SYSTEM                            │
│                         Zero-Overhead Registration                           │
└─────────────────────────────────────────────────────────────────────────────┘
                                      │
    ┌─────────────────┬───────────────┴────────────────┬─────────────────┐
    ▼                 ▼                                 ▼                 ▼
┌─────────┐     ┌──────────┐                    ┌─────────┐      ┌────────┐
│TRANSFORMS│     │ KERNELS  │                    │BACKENDS │      │ STEPS  │
├─────────┤     ├──────────┤                    ├─────────┤      ├────────┤
│@transform│     │ @kernel  │                    │@backend │      │ @step  │
│decorator │     │decorator │                    │decorator│      │decorator│
└────┬────┘     └────┬─────┘                    └────┬────┘      └────┬───┘
     │               │                                 │                │
     ▼               ▼                                 ▼                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         PLUGIN COLLECTIONS                               │
├─────────────┬──────────────┬─────────────────┬────────────────────────┤
│  Cleanup    │ Optimization │   HW Kernels    │   Code Generation      │
│ • RemoveId  │ • Streamline │ • LayerNorm     │ • FINN HLS            │
│ • FoldConst │ • SetTiled   │ • Softmax       │ • FINN RTL            │
│ • Tidy      │ • SetPumped  │ • MatMul        │ • Future Backend      │
└─────────────┴──────────────┴─────────────────┴────────────────────────┘
                                      │
                              ┌───────▼────────┐
                              │ Direct Access  │
                              │   • tfm.X()    │
                              │   • kn['Y']    │
                              │   • bk.find()  │
                              └────────────────┘
```

## Plugin Types and Registration

### Transform Plugins
```python
@transform(name="CustomOpt", stage="optimization", framework="qonnx")
class CustomOptimization:
    def apply(self, model):
        # Transform implementation
        return model
```

**Categories**:
- **Cleanup**: RemoveIdentity, FoldConstants, RemoveUnusedTensors
- **Optimization**: Streamline, SetTiled, SetPumped
- **Dataflow**: InferDataLayouts, AnnotateCycles
- **Hardware**: ConvertToHW, SpecializeLayers

### Kernel Plugins
```python
@kernel(name="MatMul", backends=["hls", "rtl"])
class MatMul:
    def get_nodeattr_types(self):
        return {"folding": ("i", True, 1)}
```

**Implemented Kernels**:
- **LayerNorm**: Layer normalization for transformers
- **Softmax**: Attention mechanism support
- **MatMul**: Matrix multiplication variants
- **Shuffle**: Channel shuffle operations
- **Crop**: Tensor cropping

### Backend Plugins
```python
@backend(kernel="MatMul", language="hls")
class MatMulHLSBackend:
    def generate(self, node):
        # HLS code generation
        pass
```

**Backend Types**:
- **FINN HLS**: High-Level Synthesis C++
- **FINN RTL**: Direct Verilog/VHDL generation
- **Future Brainsmith**: Plugin-based extensible backends
- **Mock**: Testing and simulation

### Step Plugins
```python
@step(name="PrepareIP", category="synthesis")
class PrepareIPStep:
    def execute(self, model, config):
        # Build step implementation
        return model
```

**Step Categories**:
- **Preprocessing**: Model preparation
- **Synthesis**: Hardware generation
- **Verification**: Correctness checking
- **Packaging**: IP creation

## Plugin Discovery Patterns

### Direct Access (Recommended)
```python
from brainsmith.plugins import transforms as tfm, kernels as kn

# Direct attribute access - O(1) performance
model = tfm.RemoveIdentity().apply(model)
model = tfm.qonnx.FoldConstants().apply(model)  # Framework-qualified
```

### Dictionary Access
```python
# For dynamic plugin names
transform_class = tfm['BatchNormToAffine']
model = transform_class().apply(model)
```

### Query Access
```python
# Find plugins by attributes
cleanup_transforms = tfm.find(stage="cleanup")
hls_kernels = kn.find(backend="hls")
backends = bk.find(kernel="LayerNorm", language="hls")
```

### Framework Collections
```python
# Access framework-specific plugins
finn_transforms = tfm.finn.all()
qonnx_transforms = tfm.qonnx.all()
brainsmith_kernels = kn.brainsmith.all()
```

## Key Design Principles

1. **Zero-Overhead**: Plugins register at decoration time, no runtime cost
2. **Direct Access**: No wrappers or proxies, direct class references
3. **Type Safety**: Full IDE support and type hints
4. **Framework Integration**: QONNX/FINN plugins registered seamlessly
5. **Pre-computed Indexes**: O(1) lookup for all access patterns
6. **Lazy Loading**: Plugins loaded only when accessed
7. **Production Optimized**: Blueprint loader creates minimal registries