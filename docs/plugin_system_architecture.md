# BrainSmith Plugin System Architecture

## Overview

The BrainSmith plugin system is a high-performance, extensible architecture designed for FPGA AI accelerator compilation. It provides a unified registry for transforms, kernels, backends, and steps while offering significant performance optimizations for production deployments.

## Design Principles

1. **Zero-Friction Development**: Automatic discovery and registration via decorators
2. **Production Efficiency**: Blueprint-driven loading for optimal performance
3. **Framework Agnostic**: Clean adapters for QONNX/FINN integration
4. **Type Safety**: Explicit decorators with validation
5. **Memory Efficient**: Weak references and lazy loading

## Architecture Components

### 1. Three-Pronged Discovery

The system uses three complementary discovery mechanisms:

```
┌─────────────────────────────────────────────────────────┐
│                   Plugin Discovery                       │
├─────────────────────┬──────────────────┬───────────────┤
│  Module Scanning    │  Stevedore      │  Framework     │
│  (Internal)         │  (External)     │  Adapters      │
├─────────────────────┼──────────────────┼───────────────┤
│ • Always enabled    │ • Always enabled │ • Conditional  │
│ • Zero-friction     │ • pip install   │ • QONNX/FINN   │
│ • @decorators       │ • Entry points  │ • Graceful     │
│ • Auto-register     │ • Priority      │   degradation  │
└─────────────────────┴──────────────────┴───────────────┘
```

### 2. Plugin Registry

Unified registry with type-specific storage:

```python
registry = {
    "transform:ExpandNorms": {
        "type": "transform",
        "name": "ExpandNorms",
        "class": ExpandNorms,
        "stage": "topology_opt",
        ...
    },
    "kernel:LayerNorm": {
        "type": "kernel",
        "name": "LayerNorm",
        "class": LayerNorm,
        ...
    },
    "backend:LayerNormHLS": {
        "type": "backend",
        "name": "LayerNormHLS",
        "kernel": "LayerNorm",
        "backend_type": "hls",
        ...
    }
}
```

### 3. Discovery Modes

Three modes optimize for different use cases:

| Mode | Use Case | Discovery | Performance |
|------|----------|-----------|-------------|
| **full** | Development/Testing | All plugins | 25ms startup |
| **blueprint** | Production | Only required | 5ms startup |
| **selective** | Advanced | Specific types | Variable |

### 4. Caching System

Multi-level caching for performance:

```
┌─────────────────────────────────────────┐
│          Discovery Cache (TTL)          │
│  • 5-minute default TTL                 │
│  • Keyed by (modes, frameworks, types) │
│  • Invalidated on file changes          │
└─────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────┐
│       Instance Cache (Weak Refs)        │
│  • Prevents duplicate instances         │
│  • Garbage collected when unused        │
│  • Thread-safe access                   │
└─────────────────────────────────────────┘
```

### 5. Blueprint Manager

Parses YAML blueprints for selective loading:

```yaml
hw_compiler:
  kernels:
    - "matmul"
    - {"kernel": "softmax", "backends": ["hls"]}
  transforms:
    - "quantization"
    - "~folding"  # Optional
  transforms_phased:
    pre_hw: ["cleanup_transforms"]
    post_hw: ["optimization_transforms"]
```

### 6. Framework Adapters

Clean adapter pattern for framework isolation:

```
┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
│  QONNXAdapter    │     │  FINNAdapter     │     │ BrainSmithAdapter│
├──────────────────┤     ├──────────────────┤     ├──────────────────┤
│ is_available()   │     │ is_available()   │     │ is_available()   │
│ discover_plugins()│     │ discover_plugins()│     │ discover_plugins()│
│ get_metadata()   │     │ get_metadata()   │     │ get_metadata()   │
└──────────────────┘     └──────────────────┘     └──────────────────┘
         ↓                        ↓                         ↓
    QONNX Registry          FINN Registry          Internal Modules
```

## Performance Characteristics

### Startup Time Comparison

```
Full Discovery (255+ plugins):     ████████████████████████ 25ms
Blueprint Mode (10-20 plugins):    █████ 5ms
Cache Hit (any mode):              ▌ <1ms
```

### Memory Usage

```
All Plugins Loaded:    ████████████████████████████████ ~500MB
Blueprint Subset:      ███ ~50MB
With Weak References:  ██ ~30MB (after GC)
```

### Discovery Performance

| Operation | Time | Description |
|-----------|------|-------------|
| Module Scan | 10ms | Scan internal modules |
| Stevedore | 5ms | Check entry points |
| QONNX Adapter | 8ms | Load QONNX transforms |
| FINN Adapter | 7ms | Load FINN transforms |
| Cache Save | 2ms | Serialize to cache |
| Cache Load | <1ms | Deserialize from cache |

## Usage Examples

### Transform Usage with QONNX Models

```python
from brainsmith.plugins import transforms as tfm

# Load QONNX model
from qonnx.core.modelwrapper import ModelWrapper
model = ModelWrapper("model.onnx")

# Apply transforms using model.transform()
model = model.transform(tfm.ExpandNorms())
model = model.transform(tfm.ConvertDivToMul())
model = model.transform(tfm.qonnx.FoldConstants())
model = model.transform(tfm.finn.Streamline())
```

### FINN Integration

```python
from brainsmith.plugins import transforms as tfm, steps

# Create FINN BUILD_STEPS
BUILD_STEPS = [
    steps.cleanup,
    lambda m, cfg: m.transform(tfm.ExpandNorms()),
    steps.qonnx_to_finn,
    lambda m, cfg: m.transform(tfm.finn.Streamline()),
    steps.hardware_inference,
]

# Or create custom step functions
def optimize_topology(model, cfg):
    model = model.transform(tfm.ExpandNorms())
    model = model.transform(tfm.ConvertDivToMul())
    model = model.transform(tfm.MergeOpChains())
    return model
```

## Implementation Flow

### 1. Plugin Registration

```python
from brainsmith.plugin.core import transform

@transform(
    name="ExpandNorms",
    stage="topology_opt",
    description="Expand normalization operations"
)
class ExpandNorms(Transformation):
    def apply(self, model):
        return model, graph_modified
```

### 2. Discovery Process

```python
# Development - Full discovery
from brainsmith.plugin import get_plugin_manager

manager = get_plugin_manager()
manager.discover_plugins(modes=['full'])

# Production - Blueprint-driven
from brainsmith.plugin import load_blueprint_plugins

plugins = load_blueprint_plugins('model.yaml')
```

### 3. Plugin Access Patterns

```python
# Recommended: Concise imports with QONNX pattern
from brainsmith.plugins import transforms as tfm

model = model.transform(tfm.ExpandNorms())
model = model.transform(tfm.finn.Streamline())

# Get class for FINN BUILD_STEPS
transform_cls = tfm.ExpandNorms  # Note: no parentheses

# Direct registry access (advanced)
from brainsmith.plugin.core import get_registry

registry = get_registry()
transform_cls = registry.get("transform", "ExpandNorms")
```

## Memory Management

### Weak Reference Strategy

```python
class PluginManager:
    def __init__(self):
        self._instance_cache = weakref.WeakValueDictionary()
    
    def get_instance(self, plugin_type, name):
        cache_key = f"{plugin_type}:{name}"
        
        # Check weak reference cache
        if cache_key in self._instance_cache:
            return self._instance_cache[cache_key]
        
        # Create new instance
        instance = self._create_instance(plugin_type, name)
        self._instance_cache[cache_key] = instance
        return instance
```

### Garbage Collection

- Plugin instances are garbage collected when no longer referenced
- Discovery cache persists based on TTL
- Explicit cache clearing available via `manager.clear_cache()`

## Configuration

### Environment Variables

```bash
# Discovery cache TTL (seconds)
export BRAINSMITH_PLUGIN_CACHE_TTL=300

# Plugin discovery mode
export BRAINSMITH_PLUGIN_MODE=blueprint

# Enable debug logging
export BRAINSMITH_PLUGIN_DEBUG=1
```

### Configuration File

```yaml
# .brainsmith/config.yaml
plugin_system:
  cache_ttl: 300
  discovery_mode: blueprint
  frameworks:
    - brainsmith
    - qonnx
  performance_tracking: true
```

## Best Practices

### 1. Development Workflow

```python
# Use concise imports for clean code
from brainsmith.plugins import transforms as tfm, kernels as kn, backends as bk

# List available plugins
print(tfm.list_plugins())

# Access plugins with QONNX pattern
model = model.transform(tfm.MyTransform())
model = model.transform(tfm.qonnx.RemoveIdentityOps())
```

### 2. Production Deployment

```python
# Use blueprint-driven loading
from brainsmith.plugin import load_blueprint_plugins

plugins = load_blueprint_plugins('production_model.yaml')

# Access only loaded plugins with concise names
tfm = plugins['transforms']
kn = plugins['kernels']
bk = plugins['backends']

# Use with QONNX models
model = model.transform(tfm.ExpandNorms())
```

### 3. Testing

```python
# Reset plugin system between tests
from brainsmith.plugins import reset_plugin_system

def test_my_plugin():
    reset_plugin_system()
    # Test code here
```

## Monitoring and Debugging

### Performance Metrics

```python
manager = get_plugin_manager()
stats = manager.get_performance_stats()

print(f"Discovery time: {stats['discovery_time_ms']}ms")
print(f"Cache hit rate: {stats['cache_hit_rate']:.2%}")
print(f"Memory usage: {stats['memory_usage_mb']}MB")
print(f"Plugin count: {stats['plugin_count']}")
```

### Debug Mode

```python
import logging
logging.getLogger('brainsmith.plugin').setLevel(logging.DEBUG)

manager = get_plugin_manager()
manager.set_debug_mode(True)
```

### Health Checks

```python
# Validate plugin system health
health = manager.health_check()
assert health['status'] == 'healthy'
assert health['cache_available']
assert health['frameworks_available'] >= 1
```

## Future Enhancements

1. **Distributed Caching**: Redis-based cache for multi-node deployments
2. **Plugin Versioning**: Semantic versioning with compatibility checks
3. **Hot Reloading**: Dynamic plugin updates without restart
4. **Metrics Export**: Prometheus/OpenTelemetry integration
5. **Plugin Marketplace**: Central repository for community plugins

## Conclusion

The BrainSmith plugin system architecture balances developer productivity with production performance through intelligent discovery, caching, and memory management. The three-pronged discovery approach ensures plugins are available when needed while minimizing overhead, making it ideal for both development and production FPGA compilation workflows.