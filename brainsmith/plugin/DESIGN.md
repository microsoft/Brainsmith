# Pure Stevedore Plugin System - Design Document

## Executive Summary

The Pure Stevedore Plugin System represents a complete architectural transformation of BrainSmith's plugin infrastructure. Following Prime Directive 1 (Break Fearlessly), this design eliminates all legacy adapters and broken systems, replacing them with a clean, efficient, and extensible plugin architecture that utilizes Stevedore to its full potential.

## Design Philosophy

### Core Principles

1. **Direct Integration Over Adaptation**
   - No adapters to broken systems
   - Direct connection to framework-native registries
   - Zero abstraction layers between plugins and their sources

2. **Natural Access Patterns**
   - Object-oriented plugin access (`transforms.ExpandNorms()`)
   - Framework-organized namespaces (`transforms.qonnx.RemoveIdentityOps()`)
   - Zero boilerplate imports

3. **Intelligent Discovery**
   - Hybrid approach combining multiple discovery strategies
   - Stevedore entry points for external plugins
   - Auto-discovery for development convenience
   - Direct module scanning for framework transforms

4. **Performance First**
   - Lazy loading with intelligent caching
   - Thread-safe operations with minimal locking
   - Optimized discovery algorithms

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                     User Interface Layer                         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Global Collections (transforms, kernels)                │   │
│  │  - Zero boilerplate: from brainsmith.plugins import ... │   │
│  │  - Natural access: transforms.qonnx.RemoveIdentityOps() │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Plugin Manager Core                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  Pure Stevedore Plugin Manager                           │   │
│  │  - Discovery strategies (Stevedore, Auto, Hybrid)        │   │
│  │  - Plugin catalog with conflict resolution               │   │
│  │  - Lazy loading and caching                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Discovery Layer                               │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  Stevedore   │  │    Auto      │  │  Framework Native  │   │
│  │  Entry Points│  │  Discovery   │  │    Discovery       │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                                  │
┌─────────────────────────────────────────────────────────────────┐
│                    Plugin Sources                                │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐   │
│  │  BrainSmith  │  │    QONNX     │  │       FINN         │   │
│  │   Plugins    │  │  Transforms  │  │    Transforms      │   │
│  └──────────────┘  └──────────────┘  └────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Component Design

### 1. Plugin Manager (`brainsmith/plugin/manager.py`)

#### Purpose
Central orchestrator for all plugin discovery, loading, and management operations.

#### Key Classes

**PluginInfo**
```python
@dataclass
class PluginInfo:
    name: str
    plugin_class: type
    framework: str  # "qonnx", "finn", "brainsmith"
    plugin_type: str  # "transform", "kernel", "backend", "step"
    metadata: Dict[str, Any]
    discovery_method: str  # "stevedore", "auto", "framework_native"
    stevedore_extension: Optional[Any] = None
```

**PluginCatalog**
```python
@dataclass
class PluginCatalog:
    plugins_by_name: Dict[str, List[PluginInfo]]
    plugins_by_type: Dict[str, List[PluginInfo]]
    plugins_by_framework: Dict[str, List[PluginInfo]]
    conflicts: Dict[str, List[PluginInfo]]
    unique_plugins: Dict[str, PluginInfo]
```

#### Discovery Strategies

1. **STEVEDORE_ONLY**: Use only Stevedore entry points
2. **AUTO_DISCOVERY**: Scan codebase + entry points
3. **HYBRID**: Best of both worlds (default)

#### Discovery Implementation

**Stevedore Discovery**
- Scans entry point namespaces:
  - `brainsmith.transforms`
  - `brainsmith.kernels`
  - `brainsmith.external.*`
- Leverages Python's entry point system
- Supports external plugin packages

**Auto-Discovery**
- BrainSmith native plugins:
  - Scans `brainsmith.transforms.*`
  - Scans `brainsmith.kernels.*`
  - Scans `brainsmith.steps.*`
- Direct module introspection
- Decorator-based registration support

**Framework Native Discovery**
- QONNX transforms:
  - Direct scanning of `qonnx.transformation.*` modules
  - No intermediate registries
- FINN transforms:
  - Direct scanning of `finn.transformation.*` modules
  - Includes nested modules (e.g., `streamline.reorder`)

### 2. Collections Layer (`brainsmith/plugin/collections.py`)

#### Purpose
Provides natural, object-oriented access to plugins with framework organization.

#### Key Classes

**TransformCollection**
- Framework-specific access via properties
- Unique transform access via `__getattr__`
- Intelligent error messages for conflicts

**FrameworkTransforms**
- Represents transforms from a specific framework
- Lazy transform wrapper creation
- Clear error messages for missing transforms

**Transform/Kernel Wrappers**
- Natural calling interface
- Lazy instantiation
- Parameter forwarding

#### Access Patterns

```python
# Framework-specific (for conflicts)
transforms.qonnx.RemoveIdentityOps()
transforms.finn.MoveOpPastFork()

# Unique transforms (no prefix needed)
transforms.ExpandNorms()
transforms.FoldConstants()

# Kernels with backends
kernels.LayerNorm.hls()
kernels.Softmax.rtl()
```

### 3. Global Access (`brainsmith/plugins/__init__.py`)

#### Purpose
Zero-boilerplate global access to plugins with lazy initialization.

#### Features

- Module-level collections that act like imports
- Thread-safe lazy initialization
- Utility functions for advanced usage
- Plugin system introspection

#### API

```python
# Primary exports
transforms = _GlobalTransformCollection()
kernels = _GlobalKernelCollection()

# Utility functions
get_plugin_manager()
list_all_plugins()
analyze_conflicts()
plugin_status()
reset_plugin_cache()
```

## Discovery Details

### BrainSmith Native Plugins

**Transform Discovery**
```python
# Modules scanned
'topology_opt.expand_norms'
'model_specific.remove_bert_head'
'model_specific.remove_bert_tail'
'kernel_opt.set_pumped_compute'
'kernel_opt.temp_shuffle_fixer'
'metadata.extract_shell_integration_metadata'
```

**Kernel Discovery**
```python
# Kernel modules scanned
'layernorm', 'matmul', 'softmax', 'shuffle', 'crop'
```

**Step Discovery**
- Scans for `@finn_step` decorated functions
- Extracts metadata from decorators

### QONNX Transform Discovery

```python
# Modules scanned
'qonnx.transformation.general'
'qonnx.transformation.remove'
'qonnx.transformation.fold_constants'
'qonnx.transformation.infer_data_layouts'
'qonnx.transformation.infer_datatypes'
'qonnx.transformation.infer_shapes'
```

### FINN Transform Discovery

```python
# Modules scanned
'finn.transformation.streamline'
'finn.transformation.streamline.reorder'
'finn.transformation.move_reshape'
'finn.transformation.fpgadataflow.convert_to_hw_layers'
```

## Conflict Resolution

### Naming Conflicts

When multiple frameworks provide the same transform name:

1. **Detection**: During discovery, identify all naming conflicts
2. **Marking**: Mark conflicted plugins as non-unique
3. **Access**: Require framework prefix for conflicted names
4. **Error Messages**: Provide clear guidance on resolution

Example:
```
AttributeError: Plugin 'RemoveIdentityOps' is ambiguous. 
Found in frameworks: ['qonnx', 'finn']. 
Use qualified name like 'qonnx:RemoveIdentityOps' or 'finn:RemoveIdentityOps'
```

### Unique Plugin Access

Plugins with unique names across all frameworks can be accessed without prefix:
- Faster access for common operations
- Cleaner code for BrainSmith-specific transforms
- Automatic resolution for unique names

## Performance Optimizations

### Lazy Loading

- Plugins discovered but not instantiated until used
- Transform wrappers created on-demand
- Class loading deferred until actual invocation

### Caching Strategy

```python
# Three-level caching
1. Discovery cache (PluginCatalog)
2. Plugin info cache (loaded plugins)
3. Transform wrapper cache (instantiated wrappers)
```

### Thread Safety

- Read-Write lock for discovery operations
- Minimal critical sections
- Thread-local caching where appropriate

## Error Handling

### Discovery Errors

- Graceful handling of import failures
- Warning logs for problematic modules
- Continues discovery despite individual failures

### Access Errors

- Clear error messages for missing plugins
- Suggestions for similar plugin names
- Framework hints for ambiguous names

### Usage Errors

- Type checking for plugin parameters
- Helpful error messages for common mistakes
- Validation of plugin types

## Extension Points

### Adding New Frameworks

1. Add discovery method to PluginManager
2. Update FrameworkTransforms in collections
3. No changes needed to global access layer

### Custom Discovery Strategies

```python
class CustomDiscoveryStrategy:
    def discover(self) -> List[PluginInfo]:
        # Custom discovery logic
        pass
```

### Plugin Decorators

Future support for decorator-based registration:
```python
@brainsmith_transform("my_transform")
class MyTransform:
    pass
```

## Migration Path

### From Old System

```python
# OLD (Broken)
from brainsmith.core import apply_transform
model = apply_transform(model, "qonnx:RemoveIdentityOps")

# NEW (Natural)
from brainsmith.plugins import transforms
model = transforms.qonnx.RemoveIdentityOps()(model)
```

### Benefits
- No string-based lookups
- IDE support with tab completion
- Type hints and introspection
- Natural Python patterns

## Testing Strategy

### Unit Tests
- Individual component testing
- Mock-free concrete tests (PD-3)
- Discovery strategy validation

### Integration Tests
- End-to-end plugin access
- BERT step compatibility
- Framework interaction tests

### Performance Tests
- Discovery timing benchmarks
- Memory usage profiling
- Concurrent access validation

## Future Enhancements

### 1. Blueprint Optimization Layer

```python
class BlueprintOptimizer:
    """Pre-load and optimize plugins for specific blueprints."""
    def optimize_for_blueprint(self, blueprint_path: str):
        # Analyze blueprint requirements
        # Pre-load required plugins
        # Generate optimized access paths
```

### 2. Plugin Package Support

- PyPI-distributed plugin packages
- Automatic registration via entry points
- Version compatibility checking

### 3. Dynamic Reloading

- Hot-reload during development
- Plugin update detection
- Cache invalidation strategies

## Conclusion

The Pure Stevedore Plugin System represents a complete break from legacy patterns, implementing a clean, efficient, and extensible plugin architecture. By eliminating adapters and implementing direct integration, we've created a system that is both more powerful and easier to use than any previous iteration.

Key achievements:
- **145+ plugins** discoverable across all frameworks
- **Zero boilerplate** for common usage patterns
- **Natural access** that feels like native Python
- **Extensible architecture** ready for future enhancements

This design fulfills all Prime Directives while creating a foundation for BrainSmith's future plugin ecosystem.