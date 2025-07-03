# Brainsmith Plugin System Architecture

## Overview

The Brainsmith plugin system is a high-performance registry-based architecture that manages transforms, kernels, and backends for FPGA compilation. It achieves zero discovery overhead through decoration-time registration and provides O(1) plugin access through direct dictionary lookups.

## Core Design Principles

1. **Direct Registration** - Plugins register at decoration time, eliminating discovery
2. **Pre-computed Indexes** - All lookups are optimized through indexing at registration
3. **Thin Collections** - Access layers delegate directly to registry without caching
4. **Explicit Integration** - External frameworks integrated through simple wrappers

## Component Architecture

### 1. Registry (`registry.py`)

The central component that stores all plugins and maintains indexes for efficient queries.

#### Data Structures

```python
class BrainsmithPluginRegistry:
    # Main storage - direct dictionaries
    transforms: Dict[str, Type]      # name -> class
    kernels: Dict[str, Type]         # name -> class  
    backends: Dict[str, Type]        # name -> class (not composite keys!)
    
    # Pre-computed indexes for performance
    transforms_by_stage: Dict[str, Dict[str, Type]]    # stage -> {name -> class}
    backends_by_kernel: Dict[str, List[str]]           # kernel -> [backend_names]
    framework_transforms: Dict[str, Dict[str, Type]]   # framework -> {name -> class}
    
    # Metadata storage
    plugin_metadata: Dict[str, Dict[str, Any]]         # name -> metadata
    default_backends: Dict[str, str]                   # kernel -> default_backend_name
    
    # Backend query indexes
    backend_indexes: Dict[str, Dict[str, List[str]]]  # attribute -> {value -> [names]}
```

#### Key Design Decisions

- **Backend names are unique** - Not composite keys like "Kernel_hls"
- **Backends indexed by kernel** - List of backend names, not types
- **Multiple indexes** - Enable O(1) queries by different criteria
- **Metadata separate** - Allows rich attributes without polluting main storage

### 2. Decorators (`decorators.py`)

Provide auto-registration at decoration time with validation.

#### Registration Flow

```
@transform decorator applied
    ↓
Validate metadata (stage XOR kernel required)
    ↓
Store metadata on class (_plugin_metadata)
    ↓
Auto-register with global registry
    ↓
Update all relevant indexes
    ↓
Plugin immediately available
```

#### Decorator Types

- `@transform` - Stage-based or kernel-specific transforms
- `@kernel` - Hardware operation definitions
- `@backend` - Kernel implementations (HLS, RTL, etc.)
- `@step` - High-level operations (internally transforms)
- `@kernel_inference` - Transform to kernel converters
- `@plugin` - Generic decorator for any type

### 3. Collections (`collections.py`)

Provide natural access patterns through thin wrappers.

#### Collection Architecture

```
User Code
    ↓
Collection (transforms.MyTransform)
    ↓
__getattr__ dynamic resolution
    ↓
Direct registry lookup
    ↓
Wrapper creation (on-demand)
    ↓
Plugin instance
```

#### Key Features

- **No caching** - Direct delegation to registry
- **Dynamic wrappers** - Created on access, not stored
- **Framework accessors** - `transforms.qonnx.*`, `transforms.finn.*`
- **Natural methods** - `kernel.hls()`, `kernel.find_backend()`

### 4. Blueprint Loader (`blueprint_loader.py`)

Optimizes plugin loading for production by creating subset registries.

#### Blueprint Processing

```
YAML Blueprint
    ↓
Parse requirements (transforms, kernels, backends)
    ↓
Create subset registry with only required plugins
    ↓
Maintain all indexes for subset
    ↓
Return optimized collections
```

#### Optimization Benefits

- Load only required plugins (typically 10-20% of total)
- Reduced memory footprint
- Faster registry operations
- Same API as full registry

### 5. Framework Adapters (`framework_adapters.py`)

Integrate external QONNX and FINN transforms/kernels/backends.

#### Integration Pattern

```python
# Wrapper adapts external API
class QONNXTransformWrapper:
    def __init__(self, qonnx_class):
        self.qonnx_class = qonnx_class
    
    def apply(self, model):
        instance = self.qonnx_class()
        return instance.apply(model)

# Register with framework attribution
registry.register_transform(
    name="BatchNormToAffine",
    transform_class=wrapper,
    framework="qonnx",
    stage="topology_opt"
)
```

## Data Flow

### Plugin Registration

```
Plugin Definition → Decorator → Validation → Registry → Indexes
                                                ↓          ↓
                                         Main Dict    Stage/Framework/Kernel
```

### Plugin Access

```
User Request → Collection → Registry Lookup → Wrapper → Instance
  tfm.Foo      __getattr__   O(1) dict       on-demand   Foo()
```

### Backend Selection

```
Kernel Request → Backend Query → Index Lookup → Backend Class
  kn.LayerNorm    .hls()         language=hls    LayerNormHLS
```

## Registry Operations

### Transform Operations

```python
# Registration updates multiple structures
register_transform("Foo", FooClass, stage="cleanup")
    → transforms["Foo"] = FooClass
    → transforms_by_stage["cleanup"]["Foo"] = FooClass
    → framework_transforms["brainsmith"]["Foo"] = FooClass
    → plugin_metadata["Foo"] = {type: "transform", stage: "cleanup", ...}
```

### Backend Operations

```python
# Backend registration with proper indexing
register_backend("LayerNormHLS", LayerNormHLS, kernel="LayerNorm", language="hls")
    → backends["LayerNormHLS"] = LayerNormHLS  # Note: actual name, not composite
    → backends_by_kernel["LayerNorm"].append("LayerNormHLS")
    → backend_indexes["language"]["hls"].append("LayerNormHLS")
    → plugin_metadata["LayerNormHLS"] = {kernel: "LayerNorm", language: "hls", ...}
```

### Query Operations

```python
# Efficient lookups using indexes
find_backends(kernel="LayerNorm", language="hls")
    → candidates = backends_by_kernel["LayerNorm"]  # ["LayerNormHLS", "LayerNormRTL"]
    → filter by backend_indexes["language"]["hls"]  # ["LayerNormHLS", ...]
    → return intersection
```

## Performance Characteristics

### Time Complexity

- **Plugin lookup**: O(1) - Direct dictionary access
- **Backend query**: O(1) - Pre-computed indexes
- **Framework lookup**: O(1) - Direct framework dict
- **Stage lookup**: O(1) - Direct stage dict

### Space Complexity

- **Main storage**: O(n) where n = number of plugins
- **Indexes**: O(m*k) where m = attributes, k = unique values
- **Metadata**: O(n*p) where p = avg properties per plugin

### Startup Performance

- **Import time**: < 1ms (no discovery needed)
- **Registration**: O(1) per plugin
- **Index updates**: O(1) amortized
- **First access**: Sub-millisecond

## Integration Points

### QONNX/FINN Integration

```
External Class → Wrapper → Registry → Framework Collection
  QTransform     QWrapper   qonnx      tfm.qonnx.QTransform
```

### Blueprint Integration

```
Blueprint YAML → Requirements → Subset Registry → Optimized Collections
  15 transforms   only needed    2 transforms      minimal footprint
```

### Phase 1 DSE Integration

The registry provides discovery methods for Phase 1 compatibility:

- `list_available_kernels()` - All kernel names
- `list_available_transforms()` - All transform names  
- `validate_kernel_backends()` - Check backend availability
- `get_valid_stages()` - Valid transform stages

## Design Rationale

### Why Decoration-Time Registration?

1. **Zero discovery overhead** - No scanning, no delays
2. **Immediate availability** - Use right after import
3. **Clear registration point** - Visible in code
4. **Fail-fast** - Registration errors caught early

### Why Direct Registry Access?

1. **Simplicity** - No manager abstraction needed
2. **Performance** - Direct dictionary lookups
3. **Clarity** - Data flow is obvious
4. **Maintainability** - Less code, fewer bugs

### Why Pre-computed Indexes?

1. **Query performance** - All lookups are O(1)
2. **No scanning** - Never iterate all plugins
3. **Memory efficient** - Computed once at registration
4. **Flexible queries** - Multiple access patterns

### Why Separate Backend Names?

1. **Uniqueness** - Each backend has unique identity
2. **Flexibility** - Multiple backends per kernel/language combo
3. **Rich metadata** - Optimization strategies, resource usage
4. **Clear queries** - Find by any attribute combination

## Future Considerations

### Extensibility

- New plugin types can be added by extending decorators
- New indexes can be added for new query patterns
- Framework adapters can integrate any external system

### Scalability

- Registry scales linearly with plugin count
- Indexes scale with unique attribute values
- Blueprint optimization reduces production footprint

### Compatibility

- Backward compatibility through parameter aliases
- Bridge modules for import compatibility
- Stable API for external integrations