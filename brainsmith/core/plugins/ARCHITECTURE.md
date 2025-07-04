# Brainsmith Plugin System Architecture

## Overview

The Brainsmith plugin system is a high-performance registry-based architecture that manages transforms, kernels, backends, and steps for FPGA compilation. It achieves zero discovery overhead through decoration-time registration and provides O(1) plugin access through direct class returns.

## Core Design Principles

1. **Direct Registration** - Plugins register at decoration time, eliminating discovery
2. **Pre-computed Indexes** - All lookups are optimized through indexing at registration
3. **Direct Class Access** - Collections return actual classes, not wrapper objects
4. **Universal Framework Support** - All plugin types support framework qualification
5. **Perfect Code Simplicity** - Minimal abstraction layers, maximum clarity

## High-Level Architecture

```mermaid
graph TB
    subgraph "Plugin Definition"
        PD[Plugin Class Definition]
        DEC[Decorator Application]
    end
    
    subgraph "Registry Core"
        REG[BrainsmithPluginRegistry]
        MD[Metadata Storage]
        IDX[Pre-computed Indexes]
    end
    
    subgraph "Access Layer"
        COL[Plugin Collections]
        FW[Framework Accessors]
        CAT[Category/Stage Accessors]
    end
    
    subgraph "Optimization"
        BP[Blueprint Loader]
        SUB[Subset Registry]
    end
    
    PD -->|"@decorator"| DEC
    DEC -->|"Auto-register"| REG
    REG --> MD
    REG --> IDX
    REG --> COL
    COL --> FW
    COL --> CAT
    BP -->|Creates| SUB
    SUB -->|Optimized| COL
```

## Component Architecture

### 1. Registry (`registry.py`)

The central component that stores all plugins and maintains indexes for efficient queries.

#### Registry Data Structure

```mermaid
classDiagram
    class BrainsmithPluginRegistry {
        +"Dict~str,Type~ transforms"
        +"Dict~str,Type~ kernels"
        +"Dict~str,Type~ backends"
        +"Dict~str,Type~ steps"
        +"Dict~str,Dict~ transforms_by_stage"
        +"Dict~str,List~ backends_by_kernel"
        +"Dict~str,Dict~ steps_by_category"
        +"Dict~str,Dict~ framework_transforms"
        +"Dict~str,Dict~ framework_kernels"
        +"Dict~str,Dict~ framework_backends"
        +"Dict~str,Dict~ framework_steps"
        +"Dict~str,Dict~ plugin_metadata"
        +"Dict~str,str~ default_backends"
        +"Dict~str,Dict~ backend_indexes"
        +"register_transform()"
        +"register_kernel()"
        +"register_backend()"
        +"register_step()"
        +"find_backends()"
        +"get_stats()"
    }
```

#### Key Design Decisions

- **Steps are first-class plugins** - Separate registry, not transforms with metadata
- **Universal framework support** - All plugin types have framework indexes
- **Backend names are unique** - Not composite keys like "Kernel_hls"
- **Direct class storage** - No wrapper objects in registry
- **Multiple indexes** - Enable O(1) queries by different criteria

### 2. Decorators (`decorators.py`)

Provide auto-registration at decoration time with validation.

#### Registration Flow

```mermaid
flowchart TD
    A[Plugin Class Definition] -->|"@decorator"| B[Validate Metadata]
    B --> C[Auto-register with Registry]
    C --> D[Update Main Dictionary]
    C --> E[Update Stage/Category Index]
    C --> F[Update Framework Index]
    C --> G[Update Query Indexes]
    D --> H[Plugin Available]
    E --> H
    F --> H
    G --> H
```

#### Decorator Types

```mermaid
graph LR
    subgraph "Decorator Types"
        T["@transform"]
        K["@kernel"]
        B["@backend"]
        S["@step"]
        KI["@kernel_inference"]
        P["@plugin"]
    end
    
    T -->|"stage-based"| REG[Registry]
    K -->|"hardware ops"| REG
    B -->|implementations| REG
    S -->|"build steps"| REG
    KI -->|converters| REG
    P -->|generic| REG
```

All decorators support `framework` parameter for framework qualification.

### 3. Collections (`plugin_collections.py`)

Provide natural access patterns through direct registry delegation.

#### Collection Access Flow

```mermaid
sequenceDiagram
    participant User
    participant Collection
    participant Registry
    participant Plugin
    
    User->>Collection: "transforms.MyTransform"
    Collection->>Collection: "__getattr__('MyTransform')"
    Collection->>Registry: "Direct lookup"
    Registry->>Collection: "Return class"
    Collection->>User: "Return actual class"
    User->>Plugin: "Instantiate directly"
```

#### Access Patterns

```mermaid
graph TD
    subgraph "Access Patterns"
        A["Direct: collection.PluginName"]
        B["Dict: collection['PluginName']"]
        C["Framework: collection.framework.PluginName"]
        D["Qualified: collection['framework:PluginName']"]
        E["Category: steps.category.StepName"]
        F["Stage: transforms.get_by_stage('cleanup')"]
    end
    
    A --> REG[Registry Lookup]
    B --> REG
    C --> FW[Framework Index]
    D --> FW
    E --> CAT[Category Index]
    F --> STG[Stage Index]
    
    FW --> REG
    CAT --> REG
    STG --> REG
```

### 4. Framework Adapters (`framework_adapters.py`)

Integrate external QONNX and FINN plugins.

#### Framework Integration Flow

```mermaid
flowchart LR
    subgraph "External Frameworks"
        Q[QONNX Transforms]
        F[FINN Transforms]
        FK[FINN Kernels]
    end
    
    subgraph "Registration"
        R[Direct Registration]
        FI[Framework Index]
    end
    
    subgraph "Access"
        QA["transforms.qonnx.*"]
        FA["transforms.finn.*"]
        KA["kernels.finn.*"]
    end
    
    Q -->|"Import & Register"| R
    F -->|"Import & Register"| R
    FK -->|"Import & Register"| R
    
    R --> FI
    FI --> QA
    FI --> FA
    FI --> KA
```

No wrapper classes needed - external classes are registered directly.

### 5. Blueprint Loader (`blueprint_loader.py`)

Optimizes plugin loading for production by creating subset registries.

#### Blueprint Processing Pipeline

```mermaid
flowchart TD
    subgraph "Input"
        Y[YAML Blueprint]
    end
    
    subgraph "Processing"
        P[Parse Requirements]
        E[Extract Plugin Names]
        S[Create Subset Registry]
    end
    
    subgraph "Output"
        O[Optimized Collections]
        M[Minimal Memory Footprint]
    end
    
    Y --> P
    P --> E
    E --> S
    S --> O
    S --> M
    
    E -->|"transforms: 15→5"| S
    E -->|"kernels: 10→3"| S
    E -->|"backends: 20→4"| S
```

## Data Flow

### Plugin Registration

```mermaid
flowchart LR
    subgraph "Registration Path"
        DEF[Plugin Definition]
        DEC[Decorator]
        VAL[Validation]
        REG[Registry]
        MAIN[Main Dict]
        IDX[Indexes]
    end
    
    DEF --> DEC
    DEC --> VAL
    VAL --> REG
    REG --> MAIN
    REG --> IDX
    
    IDX --> STG[Stage Index]
    IDX --> FRM[Framework Index]
    IDX --> KRN[Kernel Index]
```

### Plugin Access

```mermaid
flowchart LR
    subgraph "Access Path"
        USR[User Request]
        COL[Collection]
        REG[Registry Lookup]
        CLS[Direct Class]
    end
    
    USR -->|"tfm.Foo"| COL
    COL -->|"__getattr__"| REG
    REG -->|"O(1) dict"| CLS
    CLS -->|"actual class"| USR
```

### Query Operations

```mermaid
graph TD
    subgraph "Query Examples"
        Q1["find_backends(kernel='LayerNorm', language='hls')"]
        Q2["get_framework_kernels('finn')"]
        Q3["transforms.get_by_stage('cleanup')"]
    end
    
    subgraph "Index Usage"
        I1["backends_by_kernel['LayerNorm']"]
        I2["backend_indexes['language']['hls']"]
        I3["framework_kernels['finn']"]
        I4["transforms_by_stage['cleanup']"]
    end
    
    subgraph "Results"
        R1["Intersection of indexes"]
        R2["Direct O(1) lookup"]
        R3["Direct O(1) lookup"]
    end
    
    Q1 --> I1
    Q1 --> I2
    I1 --> R1
    I2 --> R1
    
    Q2 --> I3
    I3 --> R2
    
    Q3 --> I4
    I4 --> R3
```

## Performance Characteristics

### Time Complexity

```mermaid
graph LR
    subgraph "Operation Complexity"
        A["Plugin Lookup: O(1)"]
        B["Backend Query: O(1)"]
        C["Framework Lookup: O(1)"]
        D["Stage/Category: O(1)"]
        E["Registration: O(1)"]
    end
    
    style A fill:#90EE90
    style B fill:#90EE90
    style C fill:#90EE90
    style D fill:#90EE90
    style E fill:#90EE90
```

### Space Complexity

- **Main storage**: O(n) where n = number of plugins
- **Framework indexes**: O(f*n) where f = frameworks, n = plugins per framework
- **Category/stage indexes**: O(c*n) where c = categories/stages
- **Metadata**: O(n*p) where p = avg properties per plugin

### Startup Performance

- **Import time**: < 1ms (no discovery needed)
- **Registration**: O(1) per plugin
- **Index updates**: O(1) amortized
- **First access**: Sub-millisecond

## Integration Points

### System Integration

```mermaid
graph TB
    subgraph "External Systems"
        QONNX[QONNX Framework]
        FINN[FINN Framework]
        YAML[Blueprint YAML]
    end
    
    subgraph "Plugin System"
        REG[Registry]
        COL[Collections]
        BP[Blueprint Loader]
    end
    
    subgraph "Brainsmith Core"
        P1["Phase 1: Parser"]
        P2["Phase 2: DSE"]
        P3["Phase 3: Backend"]
    end
    
    QONNX -->|"Direct Registration"| REG
    FINN -->|"Direct Registration"| REG
    YAML -->|Requirements| BP
    
    REG --> COL
    BP --> COL
    
    COL --> P1
    COL --> P2
    COL --> P3
```

## Design Rationale

### Why Direct Class Access?

1. **Performance** - No wrapper function call overhead
2. **Simplicity** - What you see is what you get
3. **Debugging** - Clearer stack traces, no wrapper confusion
4. **Type Safety** - IDEs can provide proper type hints

### Why Universal Framework Support?

1. **Consistency** - All plugin types work the same way
2. **Flexibility** - Any plugin can come from any framework
3. **Future-Proof** - New frameworks easily integrated
4. **Natural API** - `kernels.finn.SomeKernel` is intuitive

### Why Steps as Separate Plugin Type?

1. **Clarity** - Steps are not transforms, they're build operations
2. **Organization** - Category-based organization vs stage-based
3. **Metadata** - Different metadata requirements than transforms
4. **API Consistency** - All plugin types should be first-class

### Why Multiple Access Patterns?

1. **Flexibility** - Different use cases prefer different patterns
2. **Blueprint Compatibility** - String lookup needed for YAML
3. **Framework Qualification** - Namespace collision resolution
4. **Developer Experience** - Natural for different contexts

## Future Considerations

### Extensibility

- New plugin types can be added by extending the base patterns
- New indexes can be added for new query patterns
- Framework adapters can integrate any external system

### Scalability

- Registry scales linearly with plugin count
- Framework indexes scale with frameworks * plugins
- Blueprint optimization reduces production footprint

### Compatibility

- Direct class access maintains type compatibility
- Framework qualification preserves namespace separation
- Multiple access patterns provide migration paths