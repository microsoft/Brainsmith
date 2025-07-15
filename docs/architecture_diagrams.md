# Brainsmith Architecture Diagrams

## 1. High-Level Three-Phase Pipeline

```mermaid
flowchart TB
    accTitle: Brainsmith Three-Phase DSE Pipeline with Exploration Loop
    accDescr: Shows the three phases with the cyclical nature of exploration where Phase 2 repeatedly calls Phase 3 for each configuration

    subgraph Input
        BP[Blueprint YAML]
        Model[PyTorch Model]
    end

    subgraph "Phase 1: Constructor"
        Parser[Parser]
        Forge[ForgeAPI]
        Val[Validator]
        DS[Design Space]
    end

    subgraph "Phase 2: Explorer"
        Gen[Combination Generator]
        Exp[Explorer Engine]
        Hooks[Exploration Hooks]
        Prog[Progress Tracker]
        Queue[Config Queue]
        Agg[Results Aggregator]
    end

    subgraph "Phase 3: Runner"
        BR[Build Runner]
        Pre[Preprocessing]
        Backend[Backend]
        Post[Postprocessing]
        Metrics[Metrics Collector]
    end

    subgraph Output
        Results[Exploration Results]
        RTL[RTL Implementations]
    end

    BP --> Parser
    Model --> Parser
    Parser --> Forge
    Forge --> Val
    Val --> DS
    DS --> Gen
    Gen --> Queue
    Queue --> Exp
    
    Exp -->|"For each config"| BR
    BR --> Pre
    Pre --> Backend
    Backend --> Post
    Post --> Metrics
    Metrics -->|"Build Result"| Exp
    
    Exp -->|"Check next"| Queue
    Queue -->|"More configs"| Exp
    Queue -->|"All complete"| Agg
    
    Agg --> Results
    Backend --> RTL
    
    Hooks -.-> Exp
    Prog -.-> Exp

    classDef phase1 fill:#0891b2,stroke:#0e7490,stroke-width:2px,color:#fff
    classDef phase2 fill:#7c3aed,stroke:#6b21a8,stroke-width:2px,color:#fff
    classDef phase3 fill:#059669,stroke:#047857,stroke-width:2px,color:#fff
    classDef cycle fill:#dc2626,stroke:#b91c1c,stroke-width:3px,color:#fff
    
    class Parser,Forge,Val,DS phase1
    class Gen,Exp,Hooks,Prog,Queue,Agg phase2
    class BR,Pre,Backend,Post,Metrics phase3
    class Exp cycle
```

## 2. Plugin System Architecture

```mermaid
flowchart TD
    accTitle: Brainsmith Plugin System Architecture
    accDescr: Shows the plugin registry system with its various indexes and registration mechanism

    subgraph "Plugin Registration"
        Dec[Plugin Decorators]
        Reg[Plugin Registry]
    end

    subgraph "Plugin Types"
        T[Transforms]
        K[Kernels]
        B[Backends]
        S[Steps]
    end

    subgraph "Registry Indexes"
        ByType[By Type Index]
        ByStage[By Stage Index]
        ByFW[By Framework Index]
        ByKernel[By Kernel Index]
        ByCat[By Category Index]
    end

    subgraph "Access Patterns"
        PC[Plugin Collections]
        BL[Blueprint Loader]
        FA[Framework Adapters]
    end

    Dec -->|"@plugin"| Reg
    T --> Reg
    K --> Reg
    B --> Reg
    S --> Reg

    Reg --> ByType
    Reg --> ByStage
    Reg --> ByFW
    Reg --> ByKernel
    Reg --> ByCat

    ByType --> PC
    ByStage --> PC
    ByFW --> PC
    PC --> BL
    PC --> FA

    classDef registry fill:#1f2937,stroke:#374151,stroke-width:3px,color:#fff
    classDef index fill:#7c3aed,stroke:#6b21a8,stroke-width:2px,color:#fff
    
    class Reg registry
    class ByType,ByStage,ByFW,ByKernel,ByCat index
```

## 3. Phase 1 Data Flow

```mermaid
flowchart TD
    accTitle: Phase 1 Design Space Construction Flow
    accDescr: Detailed flow of how blueprints are parsed and validated to create design spaces

    YAML[Blueprint YAML]
    
    subgraph Parser Operations
        PL[Plugin Loader]
        YP[YAML Parser]
        AD[Auto Discovery]
    end
    
    subgraph Data Structures
        HW[HWCompilerSpace]
        SC[SearchConfig]
        GC[GlobalConfig]
        DS[DesignSpace]
    end
    
    subgraph Validation
        MV[Model Validator]
        CV[Combination Validator]
        VDS[Validated DesignSpace]
    end

    YAML --> PL
    PL -->|Load Required Plugins| YP
    YP --> AD
    AD -->|Discover Kernels/Transforms| HW
    YP --> SC
    YP --> GC
    HW --> DS
    SC --> DS
    GC --> DS
    DS --> MV
    MV --> CV
    CV --> VDS

    classDef input fill:#dc2626,stroke:#b91c1c,stroke-width:2px,color:#fff
    classDef process fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef output fill:#059669,stroke:#047857,stroke-width:2px,color:#fff
    
    class YAML input
    class VDS output
```

## 4. Phase 2 Exploration Strategy

```mermaid
stateDiagram-v2
    accTitle: Phase 2 Exploration State Machine
    accDescr: Shows the state transitions during design space exploration

    [*] --> Initialize
    Initialize --> GenerateCombinations
    GenerateCombinations --> CheckCache
    
    CheckCache --> ExecuteBuild: Not Cached
    CheckCache --> UseCache: Cached
    ExecuteBuild --> CollectMetrics
    UseCache --> CollectMetrics
    CollectMetrics --> UpdateProgress
    UpdateProgress --> CheckNext
    
    CheckNext --> GenerateCombinations: More Configs
    CheckNext --> Aggregate: All Complete
    Aggregate --> OptimizeResults
    OptimizeResults --> [*]

    note right of ExecuteBuild: Calls Phase 3
    note right of UpdateProgress: ETA Calculation
    note right of Aggregate: Results Analysis
```

## 5. Phase 3 Build Pipeline

```mermaid
flowchart LR
    accTitle: Phase 3 Build Execution Pipeline
    accDescr: Shows the preprocessing, backend execution, and postprocessing flow

    BC[BuildConfig]
    
    subgraph Preprocessing
        PT[Transform Pipeline]
        PV[Validation]
        PP[Prepared Model]
    end
    
    subgraph Backend Selection
        BS{Backend?}
        FINN[FINN Legacy Backend]
        FUT[Future Brainsmith Backend]
    end
    
    subgraph Postprocessing
        MT[Metrics Collection]
        EC[Error Categorization]
        BR[BuildResult]
    end

    BC --> PT
    PT --> PV
    PV --> PP
    PP --> BS
    BS -->|legacy| FINN
    BS -->|future| FUT
    FINN --> MT
    FUT --> MT
    MT --> EC
    EC --> BR

    classDef preprocessing fill:#ea580c,stroke:#c2410c,stroke-width:2px,color:#fff
    classDef backend fill:#2563eb,stroke:#1d4ed8,stroke-width:2px,color:#fff
    classDef postprocessing fill:#059669,stroke:#047857,stroke-width:2px,color:#fff
    
    class PT,PV,PP preprocessing
    class BS,FINN,FUT backend
    class MT,EC,BR postprocessing
```

## 6. Transform Stage Organization

```mermaid
graph TB
    accTitle: Transform Stage Organization
    accDescr: Shows how transforms are organized by processing stages

    subgraph Transform Stages
        PRE[pre_proc]
        CLEAN[cleanup]
        TOPO[topology_opt]
        KERNEL[kernel_opt]
        DATA[dataflow_opt]
        POST[post_proc]
    end

    subgraph Example Transforms
        PRE --> T1[InferDataLayouts]
        PRE --> T2[InsertTopK]
        CLEAN --> T3[RemoveIdentityOps]
        CLEAN --> T4[RemoveUnusedTensors]
        TOPO --> T5[StreamlineTransform]
        KERNEL --> T6[MatMulToXnorPopcount]
        DATA --> T7[InferFIFODepths]
        POST --> T8[PrepareIP]
    end

    style PRE fill:#dc2626,color:#fff
    style CLEAN fill:#ea580c,color:#fff
    style TOPO fill:#2563eb,color:#fff
    style KERNEL fill:#7c3aed,color:#fff
    style DATA fill:#059669,color:#fff
    style POST fill:#ec4899,color:#fff
```

## 7. Configuration Hierarchy

```mermaid
flowchart BT
    accTitle: Configuration Priority Hierarchy
    accDescr: Shows the configuration resolution order from defaults to project-specific

    D[Default Values]
    E[Environment Variables]
    U[User Config ~/.brainsmith/config.yaml]
    P[Project/Blueprint Config]
    
    F[Final Configuration]
    
    D -->|Lowest Priority| F
    E -->|Override Defaults| F
    U -->|Override Env| F
    P -->|Highest Priority| F

    classDef default fill:#6b7280,stroke:#4b5563,color:#fff
    classDef env fill:#ea580c,stroke:#c2410c,color:#fff
    classDef user fill:#2563eb,stroke:#1d4ed8,color:#fff
    classDef project fill:#059669,stroke:#047857,color:#fff
    classDef final fill:#7c3aed,stroke:#6b21a8,stroke-width:3px,color:#fff
    
    class D default
    class E env
    class U user
    class P project
    class F final
```

## 8. Exploration Hooks System

```mermaid
sequenceDiagram
    accTitle: Exploration Hooks Execution Sequence
    accDescr: Shows when hooks are called during the exploration process

    participant E as Explorer
    participant H as Hook System
    participant L as Logger Hook
    participant C as Cache Hook
    participant U as User Hook

    E->>H: on_exploration_start()
    H->>L: Log start
    H->>C: Check cache
    H->>U: Custom start logic

    loop For each config
        E->>H: on_config_start(config)
        H->>L: Log config
        E->>H: on_config_complete(result)
        H->>C: Cache result
        H->>L: Log metrics
        H->>U: Custom logic
    end

    E->>H: on_exploration_complete(results)
    H->>L: Log summary
    H->>C: Save cache
    H->>U: Custom complete logic
```

## 9. Plugin Discovery Flow

```mermaid
flowchart TD
    accTitle: Plugin Discovery and Loading Flow
    accDescr: Shows how plugins are discovered and loaded based on blueprint requirements

    BP[Blueprint]
    
    subgraph Discovery
        Parse[Parse Blueprint]
        Scan[Scan Requirements]
        Find[Find Plugins]
    end
    
    subgraph Loading
        Check{Loaded?}
        Load[Import Module]
        Reg[Register Plugin]
        Cache[Cache Reference]
    end
    
    subgraph Access
        Get[Get Plugin]
        Use[Use Plugin]
    end

    BP --> Parse
    Parse --> Scan
    Scan --> Find
    Find --> Check
    Check -->|No| Load
    Check -->|Yes| Cache
    Load --> Reg
    Reg --> Cache
    Cache --> Get
    Get --> Use

    classDef discover fill:#ea580c,stroke:#c2410c,color:#fff
    classDef load fill:#2563eb,stroke:#1d4ed8,color:#fff
    classDef access fill:#059669,stroke:#047857,color:#fff
    
    class Parse,Scan,Find discover
    class Check,Load,Reg,Cache load
    class Get,Use access
```

## 10. Error Handling Architecture

```mermaid
flowchart TD
    accTitle: Error Handling and Categorization
    accDescr: Shows how errors are handled and categorized throughout the system

    E[Error Occurs]
    
    subgraph Phase Detection
        P1E[Phase 1 Error]
        P2E[Phase 2 Error]
        P3E[Phase 3 Error]
    end
    
    subgraph Categorization
        VE[Validation Error]
        BE[Build Error]
        PE[Plugin Error]
        CE[Configuration Error]
    end
    
    subgraph Handling
        Log[Log Error]
        Cat[Categorize]
        Wrap[Wrap with Context]
        Report[Report to User]
    end

    E --> P1E
    E --> P2E
    E --> P3E
    
    P1E --> VE
    P1E --> CE
    P2E --> CE
    P3E --> BE
    P3E --> PE
    
    VE --> Log
    BE --> Log
    PE --> Log
    CE --> Log
    
    Log --> Cat
    Cat --> Wrap
    Wrap --> Report

    classDef error fill:#dc2626,stroke:#b91c1c,color:#fff
    classDef category fill:#ea580c,stroke:#c2410c,color:#fff
    classDef handle fill:#2563eb,stroke:#1d4ed8,color:#fff
    
    class E,P1E,P2E,P3E error
    class VE,BE,PE,CE category
    class Log,Cat,Wrap,Report handle
```

## 11. Framework Integration

```mermaid
graph LR
    accTitle: Framework Integration Architecture
    accDescr: Shows how external frameworks are integrated into Brainsmith

    subgraph Brainsmith Core
        PA[Plugin Adapters]
        REG[Registry]
    end
    
    subgraph External Frameworks
        FINN[FINN]
        QONNX[QONNX]
        BREV[Brevitas]
    end
    
    subgraph Adapted Plugins
        FT["FINN Transforms<br/>100+ plugins"]
        QT["QONNX Transforms<br/>50+ plugins"]
        BT[Brevitas Tools]
    end

    FINN --> PA
    QONNX --> PA
    BREV --> PA
    
    PA --> FT
    PA --> QT
    PA --> BT
    
    FT --> REG
    QT --> REG
    BT --> REG

    classDef framework fill:#ea580c,stroke:#c2410c,color:#fff
    classDef adapter fill:#2563eb,stroke:#1d4ed8,color:#fff
    classDef plugin fill:#059669,stroke:#047857,color:#fff
    
    class FINN,QONNX,BREV framework
    class PA adapter
    class FT,QT,BT plugin
```

## 12. Build Metrics Collection

```mermaid
classDiagram
    accTitle: Build Metrics Class Structure
    accDescr: Shows the structure of metrics collected during builds

    class BuildMetrics {
        +id: str
        +status: str
        +duration: float
        +timestamp: datetime
        +resource_estimates: ResourceEstimates
        +performance_estimates: PerformanceEstimates
        +warnings: List[str]
        +info: Dict
    }
    
    class ResourceEstimates {
        +LUT: int
        +FF: int
        +BRAM: int
        +URAM: int
        +DSP: int
    }
    
    class PerformanceEstimates {
        +max_freq_mhz: float
        +latency_cycles: int
        +ii_cycles: int
        +throughput_ops_per_sec: float
    }
    
    BuildMetrics --> ResourceEstimates
    BuildMetrics --> PerformanceEstimates
```

## 13. Combined Phase 2 and 3 Execution Flow

```mermaid
stateDiagram-v2
    accTitle: Combined Phase 2 Explorer with Phase 3 Build Execution
    accDescr: Shows how Phase 2 exploration orchestrates Phase 3 build runs within its state machine

    [*] --> Initialize
    Initialize --> GenerateCombinations
    GenerateCombinations --> RunPhase3
    
    state "Phase 3: Build Runner" as RunPhase3 {
        [*] --> Preprocessing
        Preprocessing --> BuildRunner
        
        state BuildRunner <<choice>>
        BuildRunner --> FINN: Legacy
        BuildRunner --> FINN: Future


        FINN --> BuildRunner
        
        BuildRunner --> Postprocessing
        
        Postprocessing --> MetricsCollection
        MetricsCollection --> [*]
    }
    
    RunPhase3 --> UpdateProgress
    UpdateProgress --> CheckNext

    RunPhase3 --> Aggregate: Collect Results
    
    CheckNext --> GenerateCombinations: More Configs
    CheckNext --> Aggregate: All Complete
    Aggregate --> OptimizeResults
    OptimizeResults --> [*]

    note right of UpdateProgress: ETA Calculation
    note right of Aggregate: Pareto Analysis
```

## 14. State Diagram Node Types Reference

```mermaid
stateDiagram-v2
    accTitle: State Diagram Special Node Types
    accDescr: Shows all the different node types available in stateDiagram-v2

    [*] --> Regular: Start
    Regular --> Choice
    
    state Choice <<choice>>
    Choice --> Path1: condition 1
    Choice --> Path2: condition 2
    
    Path1 --> Fork
    Path2 --> Fork
    
    state Fork <<fork>>
    Fork --> Parallel1
    Fork --> Parallel2
    Fork --> Parallel3
    
    state Join <<join>>
    Parallel1 --> Join
    Parallel2 --> Join
    Parallel3 --> Join
    
    Join --> Composite
    
    state "Composite State" as Composite {
        [*] --> Inner1
        Inner1 --> Inner2
        Inner2 --> [*]
    }
    
    Composite --> Final
    Final --> [*]: End
    
    note left of Regular : Regular state
    note right of Choice : Choice (diamond)
    note left of Fork : Fork (splits flow)
    note right of Join : Join (merges flow)
    note left of Composite : Contains substates
```

## Summary

These diagrams visualize the key architectural components of Brainsmith:

1. **Three-Phase Pipeline**: The core DSE workflow
2. **Plugin System**: Zero-discovery registration with rich metadata
3. **Data Flow**: How information moves through each phase
4. **State Machines**: Exploration process states
5. **Transform Stages**: Logical grouping of transformations
6. **Configuration**: Priority-based configuration resolution
7. **Hooks**: Extension points for customization
8. **Error Handling**: Comprehensive error categorization
9. **Framework Integration**: Clean external framework adaptation

Each diagram follows Mermaid best practices with accessibility titles, clear labeling, and appropriate diagram types for the concepts being illustrated.