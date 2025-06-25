# DSE V3 Architecture Diagrams

## System Context Diagram

```mermaid
graph TB
    subgraph "External Inputs"
        ONNX[ONNX Model]
        YAML[Blueprint YAML]
        RUNNER[Build Runner Implementation]
    end
    
    subgraph "DSE V3 Core"
        API[Public API]
        P1[Phase 1: Constructor]
        P2[Phase 2: Explorer]
        P3[Phase 3: Runner Interface]
    end
    
    subgraph "Outputs"
        DS[Design Space]
        RESULTS[Exploration Results]
        ARTIFACTS[Build Artifacts]
    end
    
    ONNX --> API
    YAML --> API
    API --> P1
    P1 --> DS
    DS --> P2
    RUNNER --> P3
    P3 --> P2
    P2 --> RESULTS
    P2 --> ARTIFACTS
    
    style ONNX fill:#e3f2fd
    style YAML fill:#e3f2fd
    style RUNNER fill:#e3f2fd
    style DS fill:#fff3e0
    style RESULTS fill:#e8f5e9
    style ARTIFACTS fill:#e8f5e9
```

## Data Flow Diagram

```mermaid
flowchart LR
    subgraph "Input Processing"
        YAML[Blueprint YAML]
        PARSE[Parse & Validate]
        DS[Design Space]
        
        YAML --> PARSE
        PARSE --> DS
    end
    
    subgraph "Combination Generation"
        DS --> KERNS[Kernel Combinations]
        DS --> TRANS[Transform Combinations]
        DS --> PROCS[Processing Combinations]
        
        KERNS --> CART[Cartesian Product]
        TRANS --> CART
        PROCS --> CART
        
        CART --> CONFIGS[Build Configs]
    end
    
    subgraph "Build Execution"
        CONFIGS --> QUEUE[Build Queue]
        QUEUE --> RUNNER[Build Runner]
        RUNNER --> RESULT[Build Result]
        RESULT --> AGG[Aggregator]
    end
    
    subgraph "Results Analysis"
        AGG --> BEST[Best Config]
        AGG --> PARETO[Pareto Frontier]
        AGG --> STATS[Statistics]
        
        BEST --> FINAL[Exploration Results]
        PARETO --> FINAL
        STATS --> FINAL
    end
```

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "Phase 1 Components"
        BP[BlueprintParser]
        VAL[Validator]
        EXC[Exception Handler]
        
        BP --> VAL
        VAL --> EXC
    end
    
    subgraph "Phase 2 Components"
        ENG[Explorer Engine]
        GEN[Combination Generator]
        TRACK[Progress Tracker]
        AGG[Results Aggregator]
        HOOK[Hook System]
        
        ENG --> GEN
        ENG --> TRACK
        ENG --> AGG
        ENG --> HOOK
        
        HOOK --> LOG[Logging Hook]
        HOOK --> CACHE[Caching Hook]
        HOOK --> CUSTOM[Custom Hooks]
    end
    
    subgraph "Interfaces"
        IRUN[BuildRunner Interface]
        IHOOK[Hook Interface]
        
        IRUN --> MOCK[Mock Runner]
        IRUN --> REAL[Real Runner]
    end
    
    VAL --> ENG
    ENG --> IRUN
```

## State Machine Diagram

```mermaid
stateDiagram-v2
    [*] --> Initialized
    
    Initialized --> Parsing: forge()
    Parsing --> Validating: parse complete
    Validating --> Ready: validation passed
    Validating --> Error: validation failed
    
    Ready --> Exploring: explore()
    Exploring --> Generating: start exploration
    Generating --> Building: combinations ready
    
    Building --> Building: next config
    Building --> Aggregating: all configs done
    Building --> Stopped: early stop
    
    Aggregating --> Complete: finalize results
    
    Complete --> [*]
    Error --> [*]
    Stopped --> [*]
    
    state Building {
        [*] --> Queued
        Queued --> Running
        Running --> Success
        Running --> Failed
        Running --> Timeout
        Success --> [*]
        Failed --> [*]
        Timeout --> [*]
    }
```

## Class Hierarchy Diagram

```mermaid
classDiagram
    class BrainsmithError {
        <<abstract>>
        +message: str
    }
    
    class BlueprintParseError {
        +line: int
        +column: int
    }
    
    class ValidationError {
        +errors: List[str]
        +warnings: List[str]
    }
    
    class ConfigurationError {
        +config_type: str
    }
    
    BrainsmithError <|-- BlueprintParseError
    BrainsmithError <|-- ValidationError
    BrainsmithError <|-- ConfigurationError
    
    class BuildStatus {
        <<enumeration>>
        PENDING
        RUNNING
        SUCCESS
        FAILED
        TIMEOUT
        SKIPPED
    }
    
    class SearchStrategy {
        <<enumeration>>
        EXHAUSTIVE
        RANDOM
        GENETIC
        BAYESIAN
    }
    
    class OutputStage {
        <<enumeration>>
        ONNX
        QUANTIZED
        INTERMEDIATE
        OPTIMIZED
        RTL
        BITSTREAM
    }
```

## Sequence Diagram: Complete Workflow

```mermaid
sequenceDiagram
    participant User
    participant API as Core API
    participant P1 as Phase 1
    participant P2 as Phase 2
    participant BR as BuildRunner
    participant FS as FileSystem
    
    User->>API: forge(model, blueprint)
    API->>P1: parse blueprint
    P1->>FS: read YAML
    FS-->>P1: blueprint data
    P1->>P1: validate
    P1-->>API: DesignSpace
    API-->>User: design_space
    
    User->>API: explore(design_space, runner_factory)
    API->>P2: start exploration
    P2->>P2: generate combinations
    
    loop For each config
        P2->>BR: run(config)
        BR->>FS: write build files
        BR->>BR: execute build
        BR->>FS: read metrics
        BR-->>P2: BuildResult
        P2->>P2: aggregate results
    end
    
    P2->>P2: analyze results
    P2-->>API: ExplorationResults
    API-->>User: results
```

## Deployment Diagram

```mermaid
graph TB
    subgraph "User Environment"
        CLI[Brainsmith CLI]
        PY[Python Script]
        NB[Jupyter Notebook]
    end
    
    subgraph "DSE V3 Core"
        API[API Layer]
        CORE[Core Logic]
        INTF[Interfaces]
    end
    
    subgraph "Build Infrastructure"
        LOCAL[Local Runner]
        CLUSTER[Cluster Runner]
        CLOUD[Cloud Runner]
    end
    
    subgraph "Storage"
        WORK[Working Directory]
        CACHE[Cache Directory]
        ARTIFACTS[Artifacts Store]
    end
    
    CLI --> API
    PY --> API
    NB --> API
    
    API --> CORE
    CORE --> INTF
    
    INTF --> LOCAL
    INTF --> CLUSTER
    INTF --> CLOUD
    
    LOCAL --> WORK
    CLUSTER --> WORK
    CLOUD --> WORK
    
    CORE --> CACHE
    CORE --> ARTIFACTS
```

## Error Handling Flow

```mermaid
flowchart TD
    START[Operation Start]
    
    START --> TRY{Try Operation}
    
    TRY -->|Parse Error| PE[BlueprintParseError]
    TRY -->|Validation Error| VE[ValidationError]
    TRY -->|Config Error| CE[ConfigurationError]
    TRY -->|Build Error| BE[BuildError]
    TRY -->|Success| SUCCESS[Continue]
    
    PE --> LOG1[Log Error Details]
    VE --> LOG2[Log Errors & Warnings]
    CE --> LOG3[Log Config Issue]
    BE --> LOG4[Log Build Failure]
    
    LOG1 --> HANDLE[Error Handler]
    LOG2 --> HANDLE
    LOG3 --> HANDLE
    LOG4 --> HANDLE
    
    HANDLE --> RECOVER{Recoverable?}
    
    RECOVER -->|Yes| RETRY[Retry/Skip]
    RECOVER -->|No| FAIL[Raise Exception]
    
    RETRY --> TRY
    SUCCESS --> END[Operation Complete]
    FAIL --> END
```

## Performance Optimization Points

```mermaid
graph LR
    subgraph "Optimization Opportunities"
        PARSE[Blueprint Parsing<br/>Cache parsed results]
        
        COMB[Combination Generation<br/>Lazy evaluation]
        
        BUILD[Build Execution<br/>Parallel builds]
        
        AGG[Results Aggregation<br/>Incremental updates]
        
        CACHE[Caching Layer<br/>Skip completed builds]
    end
    
    PARSE --> COMB
    COMB --> BUILD
    BUILD --> AGG
    CACHE -.-> BUILD
    
    style PARSE fill:#fff3e0
    style COMB fill:#fff3e0
    style BUILD fill:#ffebee
    style AGG fill:#fff3e0
    style CACHE fill:#e8f5e9
```

These diagrams provide different perspectives on the DSE V3 architecture, helping to understand:
- System boundaries and interactions
- Data flow through the system
- Component relationships
- State transitions
- Error handling strategies
- Deployment options
- Performance optimization points