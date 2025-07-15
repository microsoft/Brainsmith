# Diagram 13 Variations: Combined Phase 2 and 3 Execution Flow

```mermaid
%%{ init: { "theme": "dark",
           "flowchart": { "curve": "basis", "nodeSpacing": 40, "rankSpacing": 30 } } }%%
flowchart LR
    %% ---------- Phase-2 control loop ----------
    Start([Start]) --> Init[Initialize Explorer]
    Init --> Gen[Generate Combinations]
    Gen --> Next{Next Config?}
    Next -- No --> Agg[Aggregate Results] --> Pareto[Pareto Analysis] --> End([End])

    %% ---------- Phase-3 worker (kept vertical) ----------
    subgraph P3["Phase 3 Runner"]
        direction TB
        Pre[Pre-processing] --> B{Backend?}
        B -- legacy --> L[FINN Legacy]
        B -- future --> F[FINN Future]
        L & F --> Post[Post-processing] --> M[Collect Metrics]
        M --> Prog[Update Progress]
    end

    %% ---------- loop wiring ----------
    Next -- Yes --> Pre
    Prog --> Store[(Store Result)]
    Store --> Next
```

```mermaid
%%{ init: { "theme":"dark",
           "flowchart": { "curve":"basis",
                          "nodeSpacing":40,
                          "rankSpacing":30 } } }%%

flowchart LR
    %% ---------- palette ----------
    classDef subgraphFill fill:#374151,stroke:#1f2937,stroke-width:2px;
    classDef primary      fill:#111827,stroke:#374151,color:#fff;
    classDef decision     fill:#7c3aed,stroke:#6b21a8,color:#fff;
    classDef green        fill:#059669,stroke:#047857,color:#fff;

    %% ---------- Phase-2 control loop ----------
    DesignSpace([DesignSpace]):::green --> Init[Initialize Explorer]:::primary
    Init --> Gen[Generate Combinations]:::primary
    Gen  --> Next{Next Config?}:::decision

    Next -- No --> Agg[Aggregate Results]:::primary
    Agg  --> Pareto[Pareto Analysis]:::primary
    Pareto --> End([End]):::primary

    %% ---------- Phase-3 worker ----------
    subgraph P3["Phase 3 Runner"]
        direction TB
        Pre[Pre-processing]:::primary --> B{Backend?}:::decision
        B -- legacy --> L[FINN Legacy]:::primary
        B -- future --> F[FINN Future]:::primary
        L & F --> Post[Post-processing]:::primary
        Post --> M[Collect Metrics]:::primary
        M --> Prog[Update Progress]:::primary
    end
    class P3 subgraphFill;

    %% ---------- loop wiring ----------
    Next  -- Yes --> Pre
    Prog  --> Store[(Store Result)]:::primary
    Store --> Next
```

## 1. Flowchart Version (Top-Bottom)

```mermaid
flowchart TB
    accTitle: Combined Phase 2 and 3 Execution (Flowchart)
    accDescr: Flowchart showing Phase 2 exploration orchestrating Phase 3 builds

    Start([Start]) --> Init[Initialize Explorer]
    Init --> GenCombo[Generate Combinations]
    GenCombo --> NextConfig{Next Config?}
    
    NextConfig -->|Yes| P3
    NextConfig -->|No| Agg[Aggregate Results]
    
    subgraph P3 ["Phase 3 Runner"]
        Pre[Preprocessing]
        Backend{Backend?}
        Legacy[FINN Legacy]
        Future[FINN Future]
        Post[Postprocessing]
        Metrics[Collect Metrics]
        
        Pre --> Backend
        Backend -->|legacy| Legacy
        Backend -->|future| Future
        Legacy --> Post
        Future --> Post
        Post --> Metrics
    end
    
    Metrics --> Progress[Update Progress]
    Progress --> Results[(Store Result)]
    Results --> NextConfig
    
    Agg --> Optimize[Pareto Analysis]
    Optimize --> End([End])
    
    style P3 fill:#374151,stroke:#1f2937,stroke-width:2px
    style Backend fill:#7c3aed,stroke:#6b21a8,color:#fff
```

## 2. Flowchart Version (Left-Right with ELK)

```mermaid
%%{init: {"flowchart": {"defaultRenderer": "elk"}} }%%
flowchart LR
    accTitle: Combined Phase 2 and 3 Execution (ELK Layout)
    accDescr: Horizontal flow with ELK renderer for better layout

    Start([Start]) --> Init[Initialize]
    Init --> Gen[Generate<br/>Combinations]
    Gen --> Loop{For Each<br/>Config}
    
    Loop -->|Config| Pre[Preprocess]
    Pre --> Backend{Backend<br/>Select}
    Backend -->|Legacy| FINN1[FINN<br/>Legacy]
    Backend -->|Future| FINN2[FINN<br/>Future]
    FINN1 --> Post[Postprocess]
    FINN2 --> Post
    Post --> Metrics[Metrics]
    Metrics --> Store[(Results)]
    Store --> Loop
    
    Loop -->|Done| Agg[Aggregate]
    Agg --> Opt[Optimize]
    Opt --> End([End])
    
    classDef phase2 fill:#7c3aed,stroke:#6b21a8,color:#fff
    classDef phase3 fill:#059669,stroke:#047857,color:#fff
    classDef decision fill:#ea580c,stroke:#c2410c,color:#fff
    
    class Init,Gen,Loop,Agg,Opt phase2
    class Pre,Post,Metrics phase3
    class Backend decision
```

## 3. Sequence Diagram Version

```mermaid
sequenceDiagram
    accTitle: Phase 2 and 3 Interaction Sequence
    accDescr: Shows the temporal interaction between Phase 2 Explorer and Phase 3 Runner

    participant P2 as Phase 2 Explorer
    participant Queue as Config Queue
    participant P3 as Phase 3 Runner
    participant Pre as Preprocessing
    participant Backend as Backend Selector
    participant FINN as FINN Backend
    participant Metrics as Metrics Collector
    participant Results as Results Store

    P2->>Queue: Generate all combinations
    
    P2->>Results: Store result
    loop For each configuration
        P2->>Queue: Get next config
        Queue-->>P2: Config
        P2->>P3: Execute build(config)
        
        rect rgb(5, 150, 105, 0.1)
            note over P3,FINN: Phase 3 Execution
            P3->>Pre: Preprocess model
            Pre->>Backend: Select backend
            alt Legacy API
                Backend->>FINN: Execute legacy
            else Future API
                Backend->>FINN: Execute future
            end
            FINN->>Metrics: Collect metrics
            Metrics-->>P3: Build result
        end
        
        P3-->>P2: Return result
        P2->>P2: Update progress
    end
    
    P2->>Results: Aggregate all
    Results->>P2: Pareto optimal set
```

## 4. Block Diagram Version

```mermaid
block-beta
    columns 4
    
    space P2Title["Phase 2: Explorer"]:2 space
    Init["Initialize"] Gen["Generate Combos"] Queue["Config Queue"] Check{"More?"}
    
    space:4
    space P3Title["Phase 3: Build Runner"]:2 space
    
    Pre["Preprocessing"] space:2 Backend{"Backend"}
    Legacy["FINN Legacy"] space:2 Future["FINN Future"]
    Post["Postprocessing"]:2 Metrics["Metrics"]:2
    
    space:4
    Progress["Update Progress"] Results["Store Result"] Agg["Aggregate"] Opt["Optimize"]
    
    Init --> Gen
    Gen --> Queue
    Queue --> Check
    Check --> Pre
    Pre --> Backend
    Backend --> Legacy
    Backend --> Future
    Legacy --> Post
    Future --> Post
    Post --> Metrics
    Metrics --> Progress
    Progress --> Results
    Results --> Check
    Check --> Agg
    Agg --> Opt
    
    style P2Title fill:#7c3aed,color:#fff
    style P3Title fill:#059669,color:#fff
```

## 5. Gitgraph Version (Creative Use)

```mermaid
gitgraph TB:
    accTitle: Phase 2/3 Execution as Branches
    accDescr: Creative use of gitgraph to show parallel execution paths

    commit id: "Initialize Explorer"
    commit id: "Generate Combinations"
    
    branch config-1
    checkout config-1
    commit id: "Preprocess C1"
    commit id: "FINN Legacy" tag: "backend"
    commit id: "Metrics C1"
    
    checkout main
    branch config-2
    checkout config-2
    commit id: "Preprocess C2"
    commit id: "FINN Future" tag: "backend"
    commit id: "Metrics C2"
    
    checkout main
    merge config-1
    merge config-2
    commit id: "Aggregate Results"
    commit id: "Pareto Analysis" tag: "optimal"
```

## 6. C4 Container Diagram Style (Using Flowchart)

```mermaid
flowchart TB
    accTitle: Phase 2/3 as C4 Containers
    accDescr: C4-style container view of the execution flow

    subgraph "Phase 2 Explorer System"
        E[["Explorer Engine<br/>(Python Process)"]]
        Q[["Config Queue<br/>(In-Memory)"]]
        H[["Hooks System<br/>(Plugin Interface)"]]
    end
    
    subgraph "Phase 3 Build System"
        BR[["Build Runner<br/>(Orchestrator)"]]
        
        subgraph "Backends"
            FL[["FINN Legacy<br/>(External Process)"]]
            FF[["FINN Future<br/>(API Calls)"]]
        end
        
        M[["Metrics Collector<br/>(Parser)"]]
    end
    
    subgraph "Storage"
        R[(Results Cache)]
        L[(Build Logs)]
    end
    
    E --> Q
    Q --> BR
    BR --> FL
    BR --> FF
    FL --> M
    FF --> M
    M --> R
    M --> L
    BR --> E
    H -.-> E
    
    style E fill:#7c3aed,stroke:#6b21a8,color:#fff
    style BR fill:#059669,stroke:#047857,color:#fff
```

## Comparison

Each diagram type offers different strengths:

1. **Flowchart TB**: Best for showing the overall flow and decision points
2. **Flowchart LR (ELK)**: Compact horizontal layout, good for wide displays
3. **Sequence**: Best for showing the temporal interaction and method calls
4. **Block**: Good for showing architectural layers and components
5. **Gitgraph**: Creative visualization of parallel execution paths
6. **C4 Style**: Best for showing system architecture and components

The state diagram (original #13) remains good for showing the state machine nature of the explorer, while these alternatives each emphasize different aspects of the same system.