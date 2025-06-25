# Brainsmith Core V3 - Detailed Architecture Visualization

## Component Interaction Diagram

```mermaid
graph TB
    subgraph "User Interface"
        UI[User Code]
    end
    
    subgraph "Phase 1: Design Space Constructor"
        subgraph "Inputs"
            ONNX[ONNX Model]
            BP[Blueprint YAML]
        end
        
        subgraph "Core Components"
            FORGE[Forge API]
            PARSER[Blueprint Parser]
            VAL[Validator]
        end
        
        subgraph "Parsed Components"
            HWC[HW Compiler Space]
            PSC[Processing Space]
            SC[Search Config]
            GC[Global Config]
        end
        
        subgraph "Output"
            DS[Design Space Object]
        end
        
        ONNX --> FORGE
        BP --> FORGE
        FORGE --> PARSER
        PARSER --> HWC
        PARSER --> PSC
        PARSER --> SC
        PARSER --> GC
        HWC --> VAL
        PSC --> VAL
        SC --> VAL
        GC --> VAL
        VAL --> DS
    end
    
    subgraph "Phase 2: Design Space Explorer"
        subgraph "Core Engine"
            EXP[Explorer Engine]
            CGEN[Combination Generator]
            RAGG[Results Aggregator]
        end
        
        subgraph "Extension Points"
            HOOKS[Exploration Hooks]
            STRAT[Strategy Interface]
        end
        
        subgraph "Data Flow"
            CONFIGS[Build Configs]
            RESULTS[Build Results]
            DATASET[Results Dataset]
        end
        
        DS --> EXP
        EXP --> CGEN
        CGEN --> CONFIGS
        CONFIGS --> EXP
        EXP --> HOOKS
        HOOKS -.-> STRAT
        RESULTS --> RAGG
        RAGG --> DATASET
    end
    
    subgraph "Phase 3: Build Runner"
        subgraph "Pipeline"
            BR[Build Runner]
            PRE[Preprocessor]
            BACK[Backend Executor]
            POST[Postprocessor]
        end
        
        subgraph "Backends"
            FINN[FINN Backend]
            LEGACY[Legacy Backend]
            FUTURE[Future Backends]
        end
        
        subgraph "Metrics"
            BM[Build Metrics]
        end
        
        CONFIGS --> BR
        BR --> PRE
        PRE --> BACK
        BACK --> FINN
        BACK --> LEGACY
        BACK -.-> FUTURE
        FINN --> POST
        POST --> BM
        BM --> RESULTS
    end
    
    subgraph "Output"
        REC[Recommendations]
        REPORT[Analysis Report]
    end
    
    UI --> FORGE
    DATASET --> REC
    DATASET --> REPORT
    REC --> UI
    REPORT --> UI
```

## Data Structure Flow

```mermaid
flowchart LR
    subgraph "Input Data"
        M[ONNX Model]
        B[Blueprint YAML]
    end
    
    subgraph "Phase 1 Data"
        DS[DesignSpace]
        HW[HWCompilerSpace]
        PS[ProcessingSpace]
        SC[SearchConfig]
        GC[GlobalConfig]
        
        DS --> HW
        DS --> PS
        DS --> SC
        DS --> GC
    end
    
    subgraph "Phase 2 Data"
        BC[BuildConfig]
        HWC[HWCompilerConfig]
        PSC[ProcessingConfig]
        
        BC --> HWC
        BC --> PSC
    end
    
    subgraph "Phase 3 Data"
        BR[BuildResult]
        BM[BuildMetrics]
        
        BR --> BM
    end
    
    subgraph "Output Data"
        ER[ExplorationResults]
        REC[Recommendations]
        
        ER --> REC
    end
    
    M --> DS
    B --> DS
    DS --> BC
    BC --> BR
    BR --> ER
```

## Hook System Architecture

```mermaid
classDiagram
    class ExplorationHook {
        <<abstract>>
        +on_exploration_start(DesignSpace)
        +on_combination_generated(BuildConfig)
        +on_build_complete(BuildResult)
        +on_exploration_complete(ExplorationResults)
    }
    
    class LoggingHook {
        +on_exploration_start(DesignSpace)
        +on_combination_generated(BuildConfig)
        +on_build_complete(BuildResult)
        +on_exploration_complete(ExplorationResults)
    }
    
    class CachingHook {
        -cache: Dict
        +on_combination_generated(BuildConfig)
        +on_build_complete(BuildResult)
    }
    
    class EarlyStoppingHook {
        -criteria: StoppingCriteria
        +on_build_complete(BuildResult)
        +should_stop(): bool
    }
    
    class MLGuidedHook {
        -model: PredictiveModel
        +on_build_complete(BuildResult)
        +suggest_next(DesignSpace): BuildConfig
    }
    
    ExplorationHook <|-- LoggingHook
    ExplorationHook <|-- CachingHook
    ExplorationHook <|-- EarlyStoppingHook
    ExplorationHook <|-- MLGuidedHook
```

## Backend Integration Pattern

```mermaid
classDiagram
    class BackendExecutor {
        <<abstract>>
        +execute(ProcessedConfig): BackendResult
        +validate_config(ProcessedConfig): bool
        +get_capabilities(): BackendCapabilities
    }
    
    class FINNBackend {
        -workflow: FINNWorkflow
        +execute(ProcessedConfig): BackendResult
        +validate_config(ProcessedConfig): bool
        +get_capabilities(): BackendCapabilities
    }
    
    class LegacyFINNBackend {
        -legacy_api: LegacyAPI
        +execute(ProcessedConfig): BackendResult
        +validate_config(ProcessedConfig): bool
        +get_capabilities(): BackendCapabilities
    }
    
    class MockBackend {
        -mock_data: Dict
        +execute(ProcessedConfig): BackendResult
        +validate_config(ProcessedConfig): bool
        +get_capabilities(): BackendCapabilities
    }
    
    class BackendFactory {
        +create_backend(BackendType): BackendExecutor
        +register_backend(BackendType, BackendClass)
    }
    
    BackendExecutor <|-- FINNBackend
    BackendExecutor <|-- LegacyFINNBackend
    BackendExecutor <|-- MockBackend
    BackendFactory --> BackendExecutor
```

## Example Blueprint Structure

```yaml
# bert_exploration.yaml
version: "3.0"

hw_compiler:
  kernels:
    matmul:
      variants: ["rtl_optimized", "hls_balanced"]
      parameters:
        precision: [8, 4]
    
    attention:
      variants: ["standard", "flash_attention"]
      parameters:
        head_dim: [64, 32]
    
    layernorm:
      variants: ["streaming", "parallel"]
  
  transforms:
    quantization:
      strategies: ["symmetric", "asymmetric"]
      bits: [8, 4]
    
    folding:
      strategies: ["layer_wise", "operator_wise"]
    
    memory_optimization:
      enabled: [true, false]
  
  build_steps:
    - synthesize
    - optimize
    - place_route
  
  config_flags:
    target_device: "U250"
    clock_target: 300  # MHz

processing:
  preprocessing:
    model_optimization:
      graph_optimization: [true, false]
      constant_folding: [true]
    
    input_processing:
      normalization: ["standard", "none"]
  
  postprocessing:
    analysis:
      performance_profiling: true
      resource_analysis: true
      accuracy_validation: true
    
    reporting:
      formats: ["json", "html"]

search:
  strategy: "exhaustive"
  
  constraints:
    max_lut_utilization: 0.85
    min_throughput: 1000  # inferences/sec
    max_latency: 10  # milliseconds
  
  early_stopping:
    enabled: false  # For future use
    criteria: "convergence"

global:
  output_stage: "rtl"
  working_directory: "./exploration_builds"
  cache_results: true
  parallel_builds: 4
  
  reporting:
    save_all_results: true
    generate_pareto_plot: true
```