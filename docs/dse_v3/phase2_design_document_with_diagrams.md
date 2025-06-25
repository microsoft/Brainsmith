# Phase 2: Design Space Explorer - Design Document

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Hook System](#hook-system)
6. [Exploration Process](#exploration-process)
7. [Result Analysis](#result-analysis)
8. [Integration Points](#integration-points)
9. [Design Rationale](#design-rationale)

## Overview

The Design Space Explorer (Phase 2) systematically explores the configuration space defined by Phase 1, executing builds through Phase 3, and collecting results for analysis. It serves as the orchestration layer that bridges configuration definition with actual hardware compilation.

### Key Responsibilities
- Generate all valid configurations from the design space
- Execute builds in a controlled manner
- Track progress and support resumability
- Aggregate and analyze results
- Provide extensibility through hooks

## Architecture

```mermaid
graph TB
    subgraph "Phase 2: Design Space Explorer"
        EE[ExplorerEngine]
        CG[CombinationGenerator]
        RA[ResultsAggregator]
        PT[ProgressTracker]
        HS[Hook System]
        
        EE --> CG
        EE --> RA
        EE --> PT
        EE --> HS
    end
    
    subgraph "Phase 1"
        DS[DesignSpace]
    end
    
    subgraph "Phase 3"
        BR[BuildRunner]
    end
    
    DS --> EE
    EE --> BR
    BR --> EE
    
    subgraph "External"
        U[User]
        FS[File System]
    end
    
    U --> EE
    HS --> FS
```

### Component Relationships

```mermaid
classDiagram
    class ExplorerEngine {
        -build_runner_factory
        -hooks
        -progress_tracker
        -exploration_results
        +explore(design_space, resume_from)
        -evaluate_config(config, build_runner)
        -should_stop_early(design_space, index)
        -fire_hook(method_name, args)
    }
    
    class CombinationGenerator {
        +generate_all(design_space)
        +filter_by_indices(configs, indices)
        +filter_by_resume(configs, last_id)
        -generate_design_space_id(design_space)
        -satisfies_constraints(config, constraints)
    }
    
    class ResultsAggregator {
        -results
        +add_result(result)
        +finalize()
        -find_best_config()
        -find_pareto_optimal()
        -calculate_metrics_summary()
    }
    
    class ProgressTracker {
        +total_configs
        +completed
        +successful
        +failed
        +update(result)
        +get_eta()
        +get_summary()
    }
    
    class ExplorationHook {
        <<interface>>
        +on_exploration_start()
        +on_combinations_generated()
        +on_build_complete()
        +on_exploration_complete()
    }
    
    class BuildRunnerInterface {
        <<interface>>
        +run(config)
    }
    
    ExplorerEngine --> CombinationGenerator
    ExplorerEngine --> ResultsAggregator
    ExplorerEngine --> ProgressTracker
    ExplorerEngine --> ExplorationHook
    ExplorerEngine --> BuildRunnerInterface
    
    LoggingHook --|> ExplorationHook
    CachingHook --|> ExplorationHook
```

## Core Components

### 1. ExplorerEngine

The central orchestrator that manages the entire exploration process.

```python
class ExplorerEngine:
    """
    Main engine for design space exploration.
    
    Responsibilities:
    - Coordinate all exploration activities
    - Manage build execution
    - Fire hooks at appropriate points
    - Handle early stopping conditions
    """
```

**Key Methods:**
- `explore()`: Main entry point for exploration
- `_evaluate_config()`: Execute a single build safely
- `_should_stop_early()`: Check stopping conditions
- `_fire_hook()`: Invoke hook methods with error handling

### 2. CombinationGenerator

Generates all valid configurations from the design space using cartesian products while preserving transform stage information.

```mermaid
graph TB
    subgraph "Design Space Input"
        K[Kernels: Gemm, Conv]
        TS[Transform Stages]
        P[Processing: resize]
    end
    
    subgraph "Transform Stages"
        T1[pre_quantization: fold, ~absorb]
        T2[quantization: quantize]
        T3[post_quantization: streamline]
    end
    
    subgraph "Stage-Aware Generation"
        SG[Stage Combinations]
        CP[Cartesian Product]
        F[Filter Empty/Optional]
        BC[BuildConfigs with Stages]
    end
    
    K --> CP
    TS --> SG
    SG --> CP
    P --> CP
    CP --> F
    F --> BC
    
    T1 --> TS
    T2 --> TS
    T3 --> TS
```

**Enhanced Generation Logic:**
1. Extract kernel combinations (with backends)
2. **Extract transform combinations by stage** (preserving execution order)
3. Extract processing step combinations
4. Generate cartesian product across all components
5. Filter empty/skipped elements while maintaining stage structure
6. Create BuildConfig objects with stage-organized transforms
7. Apply pre-build constraints

### 3. Data Structures

```mermaid
classDiagram
    class BuildConfig {
        +id: str
        +design_space_id: str
        +kernels: List[Tuple[str, List[str]]]
        +transforms: Dict[str, List[str]]
        +preprocessing: List[ProcessingStep]
        +postprocessing: List[ProcessingStep]
        +build_steps: List[str]
        +config_flags: Dict
        +global_config: GlobalConfig
        +output_dir: str
        +combination_index: int
        +total_combinations: int
    }
    
    class BuildResult {
        +config_id: str
        +status: BuildStatus
        +metrics: Optional[BuildMetrics]
        +start_time: datetime
        +end_time: Optional[datetime]
        +duration_seconds: float
        +artifacts: Dict[str, str]
        +logs: Dict[str, str]
        +error_message: Optional[str]
        +complete(status, error_message)
    }
    
    class ExplorationResults {
        +design_space_id: str
        +evaluations: List[BuildResult]
        +total_combinations: int
        +best_config: Optional[BuildConfig]
        +pareto_optimal: List[BuildConfig]
        +metrics_summary: Dict
        +add_config(config)
        +get_config(id)
        +update_counts()
    }
    
    BuildConfig --> BuildResult : produces
    BuildResult --> ExplorationResults : aggregated into
```

## Data Flow

### Exploration Lifecycle

```mermaid
sequenceDiagram
    participant User
    participant Explorer
    participant CombGen
    participant BuildRunner
    participant Aggregator
    participant Hooks
    
    User->>Explorer: explore(design_space)
    Explorer->>Hooks: on_exploration_start()
    
    Explorer->>CombGen: generate_all()
    CombGen-->>Explorer: List[BuildConfig]
    Explorer->>Hooks: on_combinations_generated()
    
    loop For each config
        Explorer->>BuildRunner: run(config)
        BuildRunner-->>Explorer: BuildResult
        Explorer->>Aggregator: add_result()
        Explorer->>Hooks: on_build_complete()
        
        alt Early stopping
            Explorer->>Explorer: should_stop_early()
            Explorer-->>User: Partial results
        end
    end
    
    Explorer->>Aggregator: finalize()
    Aggregator->>Aggregator: find_best_config()
    Aggregator->>Aggregator: find_pareto_optimal()
    Explorer->>Hooks: on_exploration_complete()
    Explorer-->>User: ExplorationResults
```

### Build Execution Flow

```mermaid
graph TD
    BC[BuildConfig] --> BR[BuildRunner]
    BR --> |Success| SM[Success Metrics]
    BR --> |Failure| EM[Error Message]
    BR --> |Timeout| TO[Timeout Status]
    
    SM --> BM[BuildMetrics]
    BM --> |Throughput| TP
    BM --> |Latency| LT
    BM --> |Resources| RS
    
    EM --> ER[Error Result]
    TO --> TR[Timeout Result]
    
    BM --> AR[Aggregated Results]
    ER --> AR
    TR --> AR
```

## Hook System

The hook system provides extensibility without modifying core logic.

### Hook Lifecycle

```mermaid
stateDiagram-v2
    [*] --> ExplorationStart
    ExplorationStart --> CombinationsGenerated
    CombinationsGenerated --> BuildLoop
    
    state BuildLoop {
        [*] --> BuildStart
        BuildStart --> BuildComplete
        BuildComplete --> [*]
    }
    
    BuildLoop --> ExplorationComplete
    ExplorationComplete --> [*]
    
    note right of ExplorationStart : on_exploration_start()
    note right of CombinationsGenerated : on_combinations_generated()
    note right of BuildComplete : on_build_complete()
    note right of ExplorationComplete : on_exploration_complete()
```

### Built-in Hooks

#### LoggingHook
Provides detailed logging throughout exploration:

```
================================================================================
DESIGN SPACE EXPLORATION STARTED
================================================================================
Model: /path/to/model.onnx
Design Space ID: dse_a1b2c3d4
Search Strategy: exhaustive
Total Combinations: 100
================================================================================

✅ Build config_001 (1/100): success | Throughput: 1234.56 | Latency: 10.23μs
❌ Build config_002 (2/100): failed | Error: Timing constraints not met
```

#### CachingHook
Enables resume functionality:

```mermaid
graph LR
    subgraph "First Run"
        C1[Config 1] --> R1[Result 1]
        C2[Config 2] --> R2[Result 2]
        C3[Config 3] --> X[Crash/Stop]
    end
    
    subgraph "Cache Files"
        CF[dse_xxx_results.jsonl]
        SF[dse_xxx_summary.json]
    end
    
    subgraph "Resume Run"
        RC[Load Cache] --> C4[Config 4]
        C4 --> R4[Result 4]
    end
    
    R1 --> CF
    R2 --> CF
    CF --> RC
```

## Exploration Process

### 1. Combination Generation

Transform stages are preserved during combination generation to maintain execution order:

```python
# Example: 2 kernels with backends, transforms organized by stage
kernels = [("Gemm", ["rtl", "hls"]), ("Conv", ["hls"])]
transforms = {
    "pre_quantization": ["fold", "~absorb_transpose"],  # optional absorb
    "quantization": ["quantize"],
    "post_quantization": ["streamline"]
}

# Generates BuildConfigs with transforms organized by stage:
# Config 1: 
#   Kernels: Gemm[rtl], Conv[hls]
#   Transforms: {
#     "pre_quantization": ["fold", "absorb_transpose"],
#     "quantization": ["quantize"],
#     "post_quantization": ["streamline"]
#   }
# Config 2: 
#   Kernels: Gemm[rtl], Conv[hls]
#   Transforms: {
#     "pre_quantization": ["fold"],  # without optional absorb_transpose
#     "quantization": ["quantize"],
#     "post_quantization": ["streamline"]
#   }
# ... etc
```

This ensures transforms are executed in the correct order during Phase 3 build execution.

### 2. Progress Tracking

```
Progress: 45/100 (45.0%) | Success: 40 (88.9%) | Failed: 5 | Skipped: 0
Elapsed: 15.3m | Avg build: 20.4s | ETA: 18.7m (14:35:20)
```

### 3. Early Stopping

```mermaid
graph TD
    E[Evaluate Config] --> C1{Max Evals?}
    C1 -->|Yes| S[Stop]
    C1 -->|No| C2{Timeout?}
    C2 -->|Yes| S
    C2 -->|No| C3{Constraints Met?}
    C3 -->|Yes| S
    C3 -->|No| N[Next Config]
```

## Result Analysis

### 1. Best Configuration
Selected based on primary metric (throughput):

```python
best_config = max(successful_results, key=lambda r: r.metrics.throughput)
```

### 2. Pareto Frontier
Multi-objective optimization (throughput vs resources):

```mermaid
graph LR
    subgraph "Objective Space"
        A[Config A: 1200 tps, 80% util]
        B[Config B: 1000 tps, 60% util]
        C[Config C: 800 tps, 70% util]
        D[Config D: 900 tps, 40% util]
    end
    
    A --> |Pareto| PF[Pareto Frontier]
    B --> |Pareto| PF
    D --> |Pareto| PF
    C --> |Dominated| X[Excluded]
```

### 3. Metrics Summary

```python
{
    "throughput": {
        "min": 500.0,
        "max": 1500.0,
        "mean": 1000.0,
        "std": 250.0
    },
    "latency": {
        "min": 5.0,
        "max": 20.0,
        "mean": 12.5,
        "std": 3.5
    },
    # ... other metrics
}
```

## Integration Points

### Phase 1 Integration

```mermaid
graph LR
    subgraph "Phase 1 Output"
        DS[DesignSpace]
        HW[HWCompilerSpace]
        PS[ProcessingSpace]
        SC[SearchConfig]
    end
    
    subgraph "Phase 2 Input"
        CG[CombinationGenerator]
        EE[ExplorerEngine]
    end
    
    DS --> CG
    HW --> CG
    PS --> CG
    SC --> EE
```

### Phase 3 Integration

Phase 3 receives BuildConfigs with transforms organized by stages, enabling proper execution order:

```python
class BuildRunnerInterface(ABC):
    @abstractmethod
    def run(self, config: BuildConfig) -> BuildResult:
        """Execute build and return results."""
        pass

# Phase 3 implements this interface
class FINNBuildRunner(BuildRunnerInterface):
    def run(self, config: BuildConfig) -> BuildResult:
        # Execute transforms in stage order
        for stage_name in ["pre_quantization", "quantization", "post_quantization"]:
            stage_transforms = config.transforms.get(stage_name, [])
            for transform in stage_transforms:
                self._apply_transform(transform)
        
        # Continue with hardware compilation...
        pass
```

**Benefits of Stage-Based Execution:**
- **Correct Transform Order**: Transforms execute in their intended phases
- **Clear Pipeline Structure**: Each stage has a specific purpose in the compilation flow
- **Better Error Handling**: Stage-specific error reporting and recovery
- **Debugging Support**: Can halt execution at specific stages for analysis

## Design Rationale

### 1. Separation of Concerns
- **CombinationGenerator**: Only generates configs, no execution
- **ExplorerEngine**: Only orchestrates, no result analysis
- **ResultsAggregator**: Only analyzes, no exploration logic

### 2. Extensibility Through Hooks
- No modification of core logic needed
- Multiple hooks can be composed
- Hooks are isolated from each other

### 3. Stateless Design
- Each exploration is independent
- Supports parallel execution
- Easier testing and debugging

### 4. Resume Support
- Built-in fault tolerance
- Minimal overhead when not used
- Transparent to exploration logic

### 5. Mock Build Runner
- Enables testing without Phase 3
- Predictable behavior for unit tests
- Configurable success rates and metrics

### 6. Transform Stage Organization
- **Execution Order Preservation**: Transforms maintain their intended phases (pre_quantization → quantization → post_quantization)
- **Blueprint Fidelity**: Direct mapping from blueprint configuration to build execution
- **Clear Separation**: Each stage has distinct responsibilities in the compilation pipeline
- **Enhanced Debugging**: Stage-specific logging and error reporting
- **Future Extensibility**: Easy to add stage-specific optimizations or conditional execution

## Error Handling

```mermaid
graph TD
    E[Execute Build] --> T{Try}
    T --> |Success| R[Return Result]
    T --> |Exception| C[Catch]
    C --> F[Create Failed Result]
    F --> L[Log Error]
    L --> R
    
    H[Fire Hook] --> HT{Try}
    HT --> |Success| OK[Continue]
    HT --> |Exception| HC[Catch]
    HC --> HL[Log Hook Error]
    HL --> OK
```

## Performance Considerations

### 1. Memory Usage
- Configs generated on-demand (generator pattern possible)
- Results stored incrementally
- Cache files use line-delimited JSON

### 2. Scalability
- O(n) exploration time (n = combinations)
- Constant memory per configuration
- Linear scaling with parallel builds

### 3. Optimization Opportunities
- Parallel build execution
- Distributed exploration
- Smart sampling strategies
- Constraint-based pruning

## Output Directory Structure

Phase 2 creates an intelligent directory structure for organizing exploration outputs:

```
{working_directory}/
└── {design_space_id}/                    # Root for this exploration
    ├── exploration_summary.json          # Overall exploration results
    ├── exploration_cache.jsonl           # Hook data (from CachingHook)
    ├── exploration_log.txt              # Exploration-level logs
    ├── pareto_configs.json              # Pareto-optimal configurations
    └── builds/                          # Individual build directories
        ├── config_0/                    # Dynamic padding based on total
        ├── config_1/                    # e.g., config_01 for 100 total
        └── config_N/                    # config_001 for 1000 total
```

### Key Features

1. **Dynamic Padding**: Directory names use the minimum digits needed
   - 10 configs: `config_0` to `config_9`
   - 100 configs: `config_00` to `config_99`
   - 1000 configs: `config_000` to `config_999`

2. **Phase Separation**: 
   - Phase 2 creates the directory structure and passes paths via `BuildConfig.output_dir`
   - Phase 3 creates actual build artifacts within its assigned directory

3. **Exploration-Level Data**: All exploration-wide data (hooks, summaries) stays in the parent directory

## Summary

Phase 2 provides a robust, extensible exploration engine that:
- ✅ Systematically explores all configurations
- ✅ Provides progress tracking and resumability
- ✅ Enables custom behavior through hooks
- ✅ Analyzes results for optimal configurations
- ✅ Integrates cleanly with Phases 1 and 3
- ✅ Organizes outputs in a scalable directory structure

The design prioritizes simplicity, extensibility, and reliability while avoiding premature optimization and unnecessary complexity.