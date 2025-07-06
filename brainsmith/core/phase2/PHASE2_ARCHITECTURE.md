# Phase 2: Design Space Explorer - Architecture Document

## Overview

Phase 2 of the Brainsmith DSE v3 toolchain is responsible for systematically exploring design spaces created in Phase 1. It generates all valid build configurations, coordinates their execution through Phase 3, and aggregates results to identify optimal solutions. This document provides comprehensive architectural documentation with visual diagrams.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Class Relationships](#class-relationships)
5. [Sequence Diagrams](#sequence-diagrams)
6. [Hook System](#hook-system)
7. [Progress Tracking](#progress-tracking)
8. [Results Analysis](#results-analysis)

## High-Level Architecture

Phase 2 serves as the orchestrator for systematic design space exploration, bridging Phase 1 (space definition) and Phase 3 (build execution).

```mermaid
graph TB
    subgraph "Phase 1 Output"
        DS[DesignSpace Object<br/>Validated Design Space]
    end
    
    subgraph "Phase 2: Design Space Explorer"
        EXP[ExplorerEngine<br/>Main Orchestrator]
        CG[CombinationGenerator<br/>Config Generation]
        AGG[ResultsAggregator<br/>Analysis & Optimization]
        PT[ProgressTracker<br/>Monitoring & ETA]
        HS[Hook System<br/>Extensibility]
    end
    
    subgraph "Phase 3 Interface"
        BRI[BuildRunnerInterface<br/>Execution Abstraction]
    end
    
    subgraph "Phase 3: Build Runner"
        BR[Build Runner<br/>Actual Execution]
    end
    
    subgraph "Outputs"
        ER[ExplorationResults<br/>Analysis & Recommendations]
    end
    
    DS --> EXP
    EXP --> CG
    CG --> EXP
    EXP --> BRI
    BRI --> BR
    BR --> BRI
    BRI --> EXP
    EXP --> AGG
    EXP --> PT
    EXP --> HS
    AGG --> ER
    
    style EXP fill:#2e7d32,color:#fff
    style ER fill:#1565c0,color:#fff
    style BRI fill:#f57c00,color:#fff
    style HS fill:#ad1457,color:#fff
```

## Component Architecture

Phase 2 consists of several key components working together to systematically explore the design space.

```mermaid
graph TB
    subgraph "Phase 2 Core Components"
        subgraph "Orchestration Layer"
            ENG[ExplorerEngine<br/>explorer.py]
        end
        
        subgraph "Generation Layer"
            COMBOGEN[CombinationGenerator<br/>combination_generator.py]
        end
        
        subgraph "Analysis Layer"
            AGG[ResultsAggregator<br/>results_aggregator.py]
        end
        
        subgraph "Monitoring Layer"
            PROG[ProgressTracker<br/>progress.py]
        end
        
        subgraph "Extensibility Layer"
            HOOKS[Hook System<br/>hooks.py]
            LOG[LoggingHook]
            CACHE[CachingHook]
        end
        
        subgraph "Data Layer"
            DS2[Data Structures<br/>data_structures.py]
            IFACE[Interfaces<br/>interfaces.py]
        end
        
        subgraph "External Integration"
            P1[Phase 1 Data<br/>../phase1/data_structures.py]
            P3[Phase 3 Interface<br/>BuildRunnerInterface]
        end
    end
    
    ENG --> COMBOGEN
    ENG --> AGG
    ENG --> PROG
    ENG --> HOOKS
    ENG --> P3
    COMBOGEN --> DS2
    AGG --> DS2
    HOOKS --> LOG
    HOOKS --> CACHE
    IFACE --> P3
    DS2 --> P1
    
    style ENG fill:#2e7d32,color:#fff
    style AGG fill:#0277bd,color:#fff
    style COMBOGEN fill:#0277bd,color:#fff
    style HOOKS fill:#f57c00,color:#fff
    style P3 fill:#ad1457,color:#fff
```

## Data Flow

The data flow through Phase 2 shows how design spaces are systematically explored and results aggregated.

```mermaid
flowchart LR
    subgraph "Input"
        DSPACE[DesignSpace<br/>from Phase 1]
    end
    
    subgraph "Generation"
        COMBINATIONS[Generate<br/>Combinations<br/>Cartesian Product]
        FILTER[Apply<br/>Constraints<br/>Filter Invalid]
        CONFIGS[BuildConfig<br/>Objects]
    end
    
    subgraph "Execution Loop"
        SCHEDULE[Schedule<br/>Builds]
        EXECUTE[Execute via<br/>BuildRunner]
        COLLECT[Collect<br/>Results]
    end
    
    subgraph "Monitoring"
        PROGRESS[Update<br/>Progress]
        HOOKS[Fire Hook<br/>Events]
        LOGGING[Log<br/>Progress]
    end
    
    subgraph "Analysis"
        AGGREGATE[Aggregate<br/>Results]
        PARETO[Find Pareto<br/>Optimal]
        BEST[Identify<br/>Best Config]
    end
    
    subgraph "Output"
        RESULTS[ExplorationResults<br/>Complete Analysis]
    end
    
    DSPACE --> COMBINATIONS
    COMBINATIONS --> FILTER
    FILTER --> CONFIGS
    CONFIGS --> SCHEDULE
    SCHEDULE --> EXECUTE
    EXECUTE --> COLLECT
    COLLECT --> PROGRESS
    COLLECT --> HOOKS
    COLLECT --> AGGREGATE
    PROGRESS --> LOGGING
    AGGREGATE --> PARETO
    AGGREGATE --> BEST
    PARETO --> RESULTS
    BEST --> RESULTS
    
    COLLECT -.-> SCHEDULE
```

## Class Relationships

The core data structures and their relationships define the exploration process and results.

```mermaid
classDiagram
    class ExplorerEngine {
        +build_runner_factory: Callable
        +hooks: List[ExplorationHook]
        +progress_tracker: ProgressTracker
        +exploration_results: ExplorationResults
        +explore(design_space, resume_from) ExplorationResults
        +_evaluate_config(config, build_runner) BuildResult
        +_should_stop_early(design_space, index) bool
        +_fire_hook(method_name, *args) void
    }
    
    class CombinationGenerator {
        +generate_all(design_space) List[BuildConfig]
        +filter_by_indices(configs, indices) List[BuildConfig]
        +filter_by_resume(configs, last_id) List[BuildConfig]
        +_generate_design_space_id(design_space) str
        +_satisfies_constraints(config, constraints) bool
    }
    
    class BuildConfig {
        +id: str
        +design_space_id: str
        +model_path: str
        +kernels: List[Tuple[str, List[str]]]
        +transforms_by_stage: Dict[str, List[str]]
        +build_steps: List[str]
        +config_flags: Dict[str, Any]
        +global_config: GlobalConfig
        +output_dir: str
        +combination_index: int
        +total_combinations: int
        +to_dict() Dict[str, Any]
    }
    
    class BuildResult {
        +config_id: str
        +status: BuildStatus
        +metrics: BuildMetrics
        +start_time: datetime
        +end_time: datetime
        +duration_seconds: float
        +artifacts: Dict[str, str]
        +logs: Dict[str, str]
        +error_message: str
        +complete(status, error_message) void
    }
    
    class ExplorationResults {
        +design_space_id: str
        +start_time: datetime
        +end_time: datetime
        +evaluations: List[BuildResult]
        +total_combinations: int
        +evaluated_count: int
        +success_count: int
        +failure_count: int
        +best_config: BuildConfig
        +pareto_optimal: List[BuildConfig]
        +metrics_summary: Dict[str, Dict[str, float]]
        +add_config(config) void
        +get_successful_results() List[BuildResult]
        +get_summary_string() str
    }
    
    class ResultsAggregator {
        +results: ExplorationResults
        +add_result(result) void
        +finalize() void
        +_find_best_config() BuildConfig
        +_find_pareto_optimal() List[BuildConfig]
        +_calculate_metrics_summary() Dict
        +get_top_n_configs(n, metric) List[Tuple]
    }
    
    class ProgressTracker {
        +total_configs: int
        +completed: int
        +successful: int
        +failed: int
        +total_build_time: float
        +update(result) void
        +get_eta() datetime
        +get_summary() str
        +get_progress_bar(width) str
    }
    
    class BuildRunnerInterface {
        <<interface>>
        +run(config) BuildResult
    }
    
    class ExplorationHook {
        <<abstract>>
        +on_exploration_start(design_space, results) void
        +on_combinations_generated(configs) void
        +on_build_complete(config, result) void
        +on_exploration_complete(results) void
    }
    
    ExplorerEngine --> CombinationGenerator
    ExplorerEngine --> ResultsAggregator
    ExplorerEngine --> ProgressTracker
    ExplorerEngine --> BuildRunnerInterface
    ExplorerEngine --> ExplorationHook
    CombinationGenerator --> BuildConfig
    BuildRunnerInterface --> BuildResult
    ResultsAggregator --> ExplorationResults
    ExplorationResults --> BuildConfig
    ExplorationResults --> BuildResult
    ProgressTracker --> BuildResult
```

## Sequence Diagrams

### Main Exploration Process

The sequence of operations during design space exploration.

```mermaid
sequenceDiagram
    participant User
    participant ExplorerEngine
    participant CombinationGenerator
    participant BuildRunner
    participant ResultsAggregator
    participant ProgressTracker
    participant Hooks
    
    User->>ExplorerEngine: explore(design_space)
    ExplorerEngine->>ExplorerEngine: initialize exploration
    ExplorerEngine->>Hooks: on_exploration_start()
    
    ExplorerEngine->>CombinationGenerator: generate_all(design_space)
    CombinationGenerator->>CombinationGenerator: cartesian product
    CombinationGenerator->>CombinationGenerator: apply constraints
    CombinationGenerator-->>ExplorerEngine: List[BuildConfig]
    
    ExplorerEngine->>Hooks: on_combinations_generated()
    ExplorerEngine->>ProgressTracker: initialize(total_configs)
    
    loop For each configuration
        ExplorerEngine->>BuildRunner: run(config)
        BuildRunner-->>ExplorerEngine: BuildResult
        
        ExplorerEngine->>ResultsAggregator: add_result(result)
        ExplorerEngine->>ProgressTracker: update(result)
        ExplorerEngine->>Hooks: on_build_complete()
        
        alt Early stopping conditions
            ExplorerEngine->>ExplorerEngine: check_should_stop_early()
            break when conditions met
                ExplorerEngine->>ExplorerEngine: break loop
            end
        end
    end
    
    ExplorerEngine->>ResultsAggregator: finalize()
    ResultsAggregator->>ResultsAggregator: find best config
    ResultsAggregator->>ResultsAggregator: find pareto optimal
    ResultsAggregator->>ResultsAggregator: calculate metrics summary
    
    ExplorerEngine->>Hooks: on_exploration_complete()
    ExplorerEngine-->>User: ExplorationResults
```

### Hook System Execution

How hooks are fired during exploration events.

```mermaid
sequenceDiagram
    participant ExplorerEngine
    participant HookRegistry
    participant LoggingHook
    participant CachingHook
    participant CustomHook
    
    ExplorerEngine->>HookRegistry: _fire_hook("on_build_complete", config, result)
    
    HookRegistry->>LoggingHook: on_build_complete(config, result)
    LoggingHook->>LoggingHook: log build status with emoji
    LoggingHook->>LoggingHook: log metrics if successful
    
    HookRegistry->>CachingHook: on_build_complete(config, result)
    CachingHook->>CachingHook: serialize result to JSON
    CachingHook->>CachingHook: append to cache file
    
    HookRegistry->>CustomHook: on_build_complete(config, result)
    CustomHook->>CustomHook: custom processing
    
    Note over HookRegistry: Continue even if individual hooks fail
    
    HookRegistry-->>ExplorerEngine: all hooks completed
```

## Hook System

The extensible hook system allows injection of custom behavior at key exploration points.

```mermaid
graph TB
    subgraph "Hook Events"
        ES[on_exploration_start<br/>Setup phase]
        CG[on_combinations_generated<br/>After config generation]
        BC[on_build_complete<br/>After each build]
        EC[on_exploration_complete<br/>Cleanup phase]
    end
    
    subgraph "Built-in Hooks"
        LOG[LoggingHook<br/>Detailed progress logging]
        CACHE[CachingHook<br/>Result caching & resume]
    end
    
    subgraph "Custom Hooks"
        EARLY[EarlyStoppingHook<br/>Stop on criteria]
        SAMPLE[SamplingHook<br/>Filter combinations]
        ML[MLGuidedHook<br/>ML-based guidance]
        NOTIF[NotificationHook<br/>Status notifications]
    end
    
    subgraph "Hook Registry"
        REG[HookRegistry<br/>Manages all hooks]
    end
    
    ES --> LOG
    ES --> CACHE
    CG --> LOG
    CG --> SAMPLE
    BC --> LOG
    BC --> CACHE
    BC --> EARLY
    BC --> ML
    BC --> NOTIF
    EC --> LOG
    EC --> CACHE
    
    LOG --> REG
    CACHE --> REG
    EARLY --> REG
    SAMPLE --> REG
    ML --> REG
    NOTIF --> REG
    
    style REG fill:#f57c00,color:#fff
    style LOG fill:#2e7d32,color:#fff
    style CACHE fill:#2e7d32,color:#fff
```

## Progress Tracking

Comprehensive progress monitoring with ETA calculation and detailed statistics.

```mermaid
graph TB
    subgraph "Progress Metrics"
        COMP[Completed Builds]
        SUCC[Successful Builds]
        FAIL[Failed Builds]
        SKIP[Skipped Builds]
        TIME[Build Timing Stats]
    end
    
    subgraph "Progress Calculations"
        PCT[Progress Percentage]
        RATE[Success Rate]
        AVG[Average Build Time]
        ETA[Estimated Time of Arrival]
        SPEED[Builds per Minute]
    end
    
    subgraph "Progress Outputs"
        BAR[Text Progress Bar]
        SUMMARY[Progress Summary]
        DETAILED[Detailed Statistics]
        LOGS[Progress Logging]
    end
    
    COMP --> PCT
    SUCC --> RATE
    FAIL --> RATE
    TIME --> AVG
    TIME --> ETA
    COMP --> SPEED
    
    PCT --> BAR
    RATE --> SUMMARY
    AVG --> SUMMARY
    ETA --> SUMMARY
    SPEED --> DETAILED
    
    BAR --> LOGS
    SUMMARY --> LOGS
    DETAILED --> LOGS
    
    style PCT fill:#2e7d32,color:#fff
    style ETA fill:#1565c0,color:#fff
    style LOGS fill:#f57c00,color:#fff
```

## Results Analysis

Comprehensive analysis of exploration results with optimization identification.

```mermaid
graph TB
    subgraph "Raw Results"
        BR[Build Results<br/>All evaluations]
        SUCC[Successful Results<br/>Status == SUCCESS]
        METRICS[Build Metrics<br/>Performance data]
    end
    
    subgraph "Analysis Methods"
        BEST[Best Configuration<br/>Highest throughput]
        PARETO[Pareto Optimal<br/>Throughput vs Resources]
        TOPN[Top N Configs<br/>By any metric]
        STATS[Metrics Statistics<br/>Min/Max/Mean/Std]
    end
    
    subgraph "Analysis Outputs"
        RECOMMENDATIONS[Recommendations<br/>Best configurations]
        SUMMARY[Summary Statistics<br/>Success rates, timing]
        FAILURES[Failure Analysis<br/>Error categorization]
        FRONTIER[Pareto Frontier<br/>Trade-off analysis]
    end
    
    BR --> SUCC
    SUCC --> METRICS
    METRICS --> BEST
    METRICS --> PARETO
    METRICS --> TOPN
    METRICS --> STATS
    
    BEST --> RECOMMENDATIONS
    PARETO --> FRONTIER
    TOPN --> RECOMMENDATIONS
    STATS --> SUMMARY
    BR --> FAILURES
    
    style RECOMMENDATIONS fill:#2e7d32,color:#fff
    style FRONTIER fill:#1565c0,color:#fff
    style SUMMARY fill:#f57c00,color:#fff
```

## Key Design Decisions

### 1. Exhaustive Strategy First
- Implementation focuses on exhaustive exploration initially
- Hook system designed to support future intelligent strategies
- Cartesian product generation with constraint filtering

### 2. Clean Phase 3 Interface
- BuildRunnerInterface abstracts execution details
- Phase 2 only coordinates, doesn't execute builds
- BuildConfig contains all information needed for execution

### 3. Comprehensive Results Analysis
- Multi-objective optimization with Pareto frontier
- Detailed metrics summarization and statistics
- Failure analysis and categorization

### 4. Extensible Hook System
- Clean separation of concerns through hooks
- Built-in hooks for common needs (logging, caching)
- Abstract base class enables custom extensions

### 5. Resume Capability
- Unique design space IDs for consistent identification
- JSONL cache format for incremental result storage
- Filter configurations for resuming from specific points

### 6. Progress Monitoring
- Real-time progress tracking with ETA calculation
- Multiple progress reporting formats (summary, detailed, bar)
- Performance metrics (builds per minute, success rates)

## Usage Examples

### Basic Exploration

```python
from brainsmith.core.phase2 import explore, MockBuildRunner
from brainsmith.core.phase1 import forge

# Create design space
design_space = forge("model.onnx", "blueprint.yaml")

# Simple exploration with mock runner
results = explore(
    design_space=design_space,
    build_runner_factory=lambda: MockBuildRunner(success_rate=0.8)
)

print(f"Best configuration: {results.best_config.id}")
print(f"Pareto optimal: {len(results.pareto_optimal)} configs")
```

### Advanced Exploration with Hooks

```python
from brainsmith.core.phase2 import (
    ExplorerEngine, 
    LoggingHook, 
    CachingHook
)

# Create explorer with hooks
hooks = [
    LoggingHook(log_level="INFO", log_file="exploration.log"),
    CachingHook(cache_dir=".cache")
]

explorer = ExplorerEngine(
    build_runner_factory=create_finn_runner,
    hooks=hooks
)

# Explore with resume capability
results = explorer.explore(
    design_space=design_space,
    resume_from="dse_abc12345_config_00050"  # Resume from specific config
)
```

### Custom Hook Implementation

```python
from brainsmith.core.phase2 import ExplorationHook

class EarlyStoppingHook(ExplorationHook):
    def __init__(self, min_throughput=1000.0, max_failures=50):
        self.min_throughput = min_throughput
        self.max_failures = max_failures
        self.failure_count = 0
        self.found_good_config = False
    
    def on_exploration_start(self, design_space, exploration_results):
        logger.info(f"Early stopping: min_throughput={self.min_throughput}")
    
    def on_combinations_generated(self, configs):
        pass
    
    def on_build_complete(self, config, result):
        if result.status == BuildStatus.FAILED:
            self.failure_count += 1
        elif result.metrics and result.metrics.throughput >= self.min_throughput:
            self.found_good_config = True
    
    def on_exploration_complete(self, exploration_results):
        logger.info(f"Early stopping stats: {self.failure_count} failures")
```

## Integration Points

### Phase 1 Integration
- Receives validated DesignSpace objects
- Uses Phase 1 data structures (GlobalConfig, BuildMetrics)
- Leverages Phase 1 combination generation methods

### Phase 3 Integration
- BuildRunnerInterface defines clean contract
- Passes self-contained BuildConfig to Phase 3
- Receives BuildResult with metrics and artifacts

### Plugin System Integration
- No direct plugin system dependencies
- Plugin validation handled in Phase 1
- Focus on exploration orchestration only

## Performance Characteristics

### Time Complexity
- **Combination Generation**: O(k × t) where k=kernels, t=transforms
- **Build Execution**: O(n × b) where n=configurations, b=build time
- **Results Analysis**: O(n log n) for Pareto frontier calculation
- **Progress Tracking**: O(1) per update

### Space Complexity
- **Configuration Storage**: O(n) where n=total configurations
- **Results Storage**: O(n) for all build results
- **Cache Storage**: O(n) for persistent caching

### Scalability
- Supports large design spaces through constraint filtering
- Resume capability for long-running explorations
- Hook system scales with exploration complexity
- Memory usage linear with result count

## Future Enhancements

### Planned Features
1. **Intelligent Sampling** - Hook to reduce evaluation count
2. **Parallel Execution** - Multiple concurrent builds
3. **Adaptive Strategies** - ML-guided exploration
4. **Real-time Visualization** - Live progress dashboards
5. **Distributed Execution** - Multi-node exploration

### Extension Points
1. **Search Strategies** - Beyond exhaustive exploration
2. **Constraint Solvers** - Advanced constraint satisfaction
3. **Result Persistence** - Database integration
4. **Notification Systems** - Real-time status updates
5. **Custom Metrics** - Domain-specific optimization criteria

## Summary

Phase 2 Design Space Explorer provides a robust, extensible framework for systematic design space exploration. Key strengths include:

- **Clean Architecture** - Clear separation between exploration coordination and build execution
- **Extensible Design** - Hook system enables custom behavior without core modifications
- **Comprehensive Analysis** - Multi-objective optimization with Pareto frontier analysis
- **Production Ready** - Resume capability, progress tracking, and robust error handling
- **Performance Focused** - Efficient algorithms and minimal memory overhead

The architecture successfully bridges Phase 1 (design space definition) and Phase 3 (build execution) while providing valuable exploration coordination, result analysis, and extensibility for future enhancements.