# DSE V3 Combined Design Document

## Overview

The Design Space Explorer (DSE) V3 is a complete rewrite of Brainsmith's design space exploration system, focusing on simplicity, extensibility, and clear separation of concerns. This document covers the implemented Phase 1 (Design Space Constructor) and Phase 2 (Design Space Explorer).

## Architecture Overview

```mermaid
graph TB
    subgraph "Phase 1: Design Space Constructor"
        BP[Blueprint YAML]
        MODEL[ONNX Model]
        PARSER[BlueprintParser]
        VALIDATOR[DesignSpaceValidator]
        DS[DesignSpace]
        
        BP --> PARSER
        MODEL --> PARSER
        PARSER --> DS
        DS --> VALIDATOR
        VALIDATOR --> DS
    end
    
    subgraph "Phase 2: Design Space Explorer"
        DS --> COMBGEN[CombinationGenerator]
        COMBGEN --> CONFIGS[BuildConfigs]
        CONFIGS --> ENGINE[ExplorerEngine]
        ENGINE --> RUNNER[BuildRunner]
        RUNNER --> RESULTS[BuildResults]
        RESULTS --> AGGREGATOR[ResultsAggregator]
        AGGREGATOR --> EXPLORATION[ExplorationResults]
        
        HOOKS[Hooks] -.-> ENGINE
        PROGRESS[ProgressTracker] -.-> ENGINE
    end
    
    style BP fill:#e1f5fe
    style MODEL fill:#e1f5fe
    style DS fill:#fff9c4
    style EXPLORATION fill:#c8e6c9
```

## Phase 1: Design Space Constructor

### Purpose
Parse and validate Blueprint V3 YAML files to construct a well-defined design space for exploration.

### Core Components

#### 1. Data Structures

```mermaid
classDiagram
    class DesignSpace {
        +model_path: str
        +hw_compiler_space: HWCompilerSpace
        +processing_space: ProcessingSpace
        +search_config: SearchConfig
        +global_config: GlobalConfig
        +get_total_combinations(): int
    }
    
    class HWCompilerSpace {
        +kernels: List[KernelDef]
        +transforms: TransformDef
        +build_steps: List[str]
        +config_flags: Dict[str, Any]
        +get_kernel_combinations(): List
        +get_transform_combinations(): List
    }
    
    class ProcessingSpace {
        +preprocessing: List[List[ProcessingStep]]
        +postprocessing: List[List[ProcessingStep]]
        +get_preprocessing_combinations(): List
        +get_postprocessing_combinations(): List
    }
    
    class SearchConfig {
        +strategy: SearchStrategy
        +constraints: List[SearchConstraint]
        +max_evaluations: Optional[int]
        +timeout_minutes: Optional[int]
        +parallel_builds: int
    }
    
    class GlobalConfig {
        +output_stage: OutputStage
        +working_directory: str
        +cache_results: bool
        +save_artifacts: bool
        +log_level: str
    }
    
    DesignSpace --> HWCompilerSpace
    DesignSpace --> ProcessingSpace
    DesignSpace --> SearchConfig
    DesignSpace --> GlobalConfig
```

#### 2. Blueprint Parser

The parser supports flexible kernel and transform specifications:

```yaml
# Simple kernel
kernels:
  - MatMul

# Kernel with backends
kernels:
  - [MatMul, [rtl, hls]]

# Mutually exclusive kernels
kernels:
  - [LayerNorm, RMSNorm]

# Optional kernel (can be skipped)
kernels:
  - [~, Transpose]

# Flat transforms
transforms:
  - quantize
  - fold

# Phase-based transforms
transforms:
  pre_quantization:
    - fold
    - ~streamline  # optional
  quantization:
    - quantize
  post_quantization:
    - pack
```

#### 3. Validation Flow

```mermaid
sequenceDiagram
    participant User
    participant ForgeAPI
    participant Parser
    participant Validator
    participant DesignSpace
    
    User->>ForgeAPI: forge(model_path, blueprint_path)
    ForgeAPI->>Parser: parse(blueprint_yaml)
    Parser->>DesignSpace: create initial structure
    ForgeAPI->>Validator: validate(design_space, model_path)
    Validator->>Validator: check_model_exists()
    Validator->>Validator: validate_hw_compiler_space()
    Validator->>Validator: validate_processing_space()
    Validator->>Validator: validate_search_config()
    Validator->>Validator: validate_global_config()
    alt Validation Failed
        Validator-->>ForgeAPI: ValidationError
        ForgeAPI-->>User: raise ValidationError
    else Validation Passed
        Validator-->>ForgeAPI: ValidationResult(warnings)
        ForgeAPI-->>User: DesignSpace
    end
```

### Phase 1 API

```python
from brainsmith.core_v3 import forge

# Main API function
design_space = forge(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml"
)

# Returns a validated DesignSpace ready for exploration
print(f"Total combinations: {design_space.get_total_combinations()}")
```

## Phase 2: Design Space Explorer

### Purpose
Systematically explore the design space by generating all valid configurations, executing builds, and analyzing results.

### Core Components

#### 1. Data Structures

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
        +complete(status, error_msg)
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
        +skipped_count: int
        +best_config: Optional[BuildConfig]
        +pareto_optimal: List[BuildConfig]
        +metrics_summary: Dict
        +get_successful_results()
        +get_failed_results()
        +update_counts()
    }
    
    class BuildMetrics {
        +throughput: float
        +latency: float
        +clock_frequency: float
        +lut_utilization: float
        +dsp_utilization: float
        +bram_utilization: float
        +total_power: float
        +accuracy: float
    }
    
    BuildResult --> BuildMetrics
    ExplorationResults --> BuildResult
    ExplorationResults --> BuildConfig
```

#### 2. Exploration Flow

```mermaid
sequenceDiagram
    participant User
    participant Explorer
    participant CombGen as CombinationGenerator
    participant Engine as ExplorerEngine
    participant Runner as BuildRunner
    participant Aggregator as ResultsAggregator
    participant Hooks
    
    User->>Explorer: explore(design_space, build_runner_factory)
    Explorer->>Engine: __init__(build_runner_factory, hooks)
    Explorer->>Engine: explore(design_space)
    
    Engine->>Hooks: on_exploration_start()
    Engine->>CombGen: generate_all(design_space)
    CombGen-->>Engine: List[BuildConfig]
    Engine->>Hooks: on_combinations_generated(configs)
    
    loop For each BuildConfig
        Engine->>Runner: run(config)
        Runner-->>Engine: BuildResult
        Engine->>Aggregator: add_result(result)
        Engine->>Hooks: on_build_complete(config, result)
        
        alt Early stopping check
            Engine->>Engine: should_stop_early()
        end
    end
    
    Engine->>Aggregator: finalize()
    Aggregator-->>Engine: ExplorationResults
    Engine->>Hooks: on_exploration_complete(results)
    Engine-->>User: ExplorationResults
```

#### 3. Combination Generation

The combination generator creates the Cartesian product of all options:

```mermaid
graph LR
    subgraph "Kernel Space"
        K1[MatMul with rtl]
        K2[MatMul with hls]
    end
    
    subgraph "Transform Space"
        T1[quantize + fold]
        T2[quantize + streamline]
    end
    
    subgraph "Processing Space"
        P1[resize to 224]
        P2[resize to 256]
    end
    
    subgraph "Combinations"
        C1[K1 + T1 + P1]
        C2[K1 + T1 + P2]
        C3[K1 + T2 + P1]
        C4[K1 + T2 + P2]
        C5[K2 + T1 + P1]
        C6[K2 + T1 + P2]
        C7[K2 + T2 + P1]
        C8[K2 + T2 + P2]
    end
    
    K1 --> C1
    K1 --> C2
    K1 --> C3
    K1 --> C4
    K2 --> C5
    K2 --> C6
    K2 --> C7
    K2 --> C8
```

#### 4. Hook System

```mermaid
classDiagram
    class ExplorationHook {
        <<abstract>>
        +on_exploration_start(design_space, results)
        +on_combinations_generated(configs)
        +on_build_complete(config, result)
        +on_exploration_complete(results)
    }
    
    class LoggingHook {
        +log_level: str
        +log_file: Optional[str]
        +on_exploration_start()
        +on_combinations_generated()
        +on_build_complete()
        +on_exploration_complete()
    }
    
    class CachingHook {
        +cache_dir: str
        +on_exploration_start()
        +on_build_complete()
        +on_exploration_complete()
        -load_cached_results()
    }
    
    class CustomHook {
        <<user defined>>
        +custom_behavior()
    }
    
    ExplorationHook <|-- LoggingHook
    ExplorationHook <|-- CachingHook
    ExplorationHook <|-- CustomHook
```

#### 5. Progress Tracking

```mermaid
graph LR
    subgraph "Progress Tracker"
        TOTAL[Total: 100]
        COMPLETE[Completed: 45]
        SUCCESS[Success: 40]
        FAILED[Failed: 5]
        
        RATE[Rate: 2.5/min]
        ETA[ETA: 22 min]
        
        BAR["[████████████░░░░░░░░] 45%"]
    end
    
    COMPLETE --> RATE
    RATE --> ETA
    COMPLETE --> BAR
```

### Phase 2 API

```python
from brainsmith.core_v3 import forge, explore
from brainsmith.core_v3.phase2 import MockBuildRunner, LoggingHook, CachingHook

# Get design space from Phase 1
design_space = forge("model.onnx", "blueprint.yaml")

# Create build runner factory
def build_runner_factory():
    return MockBuildRunner(success_rate=0.9)

# Set up hooks
hooks = [
    LoggingHook(log_level="INFO", log_file="exploration.log"),
    CachingHook(cache_dir=".cache")
]

# Run exploration
results = explore(
    design_space,
    build_runner_factory,
    hooks=hooks,
    resume_from=None  # Or provide config_id to resume
)

# Analyze results
print(results.get_summary_string())
print(f"Best config: {results.best_config}")
print(f"Pareto optimal: {len(results.pareto_optimal)} configs")
```

## Global Configuration System

DSE V3 includes a hierarchical global configuration system that allows users to set defaults and limits at multiple levels:

### Configuration Priority (Highest to Lowest)
1. **Blueprint/project** configuration (in blueprint YAML)
2. **User** configuration (`~/.brainsmith/config.yaml`)
3. **Environment** variables (`BRAINSMITH_MAX_COMBINATIONS`, `BRAINSMITH_TIMEOUT_MINUTES`)
4. **Built-in defaults** (embedded in code)

### Configuration Options

#### max_combinations
- **Purpose**: Maximum allowed design space combinations to prevent accidental creation of huge design spaces
- **Default**: 100,000 combinations
- **Blueprint override**: Set in `global.max_combinations`
- **Environment**: `BRAINSMITH_MAX_COMBINATIONS=500000`

#### timeout_minutes
- **Purpose**: Default timeout for DSE jobs
- **Default**: 60 minutes
- **Blueprint override**: Set in `search.timeout_minutes` or `global.timeout_minutes`
- **Environment**: `BRAINSMITH_TIMEOUT_MINUTES=120`

### Usage Examples

#### Blueprint Configuration
```yaml
version: "3.0"
global:
  max_combinations: 500000  # Override global default
  timeout_minutes: 120      # Override global default
search:
  timeout_minutes: 90       # Takes precedence over global.timeout_minutes
```

#### User Configuration (`~/.brainsmith/config.yaml`)
```yaml
# Global defaults for this user
max_combinations: 1000000
timeout_minutes: 240
```

#### Environment Variables
```bash
export BRAINSMITH_MAX_COMBINATIONS=2000000
export BRAINSMITH_TIMEOUT_MINUTES=480
```

### Validation and Error Handling

When a design space exceeds the max_combinations limit, validation fails with a helpful error message:

```
Design space has 150,000 combinations, exceeding maximum of 100,000. 
You can increase this limit by setting max_combinations in the blueprint's 
global section, or in ~/.brainsmith/config.yaml, or via 
BRAINSMITH_MAX_COMBINATIONS environment variable.
```

## Key Design Decisions

### 1. Clean Phase Separation
- Phase 1: Blueprint parsing and validation
- Phase 2: Design space exploration
- Phase 3: Build execution (future)

### 2. Explicit Configuration
- All options explicitly defined in blueprints
- No hidden defaults or magic behavior
- Clear validation with helpful error messages

### 3. Flexible Kernel/Transform Specification
- Support for simple lists, backends, mutually exclusive options
- Optional elements with `~` syntax
- Phase-based transform organization

### 4. Hook-Based Extensibility
- Clean interface for extending behavior
- Built-in hooks for common needs (logging, caching)
- Hooks fired at key lifecycle points

### 5. Hierarchical Configuration
- Global library defaults with user and project overrides
- Environment variable support for CI/CD environments
- Clear priority order from specific to general

### 6. Comprehensive Testing
- 110+ tests covering all functionality
- Unit tests for components
- Integration tests for workflows
- Test-driven development approach

## Directory Structure

```
brainsmith/core_v3/
├── __init__.py           # Public API (forge, explore)
├── phase1/               # Design Space Constructor
│   ├── __init__.py
│   ├── data_structures.py
│   ├── exceptions.py
│   ├── parser.py
│   ├── validator.py
│   └── forge.py
├── phase2/               # Design Space Explorer
│   ├── __init__.py
│   ├── data_structures.py
│   ├── combination_generator.py
│   ├── explorer.py
│   ├── results_aggregator.py
│   ├── hooks.py
│   ├── progress.py
│   └── interfaces.py
└── tests/
    ├── fixtures/
    ├── unit/
    │   ├── phase1/
    │   └── phase2/
    └── integration/
```

## Usage Example

```python
# Complete workflow
from brainsmith.core_v3 import forge, explore
from my_build_runner import RealBuildRunner

# 1. Parse blueprint
design_space = forge("bert.onnx", "bert_blueprint.yaml")
print(f"Design space has {design_space.get_total_combinations()} configurations")

# 2. Explore design space
results = explore(design_space, lambda: RealBuildRunner())

# 3. Analyze results
print(f"Explored {results.evaluated_count} configurations")
print(f"Success rate: {results.success_count / results.evaluated_count * 100:.1f}%")
print(f"Best throughput: {results.best_config} ")

# 4. Get Pareto optimal configurations
for config in results.pareto_optimal:
    metrics = results.get_result(config.id).metrics
    print(f"Config {config.id}: {metrics.throughput} GOPS, {metrics.lut_utilization:.1%} LUT")
```

## Future Enhancements

### Phase 3: Build Runner
- Real build execution with preprocessing/postprocessing
- Integration with backend compilers (FINN, Vitis)
- Artifact management and caching

### Advanced Features
- Parallel build execution
- Smart search strategies (genetic algorithms, Bayesian optimization)
- Constraint-based pruning
- Multi-objective optimization enhancements

## Conclusion

DSE V3 provides a clean, extensible foundation for design space exploration in Brainsmith. The architecture emphasizes simplicity, testability, and clear separation of concerns while providing the flexibility needed for future enhancements.