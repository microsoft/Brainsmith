# Brainsmith Core: DSE v3 Architecture

## Overview

Brainsmith DSE v3 is a three-phase design space exploration system for FPGA AI accelerator development. It systematically explores hardware implementation options for neural networks, evaluating different algorithms, parallelization strategies, and resource allocations to find optimal configurations.

## Table of Contents

1. [System Architecture](#system-architecture)
2. [Three-Phase Pipeline](#three-phase-pipeline)
3. [Data Flow](#data-flow)
4. [Plugin System](#plugin-system)
5. [Backend System](#backend-system)
6. [Configuration](#configuration)
7. [API Reference](#api-reference)
8. [Error Handling](#error-handling)
9. [Performance](#performance)
10. [Extension Guide](#extension-guide)

## System Architecture


---

```mermaid
%%{ init: { "flowchart": { "rankSpacing": 40, "nodeSpacing": 24 } } }%%
flowchart TB
    %%──────────────────────── INPUT ────────────────────────
    subgraph INPUT["Input"]
        direction TB
        MODEL[[ONNX Model]]
        BLUEPRINT[[Blueprint YAML]]
        CONFIG[[Environment Config]]
    end

    %%─────────────── PHASE 1 · DESIGN-SPACE CONSTRUCTOR ────────────
    subgraph P1["Phase 1 · Constructor"]
        direction TB
        FORGE[Forge API]
        PARSER[Blueprint Parser]
        VALIDATOR[Schema Validator]
        SPACE[Design Space]

        %% intra-phase flow
        FORGE --> PARSER --> VALIDATOR --> SPACE
    end

    %%──────────────── PHASE 2 · DESIGN-SPACE EXPLORER ──────────────
    subgraph P2["Phase 2 · Explorer"]
        direction TB
        EXPLORER[Explorer Engine]
        GENERATOR[Combination Generator]
        AGGREGATOR[Results Aggregator]

        EXPLORER --> GENERATOR --> AGGREGATOR
    end

    %%──────────────────── PHASE 3 · BUILD RUNNER ───────────────────
    subgraph P3["Phase 3 · Runner"]
        direction TB
        RUNNER[Build Runner]
        FACTORY[Backend Factory]
        METRICS[Metrics Collector]

        RUNNER --> FACTORY --> METRICS
    end

    %%──────────────────────── PLUG-INS ────────────────────────
    subgraph PLUGS["Plugin Registry"]
        direction TB
        REGISTRY[Registry]
        PLUGINS[Transforms / Kernels / Backends]

        REGISTRY --> PLUGINS
    end

    %%──────────────────────── OUTPUT ────────────────────────
    subgraph OUTPUT["Output"]
        direction TB
        RESULTS[[Exploration Results]]
        ARTIFACTS[[Build Artifacts]]
    end

    %%────────────── cross-phase conduits ──────────────
    MODEL --> FORGE
    CONFIG --> FORGE
    BLUEPRINT --> PARSER
    SPACE --> EXPLORER
    GENERATOR --> RUNNER
    RUNNER --> RESULTS
    RUNNER --> ARTIFACTS

    %%────────────── plugin integration points ──────────────
    FORGE -.-> REGISTRY
    RUNNER -.-> REGISTRY

    %%──────────────── aesthetic classes ────────────────
    classDef core fill:#1f2937,stroke:#334155,color:#ffffff;
    classDef accent1 fill:#7c3aed,stroke:#6d28d9,color:#ffffff;
    classDef accent2 fill:#0891b2,stroke:#0e7490,color:#ffffff;
    classDef accent3 fill:#059669,stroke:#047857,color:#ffffff;

    class FORGE,PARSER,VALIDATOR,SPACE core;
    class EXPLORER,GENERATOR,AGGREGATOR accent1;
    class RUNNER,FACTORY,METRICS accent2;
    class REGISTRY accent3;
```

### Core Concepts

- **Design Space**: Complete set of possible hardware implementations (`DesignSpace` class)
- **Blueprint**: YAML configuration defining the exploration space
- **Build Configuration**: Specific point in the design space (`BuildConfig`)

## Three-Phase Pipeline

### [Phase 1: Design Space Constructor](../../docs/PHASE2_ARCHITECTURE.md)

```mermaid
stateDiagram-v2
    direction LR
    [*] --> ParseBlueprint
    ParseBlueprint --> ValidateSchema
    ValidateSchema --> ExpandVariants
    ExpandVariants --> CreateDesignSpace
    CreateDesignSpace --> [*]
```

Parses blueprint YAML, validates constraints, expands all variant combinations, and outputs a complete design space for exploration.

### [Phase 2: Design Space Explorer](../../docs/PHASE2_ARCHITECTURE.md)

```mermaid
stateDiagram-v2
    direction LR
    [*] --> GenerateCombinations
    GenerateCombinations --> EstimateMetrics
    EstimateMetrics --> RankConfigurations
    RankConfigurations --> ExecuteBuilds
    ExecuteBuilds --> AggregateResults
    AggregateResults --> [*]
```

Generates build configurations from design space, estimates performance metrics, ranks by Pareto optimality, executes selected builds in parallel, and aggregates results.

### [Phase 3: Build Runner](../../docs/PHASE3_ARCHITECTURE.md)

```mermaid
stateDiagram-v2
    direction LR
    [*] --> SelectBackend
    SelectBackend --> Preprocessing
    Preprocessing --> ExecuteBuild
    ExecuteBuild --> Postprocessing
    Postprocessing --> CollectMetrics
    CollectMetrics --> [*]
```

Selects appropriate backend (FINN/Future/Mock), applies preprocessing transforms, executes hardware build, runs postprocessing steps, and collects standardized metrics.

## Data Structure Flow

```mermaid
classDiagram
    class DesignSpace {
        +hw_compiler_space: HWCompilerSpace
        +search_config: SearchConfig
        +global_config: GlobalConfig
    }
    
    class BuildConfig {
        +config_id: str
        +transform_sequence: List[str]
        +kernel_selections: Dict
        +backend: str
    }
    
    class BuildResult {
        +config_id: str
        +status: BuildStatus
        +metrics: BuildMetrics
        +artifacts: Dict
    }
    
    class ExplorationResults {
        +builds: List[BuildResult]
        +get_best()
        +get_pareto_frontier()
    }
    
    DesignSpace --> BuildConfig : generates
    BuildConfig --> BuildResult : produces
    BuildResult --> ExplorationResults : aggregates
```

## Plugin System

### Plugin Types

- **Transforms**: Model graph transformations (`@transform` decorator)
- **Kernels**: Hardware operator implementations (`@kernel` decorator)
- **Backends**: Code generation backends (`@backend` decorator)
- **Steps**: Pipeline processing steps (`@step` decorator)

**Detailed Plugin Documentation:** [Plugin Architecture](plugins/ARCHITECTURE.md)

### Registration Example

```python
@transform(name="CustomOpt", stage="optimization", framework="qonnx")
class CustomOptimization:
    def apply(self, model):
        return model

@kernel(name="MatMul", backends=["hls", "rtl"])
class MatMul:
    def get_nodeattr_types(self):
        return {"folding": ("i", True, 1)}
```

## Backend System

### Factory Pattern

```mermaid
flowchart TD
    CONFIG[BuildConfig] --> FACTORY{Backend Factory}
    FACTORY -->|legacy_finn| FINN[FINN Backend]
    FACTORY -->|future| FUTURE[Future Backend]
    FACTORY -->|mock| MOCK[Mock Backend]
    
    style FACTORY fill:#7c3aed,color:#fff
```

### Backend Interface

```python
class BuildRunnerInterface:
    def get_backend_name(self) -> str
    def get_supported_output_stages(self) -> List[OutputStage]
    def run(self, config: BuildConfig) -> BuildResult
```

## Configuration

### Blueprint Schema

```yaml
version: "3.0"
name: "BERT Exploration"
description: "Blueprint for exploring BERT acceleration options"

hw_compiler:
  # For future FINN-Brainsmith API (not yet implemented)
  kernels:
    # Kernel name and backend options
    - ["MatrixVectorUnit", ["hls", "rtl"]]
    - ["LayerNorm", ["hls"]]
    - ["Softmax", ["hls", "rtl"]]
    
  transforms:
    # Transform stages and options
    cleanup:
      - "RemoveIdentityOps"
      - "RemoveUnusedTensors"
    topology_opt:
      - "ExpandNorms"
      - "FoldConstants"
      
  # For legacy FINN backend only
  build_steps:
    # Sequential pipeline execution steps
    - "cleanup"
    - "brainsmith:qonnx_to_finn"
    - "streamline"
    - "create_dataflow_partition"
    - "specialize_layers"
    - "hw_codegen"
    - "finn:create_stitched_ip"
    
  config_flags:
    target_device: "xczu7ev-ffvc1156-2-e"
    target_clock_ns: 3.33
    target_fps: 1000

finn_config:
  board: "Pynq-Z1"
  clock_period_ns: 3.33
  auto_fifo_depths: true

search:
  strategy: "exhaustive"  # or "random", "genetic"
  constraints:
    max_luts: 100000
    max_brams: 500

global:
  output_stage: "stitched_ip"
  working_directory: "./dse_results"
```

### Environment Configuration

```yaml
dse:
  phase2:
    explorer:
      parallel_evaluations: 8
      timeout_per_build_s: 3600
    ranking:
      algorithm: "pareto"
      weights:
        performance: 0.4
        resources: 0.3
        power: 0.3
  
  phase3:
    backends:
      finn:
        vivado_path: "/opt/Xilinx/Vivado/2022.2"
        build_dir: "/tmp/finn_builds"

environment:
  output_dir: "./dse_results"
  log_level: "INFO"
```

## API Reference

### Core API

```python
# Simple usage
from brainsmith.core import explore
results = explore("model.onnx", "blueprint.yaml")

# Advanced usage
from brainsmith.core.phase1 import ForgeAPI
from brainsmith.core.phase2 import ExplorerEngine

forge = ForgeAPI()
design_space = forge.parse_blueprint("exploration.yaml")

explorer = ExplorerEngine(design_space, max_workers=8)
results = explorer.explore(selection_ratio=0.1)
```

### Plugin API

```python
from brainsmith.core.plugins import transforms as tfm

# Direct access
model = tfm.FoldConstants().apply(model)

# Query plugins
cleanup_transforms = tfm.find(stage="cleanup")
hls_kernels = kern.find(backend="hls")
```

### Results API

```python
# Analysis
best = results.get_best()
pareto = results.get_pareto_frontier()
df = results.to_dataframe()

# Visualization
results.plot_pareto_frontier(x="resources.lut", y="performance.fps")

# Export
results.export_configs("top_configs.json", top_k=10)
results.generate_report("report.html")
```

## Error Handling

### Error Hierarchy

```mermaid
flowchart TD
    ERROR[DSE Error] --> PHASE[Phase-Specific]
    PHASE --> P1[Parse/Validation]
    PHASE --> P2[Generation/Ranking]
    PHASE --> P3[Build/Backend]
    
    ERROR --> REC{Recoverable?}
    REC -->|Yes| RETRY[Retry with logging]
    REC -->|No| FAIL[Fail fast]
    
    style ERROR fill:#ef4444,color:#fff
```

### Recovery Strategy

- **Transient errors**: Retry with exponential backoff
- **Permanent errors**: Skip configuration, continue exploration
- **Fatal errors**: Clean shutdown with partial results

## Performance

### Complexity Analysis

| Component | Operation | Time | Space |
|-----------|-----------|------|-------|
| Plugin Registry | Lookup | O(1) | O(n) |
| Design Space | Generation | O(∏di) | O(∏di) |
| Ranking | Pareto frontier | O(n log n) | O(n) |
| Build Execution | Parallel | O(n/p) | O(p) |

### Optimizations

- Pre-computed plugin indexes for O(1) access
- Parallel build execution with thread pools
- Transform result caching
- Early pruning of invalid configurations

## Extension Guide

### Adding Custom Components

1. **Transform**: Inherit base class, add `@transform` decorator
2. **Kernel**: Inherit `HWCustomOp`, add `@kernel` decorator  
3. **Backend**: Implement `BuildRunnerInterface`, add `@backend` decorator
4. **Step**: Define logic, add `@step` decorator

### Lifecycle Hooks

```python
class CustomHooks(ExplorationHooks):
    def on_exploration_start(self, design_space):
        print(f"Starting exploration of {len(design_space)} configs")
    
    def on_build_complete(self, result):
        if result.status == BuildStatus.FAILED:
            print(f"Build failed: {result.error}")

explorer.register_hooks(CustomHooks())
```

### Integration Examples

```python
# MLflow integration
import mlflow
with mlflow.start_run():
    results = explore("model.onnx", "blueprint.yaml")
    mlflow.log_metric("best_fps", results.get_best().metrics.performance.fps)

# CI/CD testing
def test_model_meets_requirements():
    results = explore("model.onnx", "ci_blueprint.yaml")
    assert results.get_best().metrics.performance.fps >= 1000
```

---

*For implementation details, see the source code and inline documentation in the respective module files.*