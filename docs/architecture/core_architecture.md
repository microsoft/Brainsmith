# BrainSmith Core Architecture

This document provides a detailed analysis of the BrainSmith core module architecture, showing how classes and parameters flow through the system.

## Overview

BrainSmith is a DSE (Design Space Exploration) framework for FPGA accelerator design. The core module implements a segment-based execution tree architecture that efficiently explores design spaces by sharing computation between similar design paths.

## Architecture Components

### 1. Entry Point: Forge API

The `forge` function is the main entry point that orchestrates the entire pipeline:

```mermaid
flowchart TB
    Start([User]) --> forge["forge(model_path, blueprint_path)"]
    forge --> BP[BlueprintParser]
    BP --> DS[DesignSpace]
    BP --> ET[ExecutionTree]
    forge --> Return["(DesignSpace, ExecutionTree)"]
```

### 2. Blueprint Parsing Flow

The BlueprintParser handles YAML parsing, inheritance, and plugin resolution:

```mermaid
flowchart TD
    YAML[Blueprint YAML] --> Parser[BlueprintParser.parse]
    
    subgraph "Parsing Steps"
        Parser --> Load[Load with Inheritance]
        Load --> Extract[Extract Config & Mappings]
        Extract --> ParseSteps[Parse Steps]
        ParseSteps --> ParseKernels[Parse Kernels]
        ParseKernels --> Validate[Validate Config]
        Validate --> BuildDS[Build DesignSpace]
        BuildDS --> BuildTree[Build ExecutionTree]
    end
    
    BuildTree --> Output["(DesignSpace, ExecutionNode)"]
    
    subgraph "Step Operations"
        ParseSteps --> StepOps[StepOperation]
        StepOps --> |after| After[Insert After]
        StepOps --> |before| Before[Insert Before]
        StepOps --> |replace| Replace[Replace Step]
        StepOps --> |remove| Remove[Remove Step]
        StepOps --> |at_start| AtStart[Insert at Start]
        StepOps --> |at_end| AtEnd[Insert at End]
    end
```

### 3. Core Data Structures

```mermaid
classDiagram
    class DesignSpace {
        +model_path: str
        +steps: List[Union[str, List[str]]]
        +kernel_backends: List[Tuple[str, List[Type]]]
        +global_config: GlobalConfig
        +finn_config: Dict[str, Any]
        +validate_size()
        +get_kernel_summary()
    }
    
    class GlobalConfig {
        +output_stage: OutputStage
        +working_directory: str
        +save_intermediate_models: bool
        +fail_fast: bool
        +max_combinations: int
        +timeout_minutes: int
    }
    
    class OutputStage {
        <<enumeration>>
        COMPILE_AND_PACKAGE
        SYNTHESIZE_BITSTREAM
        GENERATE_REPORTS
    }
    
    class ExecutionNode {
        +segment_steps: List[Dict[str, Any]]
        +branch_decision: Optional[str]
        +parent: Optional[ExecutionNode]
        +children: Dict[str, ExecutionNode]
        +status: str
        +output_dir: Optional[Path]
        +finn_config: Dict[str, Any]
        +segment_id: str
        +add_child()
        +get_path()
        +get_all_steps()
    }
    
    class BlueprintParser {
        +parse(blueprint_path, model_path)
        -_load_with_inheritance()
        -_parse_steps()
        -_parse_kernels()
        -_build_execution_tree()
        -_apply_step_operation()
    }
    
    DesignSpace --> GlobalConfig
    GlobalConfig --> OutputStage
    BlueprintParser --> DesignSpace : creates
    BlueprintParser --> ExecutionNode : creates
    ExecutionNode --> ExecutionNode : parent/children
```

### 4. Execution Tree Structure

The execution tree represents the design space as a tree of segments:

```mermaid
flowchart TD
    Root[Root Node<br/>segment_steps: empty] --> B1[Branch 1<br/>segment_steps: step1, step2]
    Root --> B2[Branch 2<br/>segment_steps: step1, step3]
    
    B1 --> B1C1[Child 1<br/>segment_steps: step4]
    B1 --> B1C2[Child 2<br/>segment_steps: step5]
    
    B2 --> B2C1[Child 1<br/>segment_steps: step4]
    B2 --> B2C2[Child 2<br/>segment_steps: skip]
    
    style Root fill:#f9f,stroke:#333,stroke-width:4px
    style B1C2 fill:#9f9,stroke:#333,stroke-width:2px
    style B2C2 fill:#ff9,stroke:#333,stroke-width:2px
```

### 5. Explorer Architecture

The explorer module executes the tree using FINN:

```mermaid
flowchart LR
    subgraph Explorer
        direction TB
        ExploreFunc[explore_execution_tree] --> Executor
        Executor --> FINNAdapter
        Executor --> TreeWalk[Tree Traversal]
        TreeWalk --> SegmentExec[Segment Execution]
    end
    
    subgraph "Execution Flow"
        direction TB
        SegmentExec --> PrepareModel[Prepare Model]
        PrepareModel --> BuildConfig[Build FINN Config]
        BuildConfig --> FINNBuild[FINN Build]
        FINNBuild --> SaveResult[Save Result]
        SaveResult --> ShareArtifacts[Share Artifacts]
    end
    
    FINNAdapter --> FINNBuild
```

### 6. Explorer Classes Detail

```mermaid
classDiagram
    class Executor {
        +finn_adapter: FINNAdapter
        +base_finn_config: Dict
        +global_config: Dict
        +execute(root, model, output_dir)
        -_execute_segment()
        -_make_finn_config()
        -_mark_descendants_skipped()
    }
    
    class FINNAdapter {
        +build(input_model, config, output_dir)
        +prepare_model(source, destination)
        -_discover_output_model()
        -_verify_output_model()
        -_check_finn_dependencies()
    }
    
    class BuildContext {
        +finn_config: Dict[str, Any]
        +global_config: Dict[str, Any]
        +steps: List[Union[str, Callable]]
        +metadata: Dict[str, Any]
        +runtime: Dict[str, Any]
        +from_blueprint_config()
        +get_metadata()
        +set_runtime()
    }
    
    class SegmentResult {
        +success: bool
        +segment_id: str
        +output_model: Optional[Path]
        +output_dir: Optional[Path]
        +error: Optional[str]
        +execution_time: float
        +cached: bool
    }
    
    class TreeExecutionResult {
        +segment_results: Dict[str, SegmentResult]
        +total_time: float
        +stats: Dict[str, int]
    }
    
    Executor --> FINNAdapter : uses
    Executor --> BuildContext : creates
    Executor --> SegmentResult : produces
    Executor --> TreeExecutionResult : returns
    TreeExecutionResult --> SegmentResult : contains
```

### 7. Plugin System Architecture

The plugin system provides a registry for transforms, kernels, backends, and steps:

```mermaid
flowchart TD
    subgraph "Plugin Registry"
        Registry[Registry Singleton]
        Registry --> Transforms[Transform Plugins]
        Registry --> Kernels[Kernel Plugins]
        Registry --> Backends[Backend Plugins]
        Registry --> Steps[Step Plugins]
    end
    
    subgraph "Plugin Sources"
        BrainSmith[BrainSmith Plugins]
        QONNX[QONNX Transforms<br/>60 transforms]
        FINN[FINN Components<br/>98 transforms<br/>40 kernels<br/>30 backends]
    end
    
    subgraph "Framework Adapters"
        FA[framework_adapters.py]
        FA --> |registers| QONNX
        FA --> |registers| FINN
    end
    
    BrainSmith --> Registry
    FA --> Registry
    
    subgraph "Usage"
        BlueprintParser --> |get_step| Registry
        Executor --> |get_step| Registry
        Utils --> |get_transform| Registry
    end
```

### 8. Plugin Registry Detail

```mermaid
classDiagram
    class Registry {
        -_plugins: Dict[str, Dict[str, Tuple]]
        +register(type, name, cls, framework, metadata)
        +get(type, name): Type
        +find(type, criteria): List[Type]
        +all(type): Dict[str, Type]
        +reset()
        -_ensure_external_plugins()
    }
    
    class FrameworkAdapters {
        +QONNX_TRANSFORMS: List[Tuple]
        +FINN_TRANSFORMS: List[Tuple]
        +FINN_KERNELS: List[Tuple]
        +FINN_BACKENDS: List[Tuple]
        +initialize_framework_integrations()
        -_register_transforms()
        -_register_backends()
        -_register_steps()
    }
    
    Registry --> FrameworkAdapters : loads
```

## Data Flow Through the System

### 1. Blueprint to Execution Tree Flow

```mermaid
sequenceDiagram
    participant User
    participant Forge
    participant Parser as BlueprintParser
    participant Registry
    participant DesignSpace
    participant Tree as ExecutionTree
    
    User->>Forge: forge(model, blueprint)
    Forge->>Parser: parse(blueprint, model)
    
    Parser->>Parser: Load YAML with inheritance
    Parser->>Parser: Extract configurations
    
    Parser->>Registry: Validate steps
    Registry-->>Parser: Step classes
    
    Parser->>Registry: Validate kernels/backends
    Registry-->>Parser: Kernel/Backend classes
    
    Parser->>DesignSpace: Create DesignSpace
    Parser->>Tree: Build ExecutionTree
    
    Parser-->>Forge: (DesignSpace, ExecutionTree)
    Forge-->>User: (DesignSpace, ExecutionTree)
```

### 2. Tree Execution Flow

```mermaid
sequenceDiagram
    participant Explorer
    participant Executor
    participant FINNAdapter
    participant FINN
    participant FileSystem as FS
    
    Explorer->>Executor: execute(tree, model, output_dir)
    
    loop For each segment
        Executor->>Executor: Check cache
        alt Cached
            Executor->>FS: Load cached result
        else Not cached
            Executor->>FINNAdapter: prepare_model()
            FINNAdapter->>FS: Copy model
            
            Executor->>Executor: Make FINN config
            Executor->>FINNAdapter: build(model, config)
            
            FINNAdapter->>FINN: build_dataflow_cfg()
            FINN-->>FINNAdapter: Exit code
            
            FINNAdapter->>FS: Discover output model
            FINNAdapter-->>Executor: Output model path
            
            Executor->>FS: Save result
        end
        
        alt Is branch point
            Executor->>FS: Share artifacts to children
        end
    end
    
    Executor-->>Explorer: TreeExecutionResult
```

## Key Design Patterns

### 1. Segment-Based Execution
- Tree nodes represent execution segments between branch points
- Segments are executed as atomic FINN builds
- Results are cached and shared at branch points

### 2. Lazy Plugin Loading
- Plugins are registered on first access
- Framework adapters dynamically import external components
- Registry provides unified access to all plugin types

### 3. Configuration Flow
- Blueprint YAML → BlueprintParser → DesignSpace → ExecutionTree
- Global config flows through all components
- FINN config is built per-segment with inherited values

### 4. Error Handling
- Fail-fast mode stops on first error
- Non-fail-fast mode marks descendants as skipped
- ExecutionError provides context for debugging

## Performance Optimizations

1. **Prefix Sharing**: Common step sequences are executed once and shared
2. **Caching**: Completed segments are cached to disk
3. **Artifact Sharing**: Build outputs are copied to child segments at branch points
4. **Breadth-First Execution**: Enables parallel execution (future enhancement)

## Extension Points

1. **Custom Steps**: Register via `@step` decorator
2. **Custom Transforms**: Register via `@transform` decorator
3. **Custom Kernels**: Register via `@kernel` decorator with backends
4. **Custom Backends**: Register via `@backend` decorator with kernel reference

## Summary

The BrainSmith core architecture provides a clean separation of concerns:
- **Blueprint parsing** handles configuration and validation
- **Execution tree** represents the design space efficiently
- **Explorer** manages execution and result collection
- **Plugin system** provides extensibility
- **FINN adapter** isolates external dependencies

This design enables efficient design space exploration with minimal redundant computation.