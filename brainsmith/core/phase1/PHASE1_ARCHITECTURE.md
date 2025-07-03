# Phase 1: Design Space Constructor - Architecture Document

## Overview

Phase 1 of the Brainsmith DSE v3 toolchain is responsible for transforming user inputs (ONNX model + Blueprint YAML) into a validated, structured `DesignSpace` object. This document provides comprehensive architectural documentation with visual diagrams.

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Component Architecture](#component-architecture)
3. [Data Flow](#data-flow)
4. [Class Relationships](#class-relationships)
5. [Sequence Diagrams](#sequence-diagrams)
6. [Error Handling](#error-handling)
7. [Plugin Integration](#plugin-integration)
8. [Configuration Management](#configuration-management)

## High-Level Architecture

Phase 1 serves as the entry point for the DSE system, preparing the design space for exploration in Phase 2.

```mermaid
graph TB
    subgraph "User Inputs"
        ONNX[ONNX Model]
        BP[Blueprint YAML]
    end
    
    subgraph "Phase 1: Design Space Constructor"
        FORGE[ForgeAPI<br/>Main Entry Point]
        DS[DesignSpace Object<br/>Validated Output]
    end
    
    subgraph "Phase 2: Explorer"
        EXP[Design Space Explorer]
    end
    
    subgraph "Phase 3: Build Runner"
        BUILD[Build Runner]
    end
    
    ONNX --> FORGE
    BP --> FORGE
    FORGE --> DS
    DS --> EXP
    EXP --> BUILD
    
    style FORGE fill:#c8e6c9
    style DS fill:#e1f5fe
    style EXP fill:#fff9c4
    style BUILD fill:#ffccbc
```

## Component Architecture

Phase 1 consists of several key components working together to parse, validate, and construct the design space.

```mermaid
graph TB
    subgraph "Phase 1 Components"
        subgraph "API Layer"
            FORGE[ForgeAPI<br/>forge.py]
        end
        
        subgraph "Processing Layer"
            PARSER[BlueprintParser<br/>parser.py]
            VAL[DesignSpaceValidator<br/>validator.py]
        end
        
        subgraph "Data Layer"
            DS[Data Structures<br/>data_structures.py]
            EXC[Exceptions<br/>exceptions.py]
        end
        
        subgraph "External Dependencies"
            REG[Plugin Registry<br/>../plugins/registry.py]
            CFG[Global Config<br/>../config.py]
        end
    end
    
    FORGE --> PARSER
    FORGE --> VAL
    PARSER --> DS
    VAL --> DS
    PARSER -.-> REG
    VAL -.-> REG
    PARSER -.-> CFG
    VAL -.-> CFG
    FORGE --> EXC
    PARSER --> EXC
    VAL --> EXC
    
    style FORGE fill:#c8e6c9
    style PARSER fill:#b3e5fc
    style VAL fill:#b3e5fc
    style DS fill:#fff9c4
    style REG fill:#f8bbd0
    style CFG fill:#f8bbd0
```

## Data Flow

The data flow through Phase 1 shows how raw inputs are transformed into validated design spaces.

```mermaid
flowchart LR
    subgraph "Inputs"
        MODEL[model.onnx]
        YAML[blueprint.yaml]
        CONFIG[Config Sources<br/>- Environment<br/>- User Config<br/>- Defaults]
    end
    
    subgraph "Parsing"
        LOAD[Load Blueprint<br/>YAML → Dict]
        PARSE[Parse Sections<br/>- hw_compiler<br/>- processing<br/>- search<br/>- global]
        VALIDATE_PLUGINS[Validate Plugins<br/>- Check existence<br/>- Auto-discover backends]
    end
    
    subgraph "Construction"
        BUILD_DS[Build DesignSpace<br/>- HWCompilerSpace<br/>- ProcessingSpace<br/>- SearchConfig<br/>- GlobalConfig]
    end
    
    subgraph "Validation"
        VAL_MODEL[Validate Model<br/>- File exists<br/>- Is ONNX]
        VAL_CONFIG[Validate Config<br/>- Constraints<br/>- Limits<br/>- Paths]
        VAL_SPACE[Validate Space<br/>- Combinations<br/>- Feasibility]
    end
    
    subgraph "Output"
        DS[DesignSpace<br/>Object]
    end
    
    MODEL --> VAL_MODEL
    YAML --> LOAD
    CONFIG --> PARSE
    LOAD --> PARSE
    PARSE --> VALIDATE_PLUGINS
    VALIDATE_PLUGINS --> BUILD_DS
    BUILD_DS --> VAL_CONFIG
    VAL_MODEL --> VAL_SPACE
    VAL_CONFIG --> VAL_SPACE
    VAL_SPACE --> DS
```

## Class Relationships

The core data structures and their relationships define the design space representation.

```mermaid
classDiagram
    class DesignSpace {
        +model_path: str
        +hw_compiler_space: HWCompilerSpace
        +processing_space: ProcessingSpace
        +search_config: SearchConfig
        +global_config: GlobalConfig
        +get_total_combinations() int
    }
    
    class HWCompilerSpace {
        +kernels: List[Union[str, tuple, list]]
        +transforms: Union[List, Dict]
        +build_steps: List[str]
        +config_flags: Dict[str, Any]
        +get_kernel_combinations() List
        +get_transform_combinations() List
        +get_transform_combinations_by_stage() List
    }
    
    class ProcessingSpace {
        +preprocessing: List[List[ProcessingStep]]
        +postprocessing: List[List[ProcessingStep]]
        +get_preprocessing_combinations() List
        +get_postprocessing_combinations() List
    }
    
    class ProcessingStep {
        +name: str
        +type: str
        +parameters: Dict[str, Any]
        +enabled: bool
    }
    
    class SearchConfig {
        +strategy: SearchStrategy
        +constraints: List[SearchConstraint]
        +max_evaluations: Optional[int]
        +timeout_minutes: Optional[int]
        +parallel_builds: int
    }
    
    class SearchConstraint {
        +metric: str
        +operator: str
        +value: Union[float, int]
        +evaluate(metric_value) bool
    }
    
    class GlobalConfig {
        +output_stage: OutputStage
        +working_directory: str
        +cache_results: bool
        +save_artifacts: bool
        +log_level: str
        +max_combinations: Optional[int]
        +timeout_minutes: Optional[int]
        +start_step: Optional[Union[str, int]]
        +stop_step: Optional[Union[str, int]]
    }
    
    class SearchStrategy {
        <<enumeration>>
        EXHAUSTIVE
    }
    
    class OutputStage {
        <<enumeration>>
        DATAFLOW_GRAPH
        RTL
        STITCHED_IP
    }
    
    DesignSpace *-- HWCompilerSpace
    DesignSpace *-- ProcessingSpace
    DesignSpace *-- SearchConfig
    DesignSpace *-- GlobalConfig
    ProcessingSpace *-- ProcessingStep
    SearchConfig *-- SearchConstraint
    SearchConfig --> SearchStrategy
    GlobalConfig --> OutputStage
```

## Sequence Diagrams

### Main Forge Process

The sequence of operations when constructing a design space.

```mermaid
sequenceDiagram
    participant User
    participant ForgeAPI
    participant Parser
    participant Registry
    participant Validator
    participant DesignSpace
    
    User->>ForgeAPI: forge(model_path, blueprint_path)
    ForgeAPI->>ForgeAPI: validate_model_path()
    ForgeAPI->>Parser: load_blueprint(blueprint_path)
    Parser->>Parser: yaml.safe_load()
    Parser-->>ForgeAPI: blueprint_data
    
    ForgeAPI->>Parser: parse(blueprint_data, model_path)
    Parser->>Parser: validate_version()
    
    loop For each section
        Parser->>Parser: parse_section()
        Parser->>Registry: validate plugins exist
        Registry-->>Parser: validation result
        alt Plugin missing
            Parser->>Registry: get available alternatives
            Registry-->>Parser: suggestions
            Parser-->>User: Error with suggestions
        end
    end
    
    Parser->>Parser: resolve_config_hierarchy()
    Parser->>DesignSpace: create()
    Parser-->>ForgeAPI: design_space
    
    ForgeAPI->>Validator: validate(design_space)
    Validator->>Validator: check_model_exists()
    Validator->>Registry: validate_all_plugins()
    Validator->>Validator: check_combinations()
    Validator->>Validator: check_constraints()
    
    alt Validation fails
        Validator-->>ForgeAPI: ValidationResult(errors)
        ForgeAPI-->>User: ValidationError
    else Validation passes
        Validator-->>ForgeAPI: ValidationResult(warnings)
        ForgeAPI->>ForgeAPI: log_summary()
        ForgeAPI-->>User: design_space
    end
```

### Kernel Auto-Discovery

How kernels specified as simple strings get their backends auto-discovered.

```mermaid
sequenceDiagram
    participant Parser
    participant Registry
    participant Blueprint
    
    Blueprint->>Parser: kernels: ["LayerNorm", "MatMul"]
    
    loop For each kernel
        Parser->>Parser: is simple string?
        alt Simple string
            Parser->>Registry: get kernel "LayerNorm"
            Registry-->>Parser: kernel exists
            Parser->>Registry: list_backends_by_kernel("LayerNorm")
            Registry-->>Parser: ["LayerNormHLS", "LayerNormRTL"]
            Parser->>Parser: convert to ("LayerNorm", ["LayerNormHLS", "LayerNormRTL"])
        else Explicit backends
            Parser->>Registry: validate_kernel_backends()
            Registry-->>Parser: validation result
        end
    end
    
    Parser-->>Blueprint: enriched kernel list
```

## Error Handling

Phase 1 implements comprehensive error handling with helpful suggestions.

```mermaid
flowchart TB
    subgraph "Error Types"
        BE[BlueprintParseError<br/>- YAML syntax<br/>- Missing fields<br/>- Type errors]
        VE[ValidationError<br/>- Model not found<br/>- Invalid config<br/>- Space too large]
        PE[PluginNotFoundError<br/>- Missing kernel<br/>- Missing transform<br/>- Invalid backend]
        CE[ConfigurationError<br/>- Environment issues<br/>- Path problems]
    end
    
    subgraph "Error Context"
        LINE[Line/Column Info<br/>For YAML errors]
        SUGGEST[Suggestions<br/>Available plugins]
        MULTI[Multiple Errors<br/>Collected and reported]
    end
    
    subgraph "Error Flow"
        DETECT[Error Detection]
        CONTEXT[Add Context]
        COLLECT[Collect Errors]
        REPORT[Report to User]
    end
    
    BE --> DETECT
    VE --> DETECT
    PE --> DETECT
    CE --> DETECT
    
    DETECT --> CONTEXT
    LINE --> CONTEXT
    SUGGEST --> CONTEXT
    CONTEXT --> COLLECT
    COLLECT --> MULTI
    MULTI --> REPORT
    
    style BE fill:#ffcdd2
    style VE fill:#ffcdd2
    style PE fill:#ffcdd2
    style CE fill:#ffcdd2
```

## Plugin Integration

How Phase 1 integrates with the plugin registry for validation and auto-discovery.

```mermaid
graph TB
    subgraph "Phase 1"
        PARSER[Blueprint Parser]
        VAL[Validator]
    end
    
    subgraph "Plugin Registry"
        REG[Registry Instance]
        TRANS[transforms: Dict]
        KERN[kernels: Dict]
        BACK[backends: Dict]
        IDX[Pre-computed Indexes<br/>- backends_by_kernel<br/>- transforms_by_stage]
    end
    
    subgraph "Operations"
        CHECK["Check Existence<br/>O(1) lookup"]
        LIST[List Available<br/>For suggestions]
        AUTO[Auto-discover<br/>Backends for kernel]
        VALIDATE[Validate Combos<br/>Kernel + backends]
    end
    
    PARSER --> CHECK
    PARSER --> AUTO
    VAL --> CHECK
    VAL --> VALIDATE
    
    CHECK --> REG
    LIST --> REG
    AUTO --> IDX
    VALIDATE --> IDX
    
    REG --> TRANS
    REG --> KERN
    REG --> BACK
    
    style REG fill:#fff9c4
    style IDX fill:#e1f5fe
```

## Configuration Management

The hierarchical configuration system allows settings at multiple levels.

```mermaid
graph TB
    subgraph "Configuration Sources (Priority Order)"
        BP_SEARCH[Blueprint search.timeout_minutes<br/>Highest Priority]
        BP_GLOBAL[Blueprint global.timeout_minutes]
        USER_CFG[~/.brainsmith/config.yaml]
        ENV[Environment Variables<br/>BRAINSMITH_TIMEOUT_MINUTES]
        DEFAULT[Default Values<br/>Lowest Priority]
    end
    
    subgraph "Resolution Process"
        LOAD[Load All Sources]
        MERGE[Merge by Priority]
        APPLY[Apply to DesignSpace]
    end
    
    subgraph "Configurable Values"
        TIMEOUT[timeout_minutes<br/>Default: 60]
        MAX_COMBO[max_combinations<br/>Default: 100,000]
    end
    
    BP_SEARCH --> LOAD
    BP_GLOBAL --> LOAD
    USER_CFG --> LOAD
    ENV --> LOAD
    DEFAULT --> LOAD
    
    LOAD --> MERGE
    MERGE --> APPLY
    APPLY --> TIMEOUT
    APPLY --> MAX_COMBO
    
    style BP_SEARCH fill:#c8e6c9
    style MERGE fill:#e1f5fe
```

## Key Design Decisions

### 1. Direct Model Path
- ONNX model path is passed directly without analysis
- Validation ensures file exists and has .onnx extension
- Keeps Phase 1 focused on blueprint parsing

### 2. Plugin-Aware Parsing
- All plugin references validated at parse time
- Auto-discovery reduces blueprint verbosity
- Helpful error messages with available alternatives

### 3. Combinatorial Expansion
- Design space defines the space, not individual configs
- Lazy evaluation - combinations generated on demand
- Supports complex patterns (optional, mutually exclusive)

### 4. Fail-Fast Validation
- Comprehensive validation before expensive operations
- Errors vs warnings distinction
- Multiple errors collected and reported together

### 5. Clean Separation
- Phase 1 only constructs the space
- No execution or exploration logic
- Clear handoff to Phase 2 via DesignSpace object

## Usage Example

```python
from brainsmith.core.phase1 import forge

# Simple usage
design_space = forge("model.onnx", "blueprint.yaml")
print(f"Total combinations: {design_space.get_total_combinations()}")

# With optimization analysis
from brainsmith.core.phase1 import ForgeAPI
api = ForgeAPI(verbose=True)
design_space = api.forge_optimized("model.onnx", "blueprint.yaml")
```

## Summary

Phase 1 provides a robust, validated foundation for the DSE system by:
- Parsing complex blueprint specifications
- Validating all configurations against the plugin registry
- Auto-discovering available options
- Constructing a complete design space representation
- Providing clear error messages and helpful suggestions

The architecture emphasizes simplicity, correctness, and performance while maintaining flexibility for future enhancements.