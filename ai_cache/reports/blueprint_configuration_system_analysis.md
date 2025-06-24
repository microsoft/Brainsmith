# Blueprint Configuration System Analysis

Generated: 2025-06-23

## Executive Summary

The Blueprint V2 configuration system in BrainSmith represents a sophisticated design space definition framework that bridges high-level accelerator specifications with the underlying FINN hardware generation pipeline. It provides a hierarchical, inheritance-based approach to defining FPGA accelerator configurations with comprehensive support for design space exploration (DSE).

## Core Architecture

### 1. DesignSpaceDefinition Structure

The `DesignSpaceDefinition` class serves as the central data structure that encapsulates:

```python
@dataclass
class DesignSpaceDefinition:
    name: str                                    # Blueprint identifier
    version: str = "2.0"                        # Version control
    base_blueprint: Optional[str] = None        # Inheritance support
    nodes: NodeDesignSpace                      # Computational nodes
    transforms: TransformDesignSpace            # Processing transforms
    configuration_files: Dict[str, str]         # External configs
    dse_strategies: Optional[DSEStrategies]     # Exploration strategies
    objectives: List[Objective]                 # Optimization goals
    constraints: List[Constraint]               # Resource limits
    raw_blueprint_config: Dict[str, Any]        # Preserved YAML data
```

### 2. Component Organization

The blueprint system organizes components into two primary categories:

**Nodes (Computational Units):**
- `canonical_ops`: High-level operations (LayerNorm, MatMul, etc.)
- `hw_kernels`: Hardware implementations with multiple options per op

**Transforms (Processing Stages):**
- `model_topology`: Graph-level optimizations (cleanup, streamlining)
- `hw_kernel`: Hardware-specific optimizations (folding, parallelization)
- `hw_graph`: System-level optimizations

### 3. Exploration Rules System

Each component space includes sophisticated exploration rules:

```python
@dataclass
class ExplorationRules:
    required: List[str]                    # Must-have components
    optional: List[str]                    # Can-have components
    mutually_exclusive: List[List[str]]    # Either-or groups
    dependencies: Dict[str, List[str]]     # Component dependencies
```

This enables:
- Mandatory component enforcement
- Mutual exclusion constraints
- Dependency management
- Circular dependency detection

## Inheritance and Override System

### 1. Hierarchical Blueprints

The system supports blueprint inheritance through `base_blueprint`:

```yaml
# bert_ultra_small_v2.yaml
name: "bert_ultra_small_v2"
base_blueprint: "transformer_base"  # Inherits from base
```

### 2. Intelligent Merging Rules

The inheritance system uses context-aware merging:

- **Simple values**: Derived overrides base
- **Lists**: Extended with deduplication
- **Dicts**: Recursive deep merge
- **Component spaces**: Special merge logic for available/exploration
- **Objectives/Constraints**: Name-based override or extension

### 3. Path Resolution

Base blueprints are resolved through multiple strategies:
- Same directory
- `base/` subdirectory
- Parent directory
- With `.yaml` or `.yml` extensions

### 4. Circular Dependency Protection

The system validates inheritance chains to prevent circular dependencies before loading.

## Integration with forge() API

### 1. Blueprint Loading Flow

```python
def forge(model_path, blueprint_path, ...):
    # 1. Load with strict validation
    design_space = _load_blueprint_strict(blueprint_path)
    
    # 2. Apply runtime overrides
    if objectives or constraints:
        design_space = _apply_overrides(design_space, objectives, constraints)
    
    # 3. Create DSE explorer
    explorer = DesignSpaceExplorer(design_space, exploration_config)
    
    # 4. Execute exploration
    results = explorer.explore_design_space(model_path)
```

### 2. Runtime Overrides

The forge API allows runtime modification of:
- Objectives (replace blueprint defaults)
- Constraints (merge with blueprint)
- Target device (adds as constraint)
- DSE configuration parameters

### 3. Validation Pipeline

Multiple validation layers ensure blueprint integrity:
1. YAML structure validation
2. Component reference validation
3. Exploration rule consistency
4. Objective/constraint uniqueness
5. Strategy compatibility checks

## FINN Integration Features

### 1. Legacy FINN Support

Blueprints can enable legacy FINN compatibility:

```yaml
legacy_finn: true
legacy_preproc: [...]  # Custom step ordering
legacy_postproc: [...]
```

### 2. Direct FINN Configuration

The `finn_config` section maps directly to FINN's DataflowBuildConfig:

```yaml
finn_config:
  synth_clk_period_ns: 5.0
  target_fps: 1000
  auto_fifo_depths: true
  split_large_fifos: true
  save_intermediate_models: true
```

### 3. Runtime Adaptations

Blueprints support conditional configurations:

```yaml
runtime_adaptations:
  ultra_small_mode:
    model_overrides: {...}
    build_overrides: {...}
    constraint_overrides: {...}
    finn_config_overrides: {...}
```

## DSE Strategy System

### 1. Multi-Strategy Support

Blueprints define multiple exploration strategies:

```yaml
dse_strategies:
  primary_strategy: "balanced"
  strategies:
    ultra_fast:
      max_evaluations: 1
      sampling: "fixed"
    balanced:
      max_evaluations: 10
      sampling: "random"
    comprehensive:
      max_evaluations: 50
      sampling: "adaptive"
```

### 2. Strategy Configuration

Each strategy includes:
- `max_evaluations`: Exploration budget
- `sampling`: Algorithm (random, grid, adaptive, etc.)
- `focus_areas`: Optimization priorities
- `objectives`/`constraints`: Strategy-specific overrides

### 3. Objective Definition

Multi-objective optimization with:
- Direction (minimize/maximize)
- Weights for scalarization
- Target values for satisficing
- Descriptions for clarity

## Key Design Patterns

### 1. Flexibility Through Defaults

- Empty component spaces are valid (FINN uses defaults)
- All 6 FINN entrypoints can be optional
- Graceful degradation for missing components

### 2. Validation-First Approach

- Extensive validation at load time
- Clear error messages with context
- Fail-fast philosophy

### 3. Separation of Concerns

- Blueprint defines the "what" (design space)
- DSE explorer handles the "how" (exploration)
- FINN bridge executes the "build" (implementation)

### 4. Progressive Enhancement

- Start with base blueprints
- Override specific aspects
- Add runtime adaptations
- Maintain full YAML fidelity

## Best Practices

### 1. Blueprint Design

- Use inheritance for families of accelerators
- Define clear exploration rules
- Set realistic objectives and constraints
- Document expected characteristics

### 2. Component Definition

- Keep required components minimal
- Use mutually exclusive groups for alternatives
- Define dependencies explicitly
- Validate circular dependencies

### 3. DSE Configuration

- Start with ultra_fast for testing
- Use balanced for development
- Reserve comprehensive for production
- Match strategy to exploration budget

### 4. FINN Integration

- Leverage finn_config for direct control
- Use legacy modes only when necessary
- Enable intermediate saves for debugging
- Configure FIFO settings carefully

## Conclusion

The Blueprint V2 system provides a powerful, flexible framework for defining FPGA accelerator design spaces. Its inheritance system, comprehensive validation, and tight integration with both the forge() API and FINN backend make it an effective solution for managing the complexity of hardware accelerator development. The system successfully balances expressiveness with usability, enabling both simple configurations and sophisticated multi-objective design space exploration.