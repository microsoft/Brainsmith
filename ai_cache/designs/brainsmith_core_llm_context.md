# BrainSmith Core - LLM Context Document

## Quick Start

BrainSmith Core converts PyTorch models to FPGA implementations via:
```python
result = brainsmith.core.forge('model.onnx', 'blueprint.yaml')
```

## Module Structure

```
brainsmith/core/
├── api.py              # Main forge() function - entry point
├── blueprint.py        # Blueprint V2 configuration system
├── blueprint_inheritance.py  # Blueprint inheritance logic
├── data/              # Metrics collection and export
├── dse/               # Design Space Exploration engine
├── finn/              # FINN framework integration
├── hooks/             # Event system for extensibility
├── metrics.py         # DSE metrics definitions
├── registry/          # Base registry infrastructure
└── cli.py            # Command-line interface
```

## Key Classes and Functions

### Primary API (`api.py`)
- `forge()` - Main function that orchestrates entire flow
- `validate_blueprint()` - Validates Blueprint V2 configuration
- `_load_blueprint_strict()` - Loads blueprint with validation
- `_create_exploration_config()` - Creates DSE configuration

### Blueprint System (`blueprint.py`)
- `DesignSpaceDefinition` - Main blueprint dataclass
- `load_blueprint()` - Loads YAML to DesignSpaceDefinition
- `NodeComponents`, `TransformComponents` - Component definitions
- `Objective`, `Constraint` - Optimization targets

### DSE Engine (`dse/`)
- `ComponentCombination` - Represents specific hw configuration
- `CombinationGenerator` - Generates valid combinations
- `DesignSpaceExplorer` - Main DSE orchestrator
- `ExplorationStrategy` - Protocol for exploration strategies
- `ResultsAnalyzer` - Analyzes DSE results, finds Pareto frontier

### FINN Integration (`finn/`)
- `FINNEvaluationBridge` - Evaluates combinations via FINN
- `LegacyConversionLayer` - Converts to FINN DataflowBuildConfig
- `MetricsExtractor` - Extracts standardized metrics
- `ConfigBuilder` - Builds FINN configurations

### Data Management (`data/`)
- `BuildMetrics` - Container for all metrics types
- `collect_build_metrics()` - Primary collection function
- `export_dse_analysis()` - Exports complete DSE analysis
- `DataManager` - Caching and lifecycle management

## Code Patterns

### 1. Blueprint Loading Pattern
```python
# Blueprint always loaded with strict validation
design_space = load_blueprint(blueprint_path)
# Runtime overrides applied via _apply_overrides()
if objectives:
    design_space = _apply_overrides(design_space, objectives, constraints, target_device)
```

### 2. DSE Execution Pattern
```python
# Create explorer with config
explorer = DesignSpaceExplorer(design_space, exploration_config)
# Execute exploration (automatically uses FINN)
results = explorer.explore_design_space(model_path)
# Extract best design and Pareto frontier
best_design = results.best_combination
pareto_frontier = results.pareto_frontier
```

### 3. Component Combination Pattern
```python
# Combinations track components across 6 FINN entrypoints
combination = ComponentCombination(
    node_components={
        'canonical_ops': ['Dense_8bit'],
        'hw_kernels': ['RTL_Dense_v2']
    },
    transform_components={
        'model_topology': ['folding'],
        'hw_kernel_transforms': ['buffer_insertion']
    }
)
```

### 4. Metrics Collection Pattern
```python
# Unified collection from any result type
metrics = collect_build_metrics(
    result,           # forge result, FINN build, or DSE result
    model_path,
    blueprint_path,
    parameters
)
# Metrics include performance, resources, quality, build info
```

### 5. FINN Evaluation Pattern
```python
# DSE calls evaluation bridge
bridge = FINNEvaluationBridge(design_space)
metrics = bridge.evaluate(combination, model_path)
# Bridge handles legacy/modern FINN interface selection
```

## Important Implementation Details

### Blueprint V2 Format
- YAML-based configuration
- Supports inheritance via `base_blueprint`
- Defines available components and exploration rules
- Contains FINN configuration mapping

### Exploration Rules
- `required`: Components that must be selected
- `optional`: Components that may be selected
- `mutually_exclusive`: Groups where only one can be selected
- `dependencies`: Component selection dependencies

### DSE Strategies
1. **Hierarchical**: 3-phase approach (kernels→transforms→parameters)
2. **Adaptive**: Learns from results, balances exploration/exploitation
3. **Pareto-guided**: Multi-objective optimization focus

### Caching System
- In-memory cache for current session
- Disk cache in `cache_directory`
- Cache keys generated from combination hash
- Automatic cleanup of old entries

### Legacy Support
- V1 compatibility via `forge_v1_compat()`
- Blueprint must be V2 format (V1 not supported)
- Results converted to V1 structure

## Common Tasks

### Add New Hardware Kernel
1. Add kernel definition to blueprint YAML
2. Implement kernel in `brainsmith/libraries/kernels/`
3. DSE will automatically explore it

### Implement Custom Strategy
1. Create class implementing `ExplorationStrategy` protocol
2. Add to blueprint's `dse_strategies` section
3. Set as `primary_strategy`

### Add New Metrics
1. Extend `MetricsExtractor._extract_*` methods
2. Add to `BuildMetrics` if new category
3. Update `DSEMetrics` scoring if needed

### Custom Export Format
1. Add export function to `data/export.py`
2. Follow pattern of existing exporters
3. Handle both single metrics and lists

## Performance Considerations

- Component combinations are cached (hash-based)
- Parallel evaluation supported (configure in DSE)
- Early termination on performance plateau
- Blueprint validation is strict but one-time

## Error Handling

- Blueprint validation errors are descriptive
- FINN failures return mock results in V1 compat mode
- DSE continues on individual evaluation failures
- All errors logged with context

## Integration Points

1. **ONNX Models**: Input models must be ONNX format
2. **FINN Framework**: Actual hardware compilation
3. **Blueprint YAML**: Configuration format
4. **Export Formats**: JSON, CSV, Excel support
5. **CLI**: Simple command-line interface

## Key Files Reference

- Entry point: `api.py:forge()`
- Configuration: `blueprint.py:DesignSpaceDefinition`
- DSE logic: `dse/space_explorer.py:DesignSpaceExplorer`
- FINN bridge: `finn/evaluation_bridge.py:FINNEvaluationBridge`
- Metrics: `data/types.py:BuildMetrics`