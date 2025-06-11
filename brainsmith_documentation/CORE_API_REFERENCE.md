# BrainSmith Core API Reference

This document provides comprehensive reference for BrainSmith's core API, covering the primary `forge()` function, essential helper functions, and core classes.

## Primary `forge()` Function

The heart of BrainSmith, implemented in `brainsmith/core/api.py`:

```python
def forge(
    model_path: str,
    blueprint_path: str,
    objectives: Dict[str, Any] = None,
    constraints: Dict[str, Any] = None,
    target_device: str = None,
    is_hw_graph: bool = False,
    build_core: bool = True,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
```

### Parameters

- **model_path**: Path to pre-quantized ONNX model
- **blueprint_path**: Path to blueprint YAML (design space specification)
- **objectives**: Target objectives (latency/throughput requirements)
- **constraints**: Hardware resource budgets, optimization priorities
- **target_device**: Target FPGA device specification
- **is_hw_graph**: If True, input is already a Dataflow Graph, skip to HW optimization
- **build_core**: If False, exit after Dataflow Graph generation
- **output_dir**: Optional output directory for results

### Returns

Dictionary containing:
- **dataflow_graph**: ONNX graph of HWCustomOps describing Dataflow Core
- **dataflow_core**: Stitched IP design (if build_core=True)
- **metrics**: Performance and resource utilization metrics
- **analysis**: DSE analysis and recommendations

### Key Features

- **Dual Mode Operation**: Standard (Model→Hardware) vs Hardware Graph optimization
- **Checkpoint Support**: Option to exit after Dataflow Graph generation
- **Comprehensive Validation**: Hard errors for invalid inputs/blueprints
- **Flexible Configuration**: Objectives, constraints, and device targeting

### Example Usage

```python
# Basic usage
result = brainsmith.forge('model.onnx', 'blueprint.yaml')

# With optimization objectives
result = brainsmith.forge(
    'model.onnx', 'blueprint.yaml',
    objectives={'throughput': {'direction': 'maximize', 'weight': 1.0}},
    constraints={'max_luts': 0.8, 'max_power': 25.0}
)

# Hardware graph optimization only
result = brainsmith.forge(
    'dataflow_graph.onnx', 'blueprint.yaml',
    is_hw_graph=True
)
```

## Essential Helper Functions

BrainSmith provides 12 essential helper functions as exported in `brainsmith/core/__init__.py`:

### 1-4: Automation Helpers

#### parameter_sweep()
```python
results = brainsmith.parameter_sweep(
    model_path: str,
    blueprint_path: str, 
    param_ranges: Dict[str, List[Any]],
    max_workers: int = 4
) -> List[Dict[str, Any]]
```

Explore parameter combinations by running forge() with different configurations.

#### batch_process()
```python
batch_results = brainsmith.batch_process(
    model_blueprint_pairs: List[Tuple[str, str]],
    common_config: Optional[Dict[str, Any]] = None,
    max_workers: int = 4
) -> List[Dict[str, Any]]
```

Process multiple model/blueprint pairs in parallel.

#### find_best()
```python
best = brainsmith.find_best(
    results: List[Dict[str, Any]], 
    metric: str = 'throughput',
    maximize: bool = True
) -> Optional[Dict[str, Any]]
```

Find optimal result by specified metric from sweep or batch results.

#### aggregate_stats()
```python
stats = brainsmith.aggregate_stats(
    results: List[Dict[str, Any]]
) -> Dict[str, Any]
```

Generate statistical summary of results with success rates and metric distributions.

### 5-6: Event Management

#### log_optimization_event()
```python
brainsmith.log_optimization_event(
    event_type: str, 
    data: Dict[str, Any]
) -> None
```

Log optimization events for tracking and analysis.

#### register_event_handler()
```python
brainsmith.register_event_handler(
    event_type: str, 
    handler: EventHandler
) -> None
```

Register custom event handlers for extending the system.

### 7: FINN Integration

#### build_accelerator()
```python
finn_result = brainsmith.build_accelerator(
    model_path: str,
    blueprint_config: Dict[str, Any],
    output_dir: str = "./output"
) -> Dict[str, Any]
```

Generate complete FINN accelerator from optimized design.

### 8-9: Data Management

#### get_analysis_data()
```python
analysis_data = brainsmith.get_analysis_data(
    dse_results
) -> Dict[str, Any]
```

Extract structured data from DSE results for external analysis.

#### export_results()
```python
brainsmith.export_results(
    data: Dict[str, Any],
    output_path: str,
    format: str = 'csv'
) -> None
```

Export results to various formats (CSV, JSON, Excel).

### 10-11: Design Space Management

#### sample_design_space()
```python
points = brainsmith.sample_design_space(
    design_space: DesignSpace, 
    n_samples: int = 10,
    strategy: str = "random"
) -> List[DesignPoint]
```

Generate sample points from parameter space for exploration.

#### validate_blueprint()
```python
is_valid, errors = brainsmith.validate_blueprint(
    blueprint_path: str
) -> Tuple[bool, List[str]]
```

Validate blueprint configuration before running DSE.

## Core Classes

Three essential classes provide core concepts:

### 1. DesignSpace

Manages parameter space for blueprint instantiation:

```python
from brainsmith import DesignSpace, ParameterDefinition

# Create design space
space = DesignSpace("conv_optimization")

# Add parameters
space.add_parameter(ParameterDefinition(
    name="pe_count",
    param_type="integer",
    range_min=1,
    range_max=64,
    default=8
))

# Generate design points
points = space.sample_points(n_samples=50)
```

**Key Methods:**
- `add_parameter(param_def)`: Add parameter definition
- `create_design_point(parameters)`: Create validated design point
- `sample_points(n_samples)`: Sample parameter combinations

### 2. DSEInterface

Main interface for design space exploration:

```python
from brainsmith import DSEInterface, DSEConfiguration, DSEObjective

# Configure DSE
config = DSEConfiguration(
    objectives=[DSEObjective("throughput", OptimizationObjective.MAXIMIZE)],
    max_evaluations=100
)

# Run exploration
dse = DSEInterface(config)
results = dse.explore_design_space('model.onnx')
```

**Key Methods:**
- `explore_design_space(model_path)`: Execute complete DSE
- `optimize_dataflow_graph(graph)`: Optimize existing graph
- `get_pareto_frontier(results)`: Extract Pareto optimal solutions

### 3. DSEMetrics

Performance metrics collection and analysis:

```python
from brainsmith import DSEMetrics

# Create metrics
metrics = DSEMetrics()
metrics.performance.throughput_ops_sec = 1000000.0
metrics.resources.lut_utilization_percent = 85.0

# Get optimization score
score = metrics.get_optimization_score()
```

**Key Attributes:**
- `performance`: PerformanceMetrics (throughput, latency, frequency)
- `resources`: ResourceMetrics (LUT, DSP, BRAM utilization)
- `build_success`: Boolean indicating successful build

**Key Methods:**
- `get_optimization_score()`: Calculate combined optimization score
- `to_dict()`: Convert to dictionary for serialization
- `to_json()`: Export as JSON string

## Configuration Examples

### Objectives Configuration
```python
objectives = {
    'throughput': {
        'direction': 'maximize',
        'weight': 0.7,
        'target': 1000000  # target ops/sec
    },
    'power': {
        'direction': 'minimize', 
        'weight': 0.3,
        'target': 25.0  # target watts
    }
}
```

### Constraints Configuration
```python
constraints = {
    'max_luts': 0.8,        # 80% LUT utilization
    'max_dsps': 0.9,        # 90% DSP utilization
    'max_brams': 0.7,       # 70% BRAM utilization
    'target_frequency': 200, # 200 MHz target
    'max_power': 30.0       # 30W power budget
}
```

### Blueprint YAML Structure
```yaml
name: "efficient_cnn"
version: "1.0"
description: "Optimized CNN accelerator blueprint"

parameters:
  pe_count:
    type: "integer"
    range: [4, 64]
    default: 16
  simd_width:
    type: "integer" 
    range: [2, 32]
    default: 8
  memory_mode:
    type: "categorical"
    values: ["internal", "external"]
    default: "internal"

targets:
  throughput:
    direction: "maximize"
    priority: "high"
  latency:
    direction: "minimize"
    priority: "medium"

constraints:
  max_lut_util: 0.85
  max_dsp_util: 0.90
  target_freq_mhz: 200
```

## Error Handling

### Hard Errors (Immediate Failure)
- Invalid file paths or missing files
- Malformed blueprint YAML
- Missing critical dependencies

### Soft Errors (Graceful Fallback) 
- Missing optional components (DSE system, FINN)
- Individual design point failures
- Resource constraint violations

### Validation Hierarchy
1. **Input Validation**: File existence, format checking
2. **Blueprint Validation**: YAML structure, parameter definitions
3. **Configuration Validation**: Objective/constraint consistency
4. **Runtime Validation**: Component availability, resource limits

## Performance Characteristics

### Function Call Performance
- `forge()`: 10-60 seconds depending on model complexity
- `parameter_sweep()`: Scales linearly with parameter combinations
- `find_best()`: <100ms for typical result sets
- `validate_blueprint()`: <10ms for standard blueprints

### Memory Usage
- Minimal memory footprint for core functions
- DSE results scale with evaluation count
- Efficient caching for repeated operations

### Scalability
- Design Points: Efficiently handles 100s of evaluations
- Model Size: Scales with available system memory
- Parallel Processing: Automatic parallelization in automation functions