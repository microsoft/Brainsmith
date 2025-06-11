# BrainSmith DSE Design Document

## Functions Over Frameworks: North Star Aligned Design Space Exploration

### Version: 2.0.0-simplified
### Last Updated: December 2025

---

## 📋 Executive Summary

The BrainSmith DSE (Design Space Exploration) module has been completely redesigned to embody North Star principles. This document describes the transformation from a 6,000+ line enterprise framework to a ~1,100 line function-based system that makes FPGA design space exploration **as simple as calling a function**.

### Key Achievements
- **81% Code Reduction**: 6,000+ lines → ~1,100 lines
- **API Simplification**: 50+ enterprise classes → 8 core functions
- **Zero Configuration**: Works immediately without setup
- **Perfect Integration**: Seamless with all streamlined BrainSmith modules
- **External Tool Support**: Direct pandas/CSV/JSON export

---

## 🎯 Design Philosophy

### North Star Alignment

**Functions Over Frameworks**
- Replace enterprise objects with simple function calls
- `parameter_sweep('model.onnx', 'blueprint.yaml', parameters)` vs complex configuration objects
- Direct, immediate functionality without learning curves

**Simplicity Over Sophistication**
- Remove academic optimization algorithms (NSGA-II, SPEA2, MOEA/D)
- Focus on practical FPGA parameter sweeps
- Clear, readable code that serves real workflows

**Focus Over Feature Creep**
- FPGA-specific DSE only, no generic optimization
- Core functionality: parameter combinations, evaluation, analysis
- Remove unused research-oriented features

**Hooks Over Implementation**
- Export data to pandas/CSV/JSON for external analysis
- Enable matplotlib, scipy, R, Excel workflows
- Data accessibility over framework lock-in

**Performance Over Purity**
- Fast parameter sweeps with parallel execution
- Practical runtime estimation
- Efficient parameter sampling strategies

---

## 🏗️ Architecture Overview

### Module Structure

```
brainsmith/dse/
├── __init__.py          # Clean API exports, backwards compatibility
├── functions.py         # Core DSE functions
├── helpers.py           # Utility functions and data export
├── types.py             # Simple data structures
└── DESIGN.md           # This design document
```

### API Surface Area

**Core Functions (5)**
- `parameter_sweep()` - Main DSE function
- `batch_evaluate()` - Multiple model evaluation
- `find_best_result()` - Single metric optimization
- `compare_results()` - Multi-objective comparison
- `sample_design_space()` - Intelligent sampling

**Helper Functions (9)**
- `generate_parameter_grid()` - Cartesian product generation
- `create_parameter_samples()` - Smart sampling strategies
- `export_results()` - Data export for external tools
- `estimate_runtime()` - Practical time estimation
- `count_parameter_combinations()` - Space size calculation
- `validate_parameter_space()` - Input validation
- `create_parameter_subsets()` - Large space management
- `filter_results()` - Result filtering
- `sort_results()` - Result sorting

**Data Types (7)**
- `DSEResult` - Complete evaluation result
- `ParameterSet` - Named parameter combinations
- `ComparisonResult` - Multi-objective analysis
- `DSEConfiguration` - Simple configuration
- `ParameterSpace` - Type alias for parameter definitions
- `ParameterCombination` - Type alias for single combinations
- `MetricName` - Type alias for metric identifiers

**Total: 21 exports vs 50+ classes in enterprise framework**

---

## 🔧 Core Implementation

### 1. Parameter Space Definition

Simple dictionary-based parameter spaces:

```python
parameters = {
    'pe_count': [1, 2, 4, 8, 16],
    'simd_factor': [1, 2, 4],
    'precision': [8, 16, 32],
    'memory_mode': ['internal', 'external'],
    'clock_freq_mhz': [100, 150, 200, 250]
}
```

### 2. Core DSE Function

**`parameter_sweep()`** - The heart of DSE functionality:

```python
def parameter_sweep(
    model_path: str,
    blueprint_path: str, 
    parameters: ParameterSpace,
    config: Optional[DSEConfiguration] = None
) -> List[DSEResult]:
```

**Integration Points:**
- `brainsmith.core.api.forge()` - Core evaluation engine
- `brainsmith.blueprints.functions.load_blueprint_yaml()` - Blueprint loading
- `brainsmith.hooks.log_dse_event()` - Event logging
- `brainsmith.finn.build_accelerator()` - FINN integration fallback

**Execution Modes:**
- Sequential execution for debugging
- Parallel execution for performance
- Configurable via `DSEConfiguration.max_parallel`

### 3. Result Analysis

**Single Metric Optimization:**
```python
best = find_best_result(results, 'performance.throughput_ops_sec', 'maximize')
```

**Multi-Objective Analysis:**
```python
comparison = compare_results(
    results, 
    metrics=['performance.throughput_ops_sec', 'resources.lut_utilization_percent'],
    weights=[0.7, 0.3]
)
```

### 4. Data Export

**External Tool Integration:**
```python
# Pandas for Python analysis
df = export_results(results, 'pandas')
df.plot(x='pe_count', y='throughput', kind='scatter')

# CSV for Excel/R
export_results(results, 'csv', 'dse_results.csv')

# JSON for web tools
json_data = export_results(results, 'json')
```

---

## 🔗 Module Integration

### Core Module Integration

**API Function:** `brainsmith.core.api.forge()`
```python
# DSE calls forge() for each parameter combination
forge_result = forge(
    model_path=model_path,
    blueprint_path=None,  # Pass blueprint data directly
    **parameters
)
```

**Metrics System:** `brainsmith.core.metrics.DSEMetrics`
```python
# Standardized metrics container
metrics = create_metrics()
result = DSEResult(parameters=params, metrics=metrics, ...)
```

### Blueprints Module Integration

**Blueprint Loading:** `brainsmith.blueprints.functions.load_blueprint_yaml()`
```python
# Simple blueprint loading
blueprint_data = load_blueprint_yaml(blueprint_path)
build_steps = get_build_steps(blueprint_data)
objectives = get_objectives(blueprint_data)
```

### Hooks Module Integration

**Event Logging:** `brainsmith.hooks.log_dse_event()`
```python
# Automatic DSE event tracking
log_optimization_event('dse_start', {
    'model': model_path,
    'parameter_count': len(parameters),
    'total_combinations': total_combinations
})
```

### FINN Module Integration

**FINN Interface:** `brainsmith.finn.build_accelerator()`
```python
# Direct FINN integration for evaluation
finn_result = build_accelerator(
    model_path=model_path,
    blueprint_config=blueprint_data,
    **parameters
)
```

### External Tool Integration

**Analysis Hooks Pattern:**
- Direct pandas DataFrame export
- CSV export for Excel/R/Python
- JSON export for web tools
- No framework lock-in

---

## 📊 Data Flow

### 1. Input Processing
```
Parameter Space → Parameter Grid → Parameter Combinations
     ↓
Blueprint Loading → Build Configuration
     ↓
Model Path → ONNX Model Reference
```

### 2. Evaluation Pipeline
```
Parameter Combination → Core.forge() → Build Result
                    ↓
                Metrics Extraction → DSEResult
                    ↓
                Success/Failure → Result Collection
```

### 3. Analysis Pipeline
```
DSEResult List → Filter/Sort → Analysis Functions
               ↓
          Comparison/Optimization → ComparisonResult
               ↓
          Export Functions → pandas/CSV/JSON
```

### 4. Integration Pipeline
```
DSE Events → Hooks.log_dse_event() → Event Tracking
           ↓
Blueprint Data → Blueprints.load_blueprint_yaml() → Configuration
           ↓
FINN Integration → finn.build_accelerator() → Hardware Build
```

---

## 🎨 Usage Patterns

### Basic Parameter Sweep

```python
from brainsmith.dse import parameter_sweep, find_best_result

# Define parameter space
parameters = {
    'pe_count': [1, 2, 4, 8],
    'simd_factor': [1, 2, 4],
    'precision': [8, 16]
}

# Run parameter sweep
results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)

# Find best configuration
best = find_best_result(results, 'performance.throughput_ops_sec', 'maximize')
print(f"Best configuration: {best.parameters}")
```

### Large Design Space Sampling

```python
from brainsmith.dse import sample_design_space, parameter_sweep

# Large parameter space
large_parameters = {
    'pe_count': list(range(1, 33)),      # 32 options
    'simd_factor': [1, 2, 4, 8, 16],     # 5 options  
    'precision': [4, 8, 16, 32],         # 4 options
    'buffer_depth': [32, 64, 128, 256, 512]  # 5 options
}
# Total: 3,200 combinations

# Sample intelligently
samples = sample_design_space(large_parameters, 'lhs', n_samples=50)

# Evaluate samples only
results = []
for sample in samples:
    result = parameter_sweep('model.onnx', 'blueprint.yaml', {'single': [sample]})
    results.extend(result)
```

### Multi-Objective Optimization

```python
from brainsmith.dse import parameter_sweep, compare_results

# Run parameter sweep
results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)

# Multi-objective comparison
comparison = compare_results(
    results,
    metrics=[
        'performance.throughput_ops_sec',
        'resources.lut_utilization_percent', 
        'power.total_power_mw'
    ],
    weights=[0.5, 0.3, 0.2]  # Prioritize throughput, then resources, then power
)

# Get top configurations
top_configs = comparison.get_top_n(5)
for i, config in enumerate(top_configs):
    print(f"{i+1}. {config.parameters} (score: {comparison.ranking[i]})")
```

### External Analysis Integration

```python
from brainsmith.dse import parameter_sweep, export_results
import matplotlib.pyplot as plt
import seaborn as sns

# Run DSE
results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)

# Export to pandas
df = export_results(results, 'pandas')

# Analysis with external tools
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='pe_count', y='performance_throughput_ops_sec', 
                hue='precision', size='resources_lut_utilization_percent')
plt.title('FPGA Design Space Exploration Results')
plt.show()

# Export for other tools
export_results(results, 'csv', 'results.csv')  # For Excel/R
export_results(results, 'json', 'results.json')  # For web tools
```

---

## 🔄 Migration Guide

### From Enterprise Framework

**Old Enterprise Pattern:**
```python
# Complex configuration objects
config = DSEConfiguration(
    strategy=MultiObjectiveStrategy(
        algorithm=NSGA2Algorithm(
            population_size=100,
            crossover_probability=0.9,
            mutation_probability=0.1
        )
    ),
    objectives=[
        Objective('throughput', direction='maximize'),
        Objective('resources', direction='minimize')
    ]
)

# Abstract base classes
class CustomStrategy(OptimizationStrategy):
    def __init__(self):
        super().__init__()
        # 50+ lines of boilerplate...

# Complex execution
engine = create_dse_engine(config)
results = engine.execute(parameter_space)
```

**New Simplified Pattern:**
```python
# Just call a function
results = parameter_sweep('model.onnx', 'blueprint.yaml', parameters)
best = find_best_result(results, 'throughput_ops_sec', 'maximize')
```

### Backwards Compatibility

**Deprecated Functions with Warnings:**
- `create_dse_engine()` → Use `parameter_sweep()`
- `DSEInterface()` → Use `parameter_sweep()`
- `DSEEngine()` → Use `parameter_sweep()`

**Migration Steps:**
1. Replace complex configuration objects with simple parameter dictionaries
2. Replace engine creation with direct function calls
3. Replace result analysis classes with simple analysis functions
4. Replace framework exports with pandas/CSV exports

---

## 🛠️ Implementation Details

### Error Handling

**Graceful Degradation:**
- Module import failures use fallback implementations
- Individual evaluation failures are logged and optionally continued
- Missing dependencies trigger warnings, not failures

**Configuration Options:**
```python
config = DSEConfiguration(
    continue_on_failure=True,  # Continue on individual failures
    timeout_seconds=3600,      # Overall timeout
    max_parallel=4             # Parallel execution workers
)
```

### Performance Optimization

**Parallel Execution:**
- ThreadPoolExecutor for I/O bound evaluations
- Configurable worker count
- Timeout handling for stuck evaluations

**Memory Management:**
- Streaming result processing for large parameter spaces
- Optional result subset creation for memory constraints
- Lazy evaluation patterns where possible

**Runtime Estimation:**
```python
estimated_time = estimate_runtime(parameter_combinations, benchmark_time=30.0)
# Provides realistic time estimates for planning
```

### Sampling Strategies

**Random Sampling:**
- Uniform random selection from parameter space
- Good for initial exploration

**Latin Hypercube Sampling (LHS):**
- Better space coverage than random
- Ensures even distribution across parameter ranges
- Optimal for limited evaluation budgets

**Grid Sampling:**
- Full Cartesian product
- Exhaustive but potentially expensive
- Good for small parameter spaces

### Data Export Patterns

**Pandas Integration:**
```python
df = export_results(results, 'pandas')
# Full DataFrame with flattened metrics
# Ready for matplotlib, seaborn, scipy analysis
```

**CSV Export:**
```python
export_results(results, 'csv', 'results.csv')
# Clean CSV for Excel, R, other tools
# Works with or without pandas
```

**JSON Export:**
```python
json_data = export_results(results, 'json')
# Structured JSON for web tools, databases
# Full result structure preserved
```

---

## 🧪 Testing Strategy

### Test Coverage Areas

**Function Testing:**
- All core DSE functions with various parameter combinations
- Error handling and edge cases
- Integration with mock streamlined modules

**Integration Testing:**
- Core module integration via `forge()`
- Blueprint module integration via `load_blueprint_yaml()`
- Hooks module integration via event logging
- FINN module integration via `build_accelerator()`

**Performance Testing:**
- Large parameter space handling
- Parallel execution validation
- Memory usage optimization
- Runtime estimation accuracy

**Export Testing:**
- Pandas DataFrame generation
- CSV export with and without pandas
- JSON export and round-trip serialization
- External tool integration validation

### Test Implementation

**Location:** `tests/test_dse_simplification.py`
**Coverage:** All functions, integration points, error conditions
**Approach:** Mock external dependencies, validate function behavior
**CI Integration:** Automated testing on module changes

---

## 📈 Metrics and Monitoring

### Performance Metrics

**DSE Execution Metrics:**
- Total parameter combinations evaluated
- Success rate (successful builds / total evaluations)
- Average evaluation time per combination
- Parallel efficiency metrics

**Result Quality Metrics:**
- Pareto frontier analysis for multi-objective results
- Convergence analysis for sampling strategies
- Coverage analysis for parameter space exploration

### Event Logging

**DSE Events via Hooks Module:**
```python
log_optimization_event('dse_start', {
    'model': model_path,
    'parameter_count': len(parameters),
    'total_combinations': total_combinations,
    'strategy': 'grid' | 'random' | 'lhs'
})

log_dse_event('parameter_evaluation_complete', {
    'parameters': params,
    'success': result.build_success,
    'build_time': result.build_time,
    'metrics': result.metrics.to_dict()
})
```

---

## 🔮 Future Evolution

### Planned Enhancements

**Additional Sampling Strategies:**
- Bayesian optimization for expensive evaluations
- Evolutionary algorithms for complex spaces
- Active learning strategies

**Enhanced Integration:**
- Direct integration with more external tools
- Real-time result visualization
- Distributed execution across multiple machines

**Specialized Functions:**
- Domain-specific parameter space generators
- FPGA architecture-aware sampling
- Power-performance co-optimization helpers

### Design Principles for Future Development

**Maintain North Star Alignment:**
- New features as simple functions, not framework extensions
- Preserve zero-configuration operation
- Keep external tool integration as primary export mechanism

**Backwards Compatibility:**
- Maintain existing function signatures
- Add new optional parameters rather than breaking changes
- Provide migration guides for any necessary changes

**Performance Focus:**
- Optimize for real FPGA workflows
- Maintain fast evaluation pipelines
- Preserve memory efficiency for large spaces

---

## 📚 References

### BrainSmith Module Dependencies

- **Core Module:** `brainsmith.core.api.forge()`, `brainsmith.core.metrics`
- **Blueprints Module:** `brainsmith.blueprints.functions`
- **Hooks Module:** `brainsmith.hooks.log_dse_event()`
- **FINN Module:** `brainsmith.finn.build_accelerator()`

### External Dependencies

- **Required:** Python 3.8+, numpy
- **Optional:** pandas (for DataFrame export), matplotlib (for visualization)
- **Development:** pytest (for testing), mock (for integration testing)

### Related Documentation

- `DSE_SIMPLIFICATION_IMPLEMENTATION_PLAN.md` - Implementation roadmap
- `DSE_SIMPLIFICATION_COMPLETE.md` - Completion summary
- `tests/test_dse_simplification.py` - Test implementation
- `dse_demo.py` - Working demonstration

---

## 📝 Document Metadata

**Document Version:** 1.0  
**DSE Module Version:** 2.0.0-simplified  
**Last Updated:** December 2025  
**Next Review:** March 2026  

**Authors:** BrainSmith Development Team  
**Reviewers:** North Star Architecture Committee  
**Approval:** Technical Leadership  

---

*This design document reflects the successful transformation of the BrainSmith DSE module from enterprise complexity to North Star simplicity. The implementation achieves the core goal: **making FPGA design space exploration as simple as calling a function**.*