# Automation Libraries

This directory contains automation tools for batch processing, parameter sweeps, and workflow orchestration.

## Structure

### Core Components
- **`batch.py`** - Batch processing capabilities for large-scale operations
- **`sweep.py`** - Parameter sweep and exploration tools

### Extension Points
- **`contrib/`** - Stakeholder-contributed automation tools

## Usage

### Batch Processing
```python
from brainsmith.libraries.automation import batch

# Run batch compilation
results = batch.run_batch_compilation(
    models=model_list,
    blueprint="efficient_inference",
    output_dir="batch_results/"
)

# Process results in parallel
batch_analysis = batch.analyze_batch_results(results)
```

### Parameter Sweeps
```python
from brainsmith.libraries.automation import sweep

# Define parameter sweep
sweep_config = {
    'precision': [8, 16, 32],
    'parallelism': [1, 2, 4, 8],
    'memory_mode': ['streaming', 'block']
}

# Execute sweep
sweep_results = sweep.run_parameter_sweep(
    model=model,
    blueprint="high_throughput",
    parameters=sweep_config
)
```

### Integration with DSE
```python
from brainsmith.core.api import forge

# Automation tools work with DSE
result = forge(
    model=model,
    blueprint="optimization_sweep",
    automation={
        'batch_size': 10,
        'parallel_workers': 4,
        'sweep_parameters': sweep_config
    }
)
```

## Features
- **Parallel Processing**: Multi-threaded execution for large workloads
- **Progress Tracking**: Real-time progress monitoring and reporting
- **Result Aggregation**: Automatic collection and analysis of results
- **Error Handling**: Robust error recovery and reporting
- **Resource Management**: Intelligent resource utilization and scheduling

## Integration
- Seamless integration with DSE engine
- Support for distributed execution
- Integration with analysis tools for result processing
- Blueprint-driven automation workflows
- Hooks system integration for monitoring and callbacks