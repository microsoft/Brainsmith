# Six-Entrypoint System: Detailed Design Document

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [System Overview](#system-overview)
3. [Architecture](#architecture)
4. [Detailed Flow](#detailed-flow)
5. [Component Details](#component-details)
6. [Transform Mapping](#transform-mapping)
7. [Subprocess Execution](#subprocess-execution)
8. [Error Handling](#error-handling)
9. [Integration Points](#integration-points)
10. [Example Walkthrough](#example-walkthrough)

## Executive Summary

The 6-entrypoint system is a modern FINN interface that allows fine-grained control over the neural network compilation pipeline by inserting transforms at six specific points in the workflow. Unlike the legacy system that uses pre-packaged build steps, this system works with individual transforms, providing greater flexibility and modularity.

## System Overview

### What are the 6 Entrypoints?

The 6 entrypoints represent strategic insertion points in the FINN compilation pipeline:

1. **Entrypoint 1**: After model loading (pre-processing transforms)
2. **Entrypoint 2**: After quantization setup (model topology transforms)
3. **Entrypoint 3**: After hardware kernel mapping (kernel selection)
4. **Entrypoint 4**: After kernel specialization (kernel configuration)
5. **Entrypoint 5**: After kernel-level optimizations (kernel transforms)
6. **Entrypoint 6**: After graph-level optimizations (graph transforms)

### Key Concepts

- **Transform**: A single, atomic operation that modifies the neural network graph (e.g., `FoldConstants`, `InferShapes`)
- **Entrypoint**: A specific location in the compilation pipeline where transforms can be inserted
- **Subprocess Isolation**: Execution in a separate process for reliability and error containment

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        DSE Explorer                         │
│                   (Design Space Exploration)                │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    Backend Factory                          │
│              (detect_workflow & create_backend)             │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  SixEntrypointBackend                       │
├─────────────────────────────────────────────────────────────┤
│ • Receives EvaluationRequest                                │
│ • Extracts entrypoint configuration                        │
│ • Validates transforms                                      │
│ • Creates subprocess execution script                      │
│ • Executes in isolation                                    │
│ • Extracts metrics from results                           │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                   Subprocess Execution                      │
├─────────────────────────────────────────────────────────────┤
│ • Loads ONNX model                                         │
│ • Maps transform names to objects                          │
│ • Calls FINN build_dataflow_v2()                          │
│ • Saves results to JSON                                    │
└─────────────────────────────┬───────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      MetricsExtractor                       │
│                  (Extract standardized metrics)             │
└─────────────────────────────────────────────────────────────┘
```

## Detailed Flow

### 1. Request Initiation

```python
# DSE Explorer creates an evaluation request
request = EvaluationRequest(
    model_path="/path/to/model.onnx",
    combination={
        'entrypoint_1': ['FoldConstants', 'InferShapes'],
        'entrypoint_2': ['Streamline'],
        'entrypoint_3': ['MatrixVectorActivation'],
        'entrypoint_4': [],
        'entrypoint_5': ['MinimizeAccumulatorWidth'],
        'entrypoint_6': ['AnnotateCycles']
    },
    work_dir="/tmp/finn_work",
    timeout=3600  # 1 hour
)
```

### 2. Backend Selection

The workflow detector examines the blueprint configuration:

```python
def detect_workflow(blueprint: Dict[str, Any]) -> WorkflowType:
    # Check for legacy indicators
    if 'finn_config' in blueprint and 'build_steps' in blueprint['finn_config']:
        return WorkflowType.LEGACY
    
    # Check for 6-entrypoint indicators
    if has_nodes_and_transforms(blueprint):
        return WorkflowType.SIX_ENTRYPOINT
    
    raise ValueError("Cannot determine workflow type")
```

### 3. Entrypoint Configuration Extraction

The backend extracts and validates the entrypoint configuration:

```python
def _extract_entrypoint_config(self, combination: Dict[str, Any]) -> EntrypointConfig:
    config = EntrypointConfig(
        entrypoint_1=combination.get('entrypoint_1', []),
        entrypoint_2=combination.get('entrypoint_2', []),
        entrypoint_3=combination.get('entrypoint_3', []),
        entrypoint_4=combination.get('entrypoint_4', []),
        entrypoint_5=combination.get('entrypoint_5', []),
        entrypoint_6=combination.get('entrypoint_6', [])
    )
    
    # Handle legacy attribute names for backward compatibility
    if 'canonical_ops' in combination:
        config.entrypoint_1 = combination['canonical_ops']
    # ... etc
    
    return config
```

### 4. Transform Validation

Each transform name is validated against known FINN transforms:

```python
valid_transforms = {
    # General transforms
    'FoldConstants', 'InferShapes', 'RemoveUnusedNodes',
    'RemoveStaticGraphInputs', 'InferDataTypes',
    
    # Streamlining transforms
    'Streamline', 'ConvertBipolarToXnor', 'MoveLinearPastFork',
    'AbsorbSignBiasIntoMultiThreshold',
    
    # FPGA dataflow transforms
    'AnnotateCycles', 'SetFolding', 'MinimizeAccumulatorWidth',
    'MinimizeWeightBitWidth'
}
```

### 5. Subprocess Script Generation

A Python script is dynamically generated for subprocess execution:

```python
def _create_execution_script(self, script_path, model_path, entrypoint_config, work_dir):
    script_content = f'''
#!/usr/bin/env python3

from finn.builder.build_dataflow_v2 import build_dataflow_v2
from finn.transformation.general import *
from finn.transformation.streamline import *
from finn.transformation.fpgadataflow import *

# Transform mapping
transform_mapping = {{
    "FoldConstants": FoldConstants(),
    "InferShapes": InferShapes(),
    "Streamline": Streamline(),
    # ... etc
}}

# Convert string names to transform objects
entrypoint_transforms = {{
    1: get_transforms({entrypoint_config.entrypoint_1}),
    2: get_transforms({entrypoint_config.entrypoint_2}),
    # ... etc
}}

# Execute FINN v2 build
result = build_dataflow_v2(
    "{model_path}",
    entrypoint_transforms,
    output_dir="{work_dir}/finn_output",
    synth_clk_period_ns=5.0,
    fpga_part="xcu250-figd2104-2L-e"
)
'''
```

### 6. Subprocess Execution

The script is executed in an isolated subprocess:

```python
result = subprocess.run(
    ["python", str(script_path)],
    capture_output=True,
    text=True,
    timeout=timeout,
    cwd=work_dir
)
```

### 7. Result Processing

The subprocess writes results to a JSON file:

```json
{
    "success": true,
    "output_dir": "/tmp/finn_work/finn_output",
    "warnings": [],
    "reports": {
        "synthesis": "path/to/synthesis_report.txt",
        "utilization": "path/to/utilization_report.json"
    }
}
```

### 8. Metrics Extraction

The MetricsExtractor processes FINN outputs:

```python
metrics = {
    'throughput': 1000.0,      # fps
    'latency': 1.5,            # ms
    'lut_utilization': 0.65,   # 65%
    'dsp_utilization': 0.80,   # 80%
    'bram_utilization': 0.45,  # 45%
    'power_consumption': 12.5, # watts
    'resource_efficiency': 0.75
}
```

## Component Details

### SixEntrypointBackend Class

```python
class SixEntrypointBackend(EvaluationBackend):
    """
    Backend for modern 6-entrypoint FINN interface.
    
    Methods:
    - __init__(blueprint_config): Initialize with blueprint
    - evaluate(request): Main evaluation entry point
    - _extract_entrypoint_config(): Extract configuration
    - _execute_in_subprocess(): Run FINN in subprocess
    - _create_execution_script(): Generate execution script
    - _extract_metrics(): Extract performance metrics
    - validate_configuration(): Validate transform names
    """
```

### EntrypointConfig Dataclass

```python
@dataclass
class EntrypointConfig:
    """Configuration for 6-entrypoint execution."""
    entrypoint_1: List[str]  # After model loading
    entrypoint_2: List[str]  # After quantization setup
    entrypoint_3: List[str]  # After hardware kernel mapping
    entrypoint_4: List[str]  # After kernel specialization
    entrypoint_5: List[str]  # After kernel-level optimizations
    entrypoint_6: List[str]  # After graph-level optimizations
```

## Transform Mapping

### General Transforms (Entrypoint 1)
- `FoldConstants`: Fold constant operations
- `InferShapes`: Infer tensor shapes
- `RemoveUnusedNodes`: Remove dead nodes
- `RemoveStaticGraphInputs`: Remove static inputs
- `InferDataTypes`: Infer data types

### Streamlining Transforms (Entrypoint 2)
- `Streamline`: Apply all streamlining transforms
- `ConvertBipolarToXnor`: Convert bipolar to XNOR operations
- `MoveLinearPastFork`: Move linear ops past fork nodes
- `AbsorbSignBiasIntoMultiThreshold`: Absorb sign/bias

### Hardware Kernel Transforms (Entrypoint 3)
- Kernel names like `MatrixVectorActivation`, `ConvolutionInputGenerator`

### Kernel Specialization (Entrypoint 4)
- Specialization parameters for selected kernels

### Kernel-Level Optimizations (Entrypoint 5)
- `MinimizeAccumulatorWidth`: Optimize accumulator widths
- `MinimizeWeightBitWidth`: Optimize weight bit widths
- `SetFolding`: Set folding parameters

### Graph-Level Optimizations (Entrypoint 6)
- `AnnotateCycles`: Annotate cycle counts
- Graph-level transformations

## Subprocess Execution

### Why Subprocess?

1. **Isolation**: FINN crashes don't affect the main process
2. **Memory Management**: Clean memory state for each run
3. **Timeout Support**: Can kill long-running compilations
4. **Error Containment**: Exceptions are caught and reported

### Execution Environment

The subprocess inherits:
- Python environment with FINN installed
- Access to model files and work directory
- Environment variables for licensing/tools

### Communication Protocol

1. Input: Configuration via generated Python script
2. Output: JSON file with results
3. Logs: Captured via stdout/stderr

## Error Handling

### Transform Validation Errors
```python
if transform not in valid_transforms:
    errors.append(f"Unknown transform '{transform}' in {entrypoint}")
```

### Subprocess Failures
```python
if result.returncode != 0:
    return EvaluationResult(
        success=False,
        error=f'Process failed with code {result.returncode}',
        warnings=[result.stderr]
    )
```

### Timeout Handling
```python
except subprocess.TimeoutExpired:
    return EvaluationResult(
        success=False,
        error=f'Execution timed out after {timeout} seconds'
    )
```

## Integration Points

### 1. DSE Explorer Integration
- Receives `ComponentCombination` objects
- Returns standardized `EvaluationResult`

### 2. Blueprint Configuration
- Reads workflow type indicators
- Extracts FINN parameters

### 3. Metrics Extraction
- Reuses `MetricsExtractor` component
- Provides standardized metrics

### 4. FINN Integration
- Uses `build_dataflow_v2()` API
- Maps to FINN transform objects

## Example Walkthrough

### Step 1: DSE Creates Combination
```python
combination = {
    'entrypoint_1': ['FoldConstants', 'InferShapes'],
    'entrypoint_2': ['Streamline'],
    'entrypoint_3': ['MatrixVectorActivation'],
    'entrypoint_4': [],
    'entrypoint_5': ['MinimizeAccumulatorWidth'],
    'entrypoint_6': []
}
```

### Step 2: Backend Processes Request
```python
backend = SixEntrypointBackend(blueprint_config)
result = backend.evaluate(EvaluationRequest(
    model_path="bert.onnx",
    combination=combination,
    work_dir="/tmp/work"
))
```

### Step 3: Script Generation
```python
# Generated script maps transforms:
entrypoint_transforms = {
    1: [FoldConstants(), InferShapes()],
    2: [Streamline()],
    3: [MatrixVectorActivation()],
    4: [],
    5: [MinimizeAccumulatorWidth()],
    6: []
}
```

### Step 4: FINN Execution
```
[INFO] Starting FINN 6-entrypoint build
[INFO] Applying transforms at entrypoint 1: FoldConstants, InferShapes
[INFO] Applying transforms at entrypoint 2: Streamline
[INFO] Applying transforms at entrypoint 3: MatrixVectorActivation
[INFO] Applying transforms at entrypoint 5: MinimizeAccumulatorWidth
[INFO] Build completed successfully
```

### Step 5: Results
```python
EvaluationResult(
    success=True,
    metrics={
        'throughput': 850.0,
        'latency': 2.1,
        'lut_utilization': 0.72,
        'resource_efficiency': 0.68
    },
    reports={'synthesis': '...'},
    warnings=[]
)
```

## Advantages of 6-Entrypoint System

1. **Granular Control**: Insert specific transforms at specific points
2. **Modularity**: Transforms are independent and composable
3. **Flexibility**: Easy to experiment with different transform combinations
4. **Clean Separation**: No conflation of transforms and steps
5. **Future-Proof**: Aligns with modern FINN architecture

## Conclusion

The 6-entrypoint system provides a clean, modular interface for FINN compilation that separates individual transforms from pre-packaged build steps. Through subprocess isolation and careful transform mapping, it enables reliable and flexible neural network compilation for FPGA deployment.