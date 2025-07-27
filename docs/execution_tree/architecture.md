# Brainsmith Core Architecture Analysis

## Overview

This document provides a comprehensive analysis of the @brainsmith/core module, documenting the architecture, design patterns, and key components of the Brainsmith hardware design space exploration system.

## Core Abstractions and Their Relationships

### 1. ExecutionTree - The Central Data Structure

- **Purpose**: Represents all possible design variations as a tree where nodes are execution segments
- **Key Innovation**: Segment-based architecture where each node represents a contiguous sequence of steps between branch points
- **Benefits**: 
  - Automatic prefix sharing between execution paths
  - Natural representation of the design space exploration problem
  - Caching and reuse of intermediate results

### 2. DesignSpace - The Intermediate Representation

- **Purpose**: Clean representation between blueprint YAML and execution tree
- **Contains**:
  - `transform_stages`: Dictionary of TransformStage objects
  - `kernel_backends`: List of (kernel_name, backend_classes) tuples
  - `build_pipeline`: List of steps to execute
  - `global_config`: Execution parameters
  - `finn_config`: FINN-specific configuration
- **Key Feature**: All plugin names are resolved to actual classes at this stage

### 3. BlueprintParser - The Configuration Loader

- **Purpose**: Parse YAML blueprints into DesignSpace objects
- **Features**:
  - Blueprint inheritance via `extends` field
  - Smart parameter mapping (e.g., `platform` → `board`, `target_clk` → `synth_clk_period_ns`)
  - Validation of pipeline references and required fields
  - Pre-computation of transform stage wrappers during tree building

### 4. Plugin System - The Extension Mechanism

- **Architecture**: Registry-based with direct lookup
- **Plugin Types**:
  - **Transforms**: ONNX model transformations (QONNX/FINN)
  - **Kernels**: Hardware operation types
  - **Backends**: Implementation options for kernels
  - **Steps**: Build pipeline operations
- **Key Features**:
  - Decoration-time registration (no discovery overhead)
  - Universal framework support
  - Pre-computed indexes for efficient queries

## Major Workflows and Data Flows

### 1. Blueprint to Execution Tree Flow

```
YAML Blueprint → BlueprintParser → DesignSpace → Tree Builder → ExecutionTree
                                         ↓
                                   Plugin Registry
                                   (resolves names)
```

### 2. Tree Execution Flow

```
ExecutionTree → Explorer → Executor → FINNAdapter → FINN Build System
                              ↓
                        SegmentResults → TreeExecutionResult
```

### 3. Segment Execution with Caching

```
Parent Segment → Check Cache → Execute FINN Build → Share Artifacts → Child Segments
                      ↓                                      ↓
                 (if cached)                          (copy to children)
                  Use Result
```

## Integration Points Between Components

### 1. Plugin Registry Integration

- BlueprintParser uses registry to resolve transform/backend names
- StageWrapperFactory creates FINN-compatible wrappers for transform stages
- Framework adapters register QONNX/FINN transforms directly

### 2. Execution Tree Integration

- Tree nodes store segment steps with pre-computed wrapper names
- Explorer traverses tree depth-first using stack-based iteration
- Artifact sharing happens at branch points automatically

### 3. FINN Integration

- FINNAdapter isolates all FINN-specific workarounds
- Handles working directory changes, model discovery, configuration conversion
- Each segment executes as a complete FINN build

## Design Patterns Used

### 1. Adapter Pattern (FINNAdapter)

- Isolates external system complexities
- Documents necessary workarounds with TODO comments
- Provides clean interface to executor

### 2. Registry Pattern (Plugin System)

- Centralized plugin management
- Supports multiple access patterns
- Enables framework qualification

### 3. Builder Pattern (Tree Construction)

- Step-by-step tree construction from design space
- Pre-computes wrappers during building
- Validates size constraints

### 4. Composite Pattern (ExecutionTree)

- Tree structure with uniform node interface
- Recursive operations (get_path, count_descendants)
- Natural representation of hierarchical data

## The Role of Explorer and Executor

### Explorer (`explore_execution_tree`)

- Entry point for tree execution
- Coordinates components (FINNAdapter, Executor)
- Saves tree structure and results as JSON

### Executor

- **Unified Design**: Single class handles both traversal and execution
- **Key Responsibilities**:
  - Stack-based tree traversal (cleaner than recursion)
  - Simple file-based caching
  - Fail-fast mode support
  - Skip propagation for failed segments
  - Artifact sharing at branch points
- **Execution Strategy**:
  - Depth-first traversal ensures parents execute before children
  - Each segment gets its own output directory
  - Deterministic naming: `{segment_id}_output.onnx`

## How FINN Integration Works

### 1. Configuration Flow

```
Blueprint finn_config → Executor._make_finn_config() → FINNAdapter.build()
                                    ↓
                            DataflowBuildConfig
```

### 2. FINNAdapter - Isolation of External System Complexity

The FINNAdapter encapsulates all FINN-specific workarounds and quirks:

**Necessary Evils (with TODOs for future FINN improvements):**
- **Working Directory Changes**: FINN requires `os.chdir()` to build directory
- **Model Discovery**: Output models appear in unpredictable `intermediate_models/` subdirectories
- **Model Corruption**: FINN modifies models in-place, requiring defensive copying
- **Dynamic Imports**: FINN modules must be imported at runtime to avoid circular dependencies
- **Config Conversion**: Dictionary configs must be converted to `DataflowBuildConfig` objects

**Key Methods:**
- `build()`: Main entry point that orchestrates the FINN build
- `_run_build()`: Handles directory changes and actual FINN execution
- `_find_output_model()`: Searches for the output model in FINN's directory structure

### 3. Build Execution Details

```python
# Simplified FINNAdapter workflow
def build(model_path, build_dir, config):
    # 1. Setup build environment
    os.makedirs(build_dir)
    shutil.copy2(model_path, build_dir)
    
    # 2. Change to build directory (FINN requirement)
    original_dir = os.getcwd()
    os.chdir(build_dir)
    
    # 3. Execute FINN build
    output_path = dataflow_build(model, config)
    
    # 4. Find and copy output model
    output_model = find_in_intermediate_models()
    
    # 5. Restore directory
    os.chdir(original_dir)
    return output_model
```

### 4. Artifact Management

- Each segment has isolated directory: `{output_dir}/segment_{id}/`
- Parent results copied to children at branch points (full directory copy due to FINN limitations)
- Deterministic output naming: `{segment_id}_output.onnx`

### 5. Data Structures

**SegmentResult:**
```python
@dataclass
class SegmentResult:
    segment_id: int
    success: bool
    output_model: Optional[str]  # Path to output ONNX
    output_dir: str               # Full build directory
    error: Optional[str]
    duration: float
    cached: bool
```

**TreeExecutionResult:**
```python
@dataclass  
class TreeExecutionResult:
    tree: ExecutionTree
    segment_results: Dict[int, SegmentResult]
    total_duration: float
    success_rate: float
```

## Key Architectural Decisions

### 1. Segment-Based Execution

- Segments reduce redundancy vs. path-based execution
- Natural caching boundaries
- Efficient artifact sharing

### 2. Pre-computed Wrappers

- Transform stages wrapped during tree building
- Simple numeric indices (e.g., `cleanup_0`, `cleanup_1`)
- No runtime wrapper generation

### 3. Unified Executor Design

- Single class instead of separate TreeExecutor/SegmentExecutor
- Reduces complexity and duplication
- Clear data flow

### 4. Adapter Isolation

- All FINN workarounds in one place
- Clean executor code
- Easy to update when FINN improves

## Performance Characteristics

### Computational Complexity
- **Tree Building**: O(n) where n = number of segments
- **Plugin Lookup**: Direct through registry
- **Execution**: Linear in number of segments (parallelism is future work)
- **Caching Check**: File existence check per segment

### Resource Usage

**Memory Consumption:**
- **Per Segment**: 1-3 GB during FINN build execution
- **Tree Storage**: ~100 bytes per node (minimal)
- **Model Storage**: Input model size × number of segments (worst case)
- **Peak Usage**: Can reach 10+ GB for complex builds

**Disk Space:**
- **Per Segment**: 500 MB - 2 GB (FINN intermediate files)
- **Growth Pattern**: Linear with number of segments
- **Cleanup**: Manual cleanup required (future enhancement)

**Build Times:**
- **Per Segment**: 30 seconds to 10+ minutes
- **Full Tree**: Minutes to hours depending on size
- **Caching Impact**: Cached segments skip FINN build entirely

**CPU Usage:**
- **FINN Builds**: CPU-intensive synthesis operations
- **Parallelism**: Currently sequential (single core)
- **Future**: Natural parallelism at branch points

### Optimization Strategies

1. **Segment Sharing**: Avoids redundant computation through prefix sharing
2. **File-Based Caching**: Survives process restarts
3. **Fail-Fast Mode**: Stops exploration on first failure
4. **Skip Propagation**: Avoids executing children of failed segments

## Architecture Summary

This architecture achieves a clean separation of concerns while efficiently handling the combinatorial explosion problem in hardware design space exploration. The segment-based execution tree is the key innovation that enables practical exploration of large design spaces.

### Core Architecture Flow

```
Blueprint YAML → Parser → DesignSpace → ExecutionTree → Explorer → Results
                    ↓           ↓            ↓              ↓
              Plugin Registry  Resolved   Segments    FINN Builds
                              Classes    (optimized)
```

### Key Insights

1. **Segment-Based Design**: The ExecutionTree uses segments (contiguous sequences between branch points) instead of full paths, sharing common prefixes automatically.

2. **Clean Abstraction Layers**:
   - **Blueprint**: Human-readable YAML configuration
   - **DesignSpace**: Intermediate representation with resolved plugins
   - **ExecutionTree**: Optimized execution structure
   - **Explorer/Executor**: Execution orchestration with caching

3. **FINN Integration Pattern**: The FINNAdapter isolates all FINN-specific quirks (directory changes, model discovery, config conversion) behind a clean interface.

4. **Plugin Architecture**: Registry-based system supporting transforms, kernels, backends, and steps with universal framework support.

5. **Execution Strategy**: Depth-first traversal with file-based caching, fail-fast support, and automatic artifact sharing at branch points.

The architecture elegantly solves the combinatorial explosion problem in hardware design space exploration through intelligent tree structuring and execution optimization.