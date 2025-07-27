# BrainSmith Core - Architecture Context Document

This document provides a comprehensive overview of the @brainsmith/core module architecture to enable rapid mental model reconstruction.

## Overview

BrainSmith implements a **segment-based execution tree architecture** for hardware synthesis. The system transforms neural network models through configurable pipelines defined in YAML blueprints, automatically exploring design variations while sharing common computation.

## Main API

The primary entry point for BrainSmith is the `forge()` function:

```python
from brainsmith import forge

# Transform a neural network into an FPGA accelerator
results = forge(
    model_path="model.onnx",
    blueprint_path="blueprint.yaml",
    output_dir="./build/my_accelerator"  # Optional, defaults to $BSMITH_BUILD_DIR/forge_*
)

# Access results
print(f"Successful builds: {results.stats['successful']}")
print(f"Total segments explored: {results.stats['total']}")
print(f"Build time: {results.total_time:.2f}s")
```

## Core Architecture Components

### 1. Blueprint System
**Files**: `blueprint_parser.py`, `design_space.py`

The blueprint system provides YAML-based configuration with inheritance:

```yaml
# Example blueprint structure
extends: base.yaml  # Inheritance
design_space:
  steps:
    - qonnx_to_finn
    - [cleanup_opt0, cleanup_opt1, ~]  # Branch point with skip option
    - streamline
  kernels:
    - MatMul
    - {Conv: [HLS, RTL]}  # Restricted backends
```

**Key Features**:
- **Inheritance**: Child blueprints extend parents via deep merge
- **Step Operations**: Surgical modifications (before/after/replace/remove)
- **Branch Points**: Lists create exploration variations
- **Skip Indicator**: "~" creates skip branches

**Configuration Structure**:
```python
@dataclass
class ForgeConfig:
    # Build control settings
    output_stage: OutputStage = OutputStage.COMPILE_AND_PACKAGE
    working_directory: str = "work"
    save_intermediate_models: bool = False
    fail_fast: bool = False
    max_combinations: int = ...
    timeout_minutes: int = ...
    
    # FINN-specific parameters (passed through without validation)
    finn_params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class DesignSpace:
    model_path: str
    steps: List[Union[str, List[Optional[str]]]]
    kernel_backends: List[Tuple[str, List[Type]]]
    config: ForgeConfig
```

### 2. Execution Tree Pattern
**Files**: `execution_tree.py`, `tree_builder.py`

The execution tree optimizes exploration through automatic prefix sharing:

```
root (0 steps)
├── cleanup_opt0 (3 steps)
│   ├── skip_0 (0 steps) [leaf]
│   └── optional_step (1 step)
│       └── final_steps (2 steps) [leaf]
└── cleanup_opt1 (3 steps)
    └── final_steps (2 steps) [leaf]
```

**ExecutionNode Structure**:
```python
@dataclass
class ExecutionNode:
    segment_steps: List[Dict[str, Any]]  # Steps in this segment only
    branch_decision: Optional[str]       # How we reached this node
    parent: Optional['ExecutionNode']    
    children: Dict[str, 'ExecutionNode'] # Branch ID → child node
    status: str                          # pending/running/completed/failed
    artifacts: List[ArtifactState]       # Shared artifacts from parent
    finn_config: Dict[str, Any]
```

**Key Concepts**:
- **Segments**: Continuous execution between branch points
- **Segment IDs**: Path-based identifiers (e.g., "cleanup_opt0/skip_0")
- **Efficiency**: Common prefixes execute once, branches share artifacts
- **Breadth-First Execution**: Parents complete before children

### 3. Forge API
**File**: `forge.py`

The forge API provides the main entry point for end-to-end FPGA accelerator synthesis:

```python
def forge(model_path: str, blueprint_path: str, output_dir: str = None) -> TreeExecutionResult:
    """
    Forge an FPGA accelerator from model and blueprint.
    
    Args:
        model_path: Path to ONNX model
        blueprint_path: Path to blueprint YAML
        output_dir: Output directory (defaults to $BSMITH_BUILD_DIR/forge_YYYYMMDD_HHMMSS)
        
    Returns:
        TreeExecutionResult with build artifacts and statistics
    """
```

The forge function:
1. Parses the blueprint and creates the design space
2. Builds the execution tree with automatic prefix sharing
3. Explores the tree using the explorer subsystem
4. Returns comprehensive results including all build artifacts

### 4. Explorer Subsystem
**Files**: `explorer/explorer.py`, `explorer/executor.py`, `explorer/finn_adapter.py`

The explorer orchestrates execution tree traversal and FINN integration:

**Internal Entry Point** (called by forge):
```python
def explore_execution_tree(
    tree: ExecutionNode,
    input_model: Path,
    output_dir: Path,
    blueprint_config: Dict[str, Any],
    design_space: Optional[DesignSpace] = None
) -> TreeExecutionResult
```

**Execution Flow**:
1. Depth-first traversal using stack
2. Check for cached results (existing output models)
3. Execute segments via FINN builds
4. Share artifacts at branch points
5. Handle failures (skip descendants or fail-fast)

**Data Structures**:
```python
@dataclass
class SegmentResult:
    success: bool
    segment_id: str
    output_model: Optional[Path]
    output_dir: Optional[Path]
    error: Optional[str]
    execution_time: float
    cached: bool

@dataclass 
class TreeExecutionResult:
    segment_results: Dict[str, SegmentResult]
    total_time: float
```

**FINNAdapter**:
- Isolates FINN-specific workarounds (working directory changes, model discovery)
- Maps blueprint config to FINN's DataflowBuildConfig
- Handles output product selection (df/rtl/dcp stages)

### 5. Plugin Registry
**Files**: `plugins/registry.py`, `plugins/framework_adapters.py`

Unified plugin system managing components across frameworks:

**Plugin Types**:
- **Transforms**: Graph optimizations and modifications
- **Kernels**: Hardware implementation units (HWCustomOp)
- **Backends**: Code generators (HLS/RTL)
- **Steps**: Build pipeline stages

**Registry Pattern**:
```python
# Registration via decorators
@transform(name="custom_name", metadata={...})
class MyTransform:
    pass

# Usage
transform = get_transform("Streamline")  # Auto-resolves framework
transform = get_transform("finn:MVAU")   # Explicit framework
```

**Framework Support**:
- QONNX: 60 transforms
- FINN: 98 transforms, 40 kernels, 40 backends, 19 steps
- BrainSmith: Custom extensions

**Key Features**:
- Singleton registry with framework namespacing
- Metadata-driven discovery
- Fail-fast philosophy (KeyError on missing plugins)
- Lazy loading of external plugins

### 6. Core Utilities
**Files**: `utils.py`, `validation.py`, `yaml_utils.py`, `time_utils.py`

**Transform Pipeline** (`utils.py`):
```python
def apply_transforms(model, transform_list, debug_path=None):
    # Sequential application with optional debug saves
    for transform_name in transform_list:
        transform = get_transform(transform_name)
        model = transform.apply(model)
    return model
```

**Validation Layers** (`validation.py`):
- Step existence in registry
- FINN config requirements
- Kernel-backend compatibility

**YAML Inheritance** (`yaml_utils.py`):
- Recursive loading with relative path resolution
- Deep merge of configurations
- Parent-child override semantics

## Data Flow

```
Model + Blueprint → forge() → Parse & Validate → DesignSpace → Build ExecutionTree
                                                                        ↓
                                                           Explore (segment by segment)
                                                                        ↓
                                              Transform Pipeline → Output Model (per segment)
                                                                        ↓
                                                             Artifact Sharing at Branches
                                                                        ↓
                                                              TreeExecutionResult
```

## Key Design Principles

1. **Arete Philosophy**: Crystalline clarity in every line
2. **Segment Efficiency**: Minimize redundant computation
3. **Fail-Fast**: Explicit errors over silent failures
4. **Plugin Architecture**: Extensible without core changes
5. **Clean Separation**: Framework adapters isolate external dependencies

## Critical Code Paths

### 1. End-to-End Synthesis
```
forge() → BlueprintParser.parse() → TreeBuilder.build() → explore_execution_tree() → TreeExecutionResult
```

### 2. Tree Exploration
```
explore_execution_tree() → Executor.execute() → _execute_segment() → FINNAdapter
```

### 3. Plugin Resolution
```
get_transform("name") → Registry.get() → framework resolution → return class
```

## Common Patterns

1. **Lazy Imports**: Avoid circular dependencies
2. **Debug Support**: Save intermediate models
3. **Caching**: Check for existing outputs before rebuilding
4. **Metadata Queries**: Find plugins by attributes
5. **Breadth-First**: Ensure parent completion before children

## Extension Points

1. **Custom Transforms**: Add via `@transform` decorator
2. **New Backends**: Register with `@backend` and kernel metadata
3. **Build Steps**: Add via `@step` for new pipeline stages
4. **Blueprint Operations**: Extend step modification syntax

This architecture enables efficient exploration of hardware synthesis design spaces while maintaining clarity and extensibility throughout the system.