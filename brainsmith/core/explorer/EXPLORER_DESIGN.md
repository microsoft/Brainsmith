# Segment Executor V2 Design

## Overview

The Segment Executor V2 implements a streamlined approach to executing segment-based execution trees with FINN. It emphasizes simplicity, clarity, and proper isolation of FINN-specific workarounds.

## Core Architecture

### 1. Minimal Data Structures (types.py)
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
    
    @property
    def stats(self) -> Dict[str, int]:
        # Calculate execution statistics
```

### 2. FINN Adapter (finn_adapter.py)
**Purpose**: Isolate all FINN-specific workarounds in a single, well-documented class.

**Responsibilities**:
- Handle working directory changes (os.chdir)
- Discover output models in intermediate_models
- Manage dynamic FINN imports
- Convert configuration dictionaries to DataflowBuildConfig

**Key Methods**:
- `build()`: Execute FINN build with all necessary workarounds
- `_discover_output_model()`: Find output in unpredictable locations
- `prepare_model()`: Copy models to avoid in-place corruption

### 3. Unified Executor (executor.py)
**Purpose**: Single class handling both tree traversal and segment execution.

**Features**:
- Stack-based iteration (cleaner than recursion)
- Simple caching based on output file existence
- Deterministic output naming: `{segment_id}_output.onnx`
- Fail-fast mode toggle
- Artifact sharing at branch points
- Clear skip propagation for failed segments

**Key Methods**:
- `execute()`: Main entry point for tree execution
- `_execute_segment()`: Execute single segment using FINNAdapter
- `_make_finn_config()`: Build FINN configuration for segment

### 4. Entry Point (explorer.py)
**Purpose**: Clean API for external users.

**Responsibilities**:
- Create FINNAdapter and Executor instances
- Parse blueprint configuration
- Save tree structure and results
- Provide simple explore_execution_tree() function

### 5. Utilities (utils.py)
**Purpose**: Helper functions for serialization and artifact management.

**Key Functions**:
- `StageWrapperFactory`: Pre-compute transform wrappers with simple indices
- `serialize_tree()`: Convert execution tree to JSON
- `serialize_results()`: Convert results to JSON
- `share_artifacts_at_branch()`: Copy artifacts for child segments

## Key Design Decisions

### 1. Adapter Pattern
All FINN-specific workarounds are isolated in FINNAdapter, making it easy to:
- Document why each workaround exists
- Update when FINN improves
- Test FINN interactions separately
- Keep main executor code clean

### 2. Unified Execution
Merging TreeExecutor and SegmentExecutor into one class:
- Reduces code duplication
- Simplifies the mental model
- Makes data flow clearer
- Eliminates coordination complexity

### 3. Pre-computed Wrappers
Transform stages are wrapped during tree building:
- No dynamic wrapper generation during execution
- Simple numeric indices (cleanup_0, cleanup_1)
- Stored in segment steps as finn_step_name
- No global registry mutations

### 4. Simple Caching
Output file existence determines cache status:
- No complex state management
- Filesystem as source of truth
- Easy to manually clear cache
- Works across process restarts

### 5. Deterministic Paths
Predictable output locations:
- `{output_dir}/{segment_id}/{safe_name}_output.onnx`
- Segment IDs with slashes converted to underscores
- Consistent naming across executions

## Necessary Evils

All isolated in FINNAdapter with clear TODO comments:

### 1. Working Directory Changes
```python
# NECESSARY EVIL: FINN requires working directory change
# TODO: Fix FINN to accept absolute paths properly
os.chdir(output_dir)
```

### 2. Model Discovery
```python
# NECESSARY EVIL: FINN doesn't return output path
# We must discover it in intermediate_models directory
```

### 3. Full Directory Copying
```python
# NECESSARY EVIL: Full copy required by FINN
# TODO: Use symlinks when FINN is more robust
shutil.copytree(parent_result.output_dir, child_dir)
```

## Directory Structure

```
output_dir/
├── tree.json                    # Serialized execution tree
├── summary.json                 # Execution results summary
├── root/                        # Root segment
│   ├── input.onnx
│   ├── root_output.onnx
│   └── intermediate_models/
├── cleanup_opt0/                # First branch
│   ├── input.onnx
│   ├── cleanup_opt0_output.onnx
│   └── intermediate_models/
└── cleanup_opt0/optional_step_opt0/  # Nested segment
    ├── input.onnx
    ├── cleanup_opt0_optional_step_opt0_output.onnx
    └── intermediate_models/
```

## API Usage

```python
from brainsmith.core.blueprint_parser import BlueprintParser
from brainsmith.core.explorer import explore_execution_tree

# Parse blueprint and build tree
parser = BlueprintParser()
design_space, tree = parser.parse("blueprint.yaml", "model.onnx")

# Execute exploration
result = explore_execution_tree(
    tree=tree,
    model_path="model.onnx",
    output_dir="output",
    blueprint_config={
        "global_config": {
            "fail_fast": False,
            "output_products": "df"
        },
        "finn_config": {
            "synth_clk_period_ns": 5.0,
            "board": "U250"
        }
    }
)

# Check results
print(f"Total segments: {result.stats['total']}")
print(f"Successful: {result.stats['successful']}")
print(f"Failed: {result.stats['failed']}")
```

## Future Improvements

1. **Parallel Execution**: Execute independent branches concurrently
2. **Incremental Builds**: Detect changes and rebuild only affected segments
3. **Remote Execution**: Distribute segments across multiple machines
4. **Advanced Caching**: Content-based cache keys, not just file existence
5. **Progress Monitoring**: Real-time web UI for execution status

These improvements can be added without changing the core architecture, demonstrating the design's extensibility.