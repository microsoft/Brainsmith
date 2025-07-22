# Execution Tree Explorer Design

## Overview

The Execution Tree Explorer traverses the execution tree structure and executes FINN builds for complete paths. It maximizes computational efficiency by reusing shared prefixes and implements a hierarchical caching system based on directory structure.

## Architecture Components

### 1. **TreeExecutor** (Main Orchestrator)
- Manages depth-first traversal of the execution tree
- Coordinates path execution and result aggregation
- Handles global configuration and execution policies
- Manages checkpoint/resume functionality

### 2. **PathExecutor** (Path-to-FINN Runner)
- Executes FINN build for path segments (between branch points)
- A segment is a linear sequence of nodes without branches
- Executes up to the next branch point, then pauses
- Copies artifacts at branch points for reuse by other paths
- Manages segment-specific directories and artifacts
- Handles pre/post execution hooks

### 3. **ConfigMapper** (Configuration Translation)
- Translates execution path → DataflowBuildConfig
- Accumulates steps from all nodes in path
- Handles blueprint finn_config overrides
- Manages sensible FINN defaults

### 4. **ExecutionContext** (State Management)
- Tracks execution state across the tree
- Manages path relationships and sharing
- Handles failure propagation
- Provides simple progress reporting

## Key Design Decisions

### Segment-Based Execution with Branch-Point Sharing
- Execute path segments between branch points as single FINN builds
- Non-branching sequences become single DataflowBuildConfig
- **Critical optimization**: Execute up to branch points, then share
  - Example: For tree `root→A→B→{C,D}`, first execute `root→A→B`
  - At branch point B, copy artifacts for both C and D branches
  - Then execute `B→C` and `B→D` separately, starting from B's outputs
- Ensures only relevant artifacts are copied (not entire paths)
- Maximizes both FINN's internal optimizations and cross-path sharing
- Dramatically reduces redundant computation

### Directory-Based Caching
- Hierarchical directory structure mirrors execution tree
- Each unique path gets its own output directory
- Cache detection via checking directory existence
- No in-memory caching complexity (future enhancement)

### Simple Progress Tracking
- Print progress messages as tree is traversed
- Report completed paths and remaining work
- No complex ETA calculations
- Clear, actionable output

## Directory Structure

```
output_root/
├── execution_tree.json          # Serialized tree for visualization
├── global_config.json           # Global execution settings
├── execution_log.txt            # Simple execution log
└── paths/
    ├── root_stage1_transform1/  # Complete path directory
    │   ├── path_config.json     # Path configuration
    │   ├── finn_config.json     # Generated FINN config
    │   ├── model.onnx           # Input model
    │   ├── build/               # FINN build outputs
    │   ├── status.json          # Execution status
    │   └── report/              # FINN estimate reports
    └── root_stage1_transform2/  # Shares prefix with transform1
        ├── shared_from.json     # References root_stage1_transform1
        ├── path_config.json     # Only the delta configuration
        ├── finn_config.json     # Only new steps (transform2)
        └── build/               # Only outputs from new steps
```

### Branch-Point Execution Example

When executing a tree with branches:
1. Execute `root→stage1` (up to branch point)
2. At `stage1` branch point, copy artifacts to both branch directories
3. Execute `stage1→transform1` using copied artifacts
4. Execute `stage1→transform2` using copied artifacts
5. Continue recursively for any sub-branches

## Configuration Flow

```
Execution Path → Node collection → Step accumulation → DataflowBuildConfig
                                           ↓
                                Blueprint overrides applied
                                           ↓
                                    FINN execution
```

### Default FINN Settings
```python
defaults = {
    # Required settings - must be provided via global_config or finn_config
    "synth_clk_period_ns": None,      # No default - must be specified
    "board": None,                    # No default - must be specified
    
    # Optional settings with sensible defaults
    "save_intermediate_models": True,  # For debugging
    "auto_fifo_depths": True,         # FINN optimization
    "generate_outputs": [ESTIMATE_REPORTS],  # Start simple
}
```

### Blueprint Override System
```yaml
# In blueprint YAML
finn_config:
  synth_clk_period_ns: 5.0
  board: "U250"
  target_fps: 1000
  # Any DataflowBuildConfig field...
```

## Execution Strategy

### Key Goals
- Execute linear segments (non-branching sequences) as single FINN builds
- Share build artifacts at branch points to avoid redundant computation
- Maintain depth-first traversal for quick results
- Support caching and incremental execution

### Execution Flow
1. **Identify segments**: Linear sequences of nodes between branch points
2. **Execute segments**: Each segment becomes one FINN build
3. **Share at branches**: Copy artifacts when paths diverge
4. **Continue depth-first**: Complete one path before exploring siblings

### Critical Considerations
- Segments can span multiple nodes (e.g., `root→A→B→C` with no branches)
- Branch points are where artifacts must be copied for reuse
- Each complete path (root to leaf) has its own output directory
- Caching checks happen at segment level, not individual nodes

## Implementation Plan

### Phase 1: Foundation (Core Infrastructure)
**Goal**: Basic execution tree traversal with FINN integration

#### Tasks:
1. **Create explorer module structure**
   - `brainsmith/explorer/` directory
   - Core data structures (PathResult, ExecutionContext)
   - Configuration classes (ExplorerConfig)

2. **Implement ConfigMapper**
   - Path to FINN steps collection
   - Default FINN configuration
   - Blueprint override system

3. **Implement PathExecutor**
   - Execute complete path as single FINN build
   - Capture outputs and errors
   - Status tracking to disk

4. **Implement TreeExecutor**
   - Depth-first traversal algorithm
   - Path generation from tree
   - Basic error handling

**Deliverable**: Can execute a simple execution tree end-to-end

### Phase 2: Caching & Persistence
**Goal**: Directory-based caching with checkpoint/resume

#### Tasks:
1. **Implement directory-based caching**
   - Path-to-directory mapping
   - Cache detection via filesystem
   - Status file management

2. **Add checkpoint system**
   - Save tree traversal state
   - Mark completed paths
   - Simple checkpoint format

3. **Implement resume capability**
   - Load checkpoint state
   - Skip completed paths
   - Continue from interruption

4. **Artifact organization**
   - Organize FINN outputs
   - Copy key results
   - Clean temporary files

**Deliverable**: Can resume interrupted executions and skip cached paths

### Phase 3: Robustness & Integration
**Goal**: Solid API with comprehensive error handling

#### Tasks:
1. **Enhanced error handling**
   - Failure propagation logic
   - Continue vs fail-fast modes
   - Clear error reporting

2. **Hook system**
   - Pre/post path execution hooks
   - Simple event callbacks
   - Extension points

3. **API design**
   - Clean public interface
   - Explorer entry points
   - Configuration validation

4. **Functional validation**
   - Unit tests for components
   - Integration test suite
   - Example blueprints
   - Edge case handling

**Deliverable**: Well-tested explorer with clean API

### Phase 4: Documentation & Future Work
**Goal**: Complete documentation and future roadmap

#### Tasks:
1. **User documentation**
   - Usage guide
   - Configuration reference
   - Example workflows

2. **Developer documentation**
   - Architecture details
   - Extension guide
   - API reference

3. **Future work document**
   - Parallel execution design
   - Intelligent search strategies
   - Advanced caching mechanisms
   - Performance optimizations

**Deliverable**: Fully documented system with clear extension path

## API Design

### Main Entry Point
```python
from brainsmith.explorer import explore_execution_tree

results = explore_execution_tree(
    tree=execution_tree,
    output_dir="exploration_output",
    config=ExplorerConfig(
        fail_fast=False,
        resume_checkpoint="checkpoint.json",
        finn_overrides={...}
    )
)
```

### Explorer Configuration
```python
@dataclass
class ExplorerConfig:
    fail_fast: bool = False              # Stop on first failure
    resume_checkpoint: Optional[str] = None  # Resume from checkpoint
    finn_overrides: Dict = field(default_factory=dict)  # FINN config overrides
    pre_execution_hook: Optional[Callable] = None
    post_execution_hook: Optional[Callable] = None
```

### Path Result
```python
@dataclass
class PathResult:
    path_name: str
    status: ExecutionStatus  # SUCCESS, FAILED, CACHED, SKIPPED
    output_dir: str
    error: Optional[str] = None
    metrics: Optional[Dict] = None
    execution_time: Optional[float] = None
```

## Extension Points

The design provides several extension points for future enhancements:

1. **Custom hooks** - Pre/post execution logic
2. **Result analysis** - Metrics extraction and comparison
3. **Path prioritization** - Smart traversal ordering
4. **Caching strategies** - Advanced cache invalidation
5. **Parallel execution** - Multi-process path execution

These extension points ensure the explorer can grow with future needs while maintaining a simple, solid foundation.