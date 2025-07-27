# Execution Tree Explorer Context

## Background

The Execution Tree Explorer is part of BrainSmith's new architecture that replaces the previous Phase 2 combination-based exploration. The execution tree structure was introduced to solve the combinatorial explosion problem in design space exploration.

## Integration with BrainSmith Ecosystem

### Execution Tree Structure
- Created by `brainsmith.core.tree_builder.build_execution_tree()`
- Nodes represent FINN build steps or stages
- Tree structure naturally encodes all possible design variations
- Automatic prefix sharing reduces redundancy by 40-60%

### Blueprint System
- Blueprints define the design space in YAML format
- Support inheritance via `extends` field
- Define transform stages, kernel backends, and build pipelines
- The `finn_config` section provides direct FINN overrides

### Plugin Architecture
- All transforms, kernels, and build steps are registered plugins
- Registry provides O(1) lookup with pre-computed indexes
- Plugins are resolved at blueprint parsing time

## FINN Integration Details

### FINN Builder System
- FINN's `build_dataflow_cfg()` is the main entry point
- Expects a DataflowBuildConfig object with all settings
- Executes a pipeline of transformation steps
- Produces various outputs (estimates, RTL, bitfiles, etc.)

### Key FINN Concepts
- **Build Steps**: Transformations applied to the ONNX model
- **Folding**: Parallelization configuration for hardware
- **Board Targets**: Specific FPGA platforms (Pynq-Z1, U250, etc.)
- **Output Products**: Different artifacts (estimates, synthesis results, drivers)

## Design Constraints

### Current Scope
- Sequential depth-first execution only (parallelism is future work)
- Exhaustive exploration only (no early pruning)
- FINN-exclusive (no other compiler backends)
- Focus on systematic execution over intelligent search

### Simplification Decisions
- No runtime metrics analysis (future work)
- No progress estimation (just status reporting)
- Directory-based caching only (no in-memory cache)
- Minimal hooks (just pre/post execution)

## Technical Considerations

### Memory and Resource Usage
- FINN builds can be memory-intensive (multiple GB per build)
- Builds can take significant time (minutes to hours)
- Disk space grows with tree size and intermediate models
- CPU usage varies by synthesis complexity

### Error Scenarios
- FINN build failures (synthesis errors, resource constraints)
- Missing dependencies (Vivado, board files)
- Disk space exhaustion
- Invalid configurations

## Future Enhancement Opportunities

### Parallel Execution
- Branch points are natural parallelization boundaries
- Process pool for concurrent segment execution
- Resource-aware scheduling

### Intelligent Search
- Early termination based on estimates
- Priority-based traversal
- ML-guided exploration

### Advanced Caching
- Content-based deduplication
- Distributed cache sharing
- Incremental synthesis

### Integration Points
- Real-time monitoring dashboards
- CI/CD integration
- Cloud execution backends
- Results database

## Related Components

### Existing Phase 3
- `LegacyFINNBackend` provides FINN integration patterns
- Build runner pattern demonstrates hooks and lifecycle
- Note: Existing metrics extraction may need revision

### Tree Building
- `DesignSpace` is the intermediate representation
- `ExecutionNode` provides the tree structure
- `TransformStage` handles variation points

## Development Guidelines

### Testing Strategy
- Start with simple linear trees
- Test branch point handling separately
- Validate caching behavior
- Ensure failure propagation works correctly

### Debugging Tips
- FINN saves intermediate models for inspection
- Each segment has its own log file
- Tree can be serialized for visualization
- Status files track execution state

### Performance Considerations
- Minimize artifact copying at branch points
- Use hard links where possible
- Clean up intermediate files promptly
- Monitor disk usage during large explorations