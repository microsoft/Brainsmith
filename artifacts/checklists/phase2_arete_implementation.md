# Phase 2 Arete Implementation Checklist

## Goal: Transform Phase 2 into a tree-based explorer that leverages FINN's linear pipeline

## Phase A: Core Data Structures (1 hour)

### A1. Create Build Tree Classes
- [ ] Create `brainsmith/core/phase2/build_tree.py`
- [ ] Implement `BuildNode` class:
  ```python
  @dataclass
  class BuildNode:
      step_name: str
      config: Dict[str, Any]
      parent: Optional['BuildNode']
      children: List['BuildNode']
      output_dir: Optional[Path]
      metrics: Optional[BuildMetrics]
  ```
- [ ] Implement `BuildTree` class with methods:
  - [ ] `from_design_space(design_space: DesignSpace) -> BuildNode`
  - [ ] `get_leaf_nodes() -> List[BuildNode]`
  - [ ] `get_branch_points() -> List[BuildNode]`

### A2. Create Build Path Representation
- [ ] Implement `BuildPath` class:
  ```python
  @dataclass
  class BuildPath:
      steps: List[Tuple[str, Dict[str, Any]]]
      
      def common_prefix_length(self, other: 'BuildPath') -> int
      def divergence_point(self, other: 'BuildPath') -> Optional[int]
  ```

## Phase B: Tree Construction (2 hours)

### B1. Implement Tree Builder
- [ ] Create `_extract_build_paths()` to get all paths through design space
- [ ] Create `_merge_paths_into_tree()` to build optimal tree
- [ ] Handle transform stage combinations
- [ ] Handle kernel combinations
- [ ] Ensure deterministic ordering for reproducibility

### B2. Optimize Tree Structure
- [ ] Identify common prefixes automatically
- [ ] Merge identical subtrees
- [ ] Add tree statistics (depth, branching factor, savings)

## Phase C: FINN Integration (2 hours)

### C1. Create FINN-Aware Build Runner
- [ ] Create `brainsmith/core/phase2/finn_build_runner.py`
- [ ] Implement directory copying for shared prefixes
- [ ] Implement FINN command generation with start/stop steps
- [ ] Add proper error handling and recovery

### C2. Implement Tree Execution
- [ ] Create `_execute_node()` method:
  ```python
  def _execute_node(self, node: BuildNode, model_path: str):
      if node.parent and node.parent.output_dir:
          # Copy parent output
          shutil.copytree(node.parent.output_dir, node.output_dir)
          start_step = node.step_name
      else:
          start_step = None
      
      # Run FINN
      run_build_dataflow(
          model_path=model_path,
          output_dir=node.output_dir,
          start_step=start_step,
          stop_step=node.step_name,
          **node.config
      )
  ```

## Phase D: Parallel Execution (1 hour)

### D1. Add Parallel Node Execution
- [ ] Identify parallelizable nodes (siblings)
- [ ] Implement thread pool executor
- [ ] Add resource limits (max parallel builds)
- [ ] Handle failures gracefully

### D2. Progress Tracking
- [ ] Track nodes: pending, running, completed, failed
- [ ] Show tree visualization during execution
- [ ] Estimate time remaining based on completed nodes

## Phase E: Simplify Existing Code (1 hour)

### E1. Remove Over-Engineering
- [ ] Delete `SearchStrategy` enum and related code
- [ ] Delete `SearchConstraint` and `_satisfies_constraints()`
- [ ] Remove unused hooks (keep only essential 3)
- [ ] Inline `ProgressTracker` into main explorer
- [ ] Simplify design space ID generation

### E2. Update Interfaces
- [ ] Update `ExplorerEngine` to `TreeExplorer`
- [ ] Simplify `BuildConfiguration` generation
- [ ] Update result collection from leaf nodes

## Phase F: Testing & Documentation (1 hour)

### F1. Update Tests
- [ ] Create tree construction tests
- [ ] Test common prefix detection
- [ ] Test parallel execution
- [ ] Test failure recovery

### F2. Documentation
- [ ] Document tree-based approach
- [ ] Add examples of savings
- [ ] Show how to visualize execution tree

## Validation Steps

### 1. Correctness
```python
# Traditional flat exploration
results_flat = FlatExplorer().explore(design_space)

# New tree exploration  
results_tree = TreeExplorer().explore(design_space)

# Should produce identical results
assert results_flat.configs == results_tree.configs
```

### 2. Performance
```python
# Measure step executions
flat_steps = len(design_space.combinations) * len(build_steps)
tree_steps = tree.total_nodes

print(f"Reduction: {(flat_steps - tree_steps) / flat_steps * 100:.1f}%")
```

### 3. Resource Usage
- Monitor peak disk usage (should be much lower)
- Check parallel execution efficiency
- Verify cleanup of intermediate directories

## Success Metrics

- [ ] 40-60% reduction in build steps for typical design spaces
- [ ] Clean tree visualization of exploration
- [ ] Parallel execution of independent branches
- [ ] Instant resume from any failure point
- [ ] ~500 lines of code deleted
- [ ] All tests pass

## Example Tree Output
```
Design Space Tree (6 configurations, 20 total steps)
├── step_qonnx_to_finn [shared by 6]
└── cleanup:RemoveIdentityOps [shared by 6]
    ├── optimization:FoldConstants [shared by 3]
    │   ├── kernels:MVAU+Thresh → Result 1
    │   └── kernels:MVAU → Result 2
    └── optimization:Streamline [shared by 3]
        ├── kernels:MVAU+Thresh → Result 3
        └── kernels:ElementwiseOp → Result 4

Execution: 20 steps (58.3% savings over flat approach)
```