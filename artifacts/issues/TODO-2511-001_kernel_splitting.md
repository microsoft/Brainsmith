# Implement kernel-specific step insertion with infer_kernels splitting

**Created**: 2025-11-26 10:30 | **Status**: Open
**Priority**: P1 | **Effort**: Medium

## Context
The brainsmith blueprint system now supports listing kernels (like LayerNorm, MVAU, etc.) that need hardware inference. The `infer_kernels` step currently processes all kernels in a single pass. We need to add the ability to insert steps before/after specific kernel inferences, which requires splitting `infer_kernels` into multiple substeps.

Current state:
- Blueprints can define kernels in `design_space.kernels`
- The `infer_kernels` step processes all kernels sequentially
- Step operations (after/before) only work between complete steps
- No mechanism exists to insert steps between kernel inferences

## Success Criteria
1. Blueprint syntax supports `after_kernel` and `before_kernel` operations
2. The `infer_kernels` step is automatically split into minimal substeps based on insertion points
3. Each substep processes only its assigned kernels
4. Branching steps work correctly between kernel inferences
5. Backward compatibility maintained for blueprints without kernel operations

## Implementation
### Phase 1: Data Structures
- Add `KernelOperation` dataclass in `blueprint_parser.py`
- Update `DesignSpace` class to include `kernel_operations` field

### Phase 2: Parsing
- Implement `_parse_kernel_operations()` method to extract operations from blueprint
- Update `BlueprintParser.parse()` to detect and parse kernel operations
- Support syntax like:
  ```yaml
  - after_kernel: "LayerNorm"
    insert: "validate_layernorm"
  - before_kernel: "MVAU"
    insert: [["prepare_mvau", ~]]  # branching
  ```

### Phase 3: Tree Building
- Modify `_build_execution_tree()` to split infer_kernels based on operations
- Calculate minimal split points (only where insertions occur)
- Generate `infer_kernels_0`, `infer_kernels_1`, etc. substeps
- Each substep gets subset of kernels via step dict

### Phase 4: Execution
- Update `infer_kernels_step` to handle subset of kernels from config
- Fix `Executor._make_finn_config()` to pass kernel data to steps

### Example Transformation
```yaml
# Input:
kernels:
  - LayerNorm
  - MVAU  
  - Conv

- after_kernel: "LayerNorm"
  insert: "validate"

# Output steps:
infer_kernels_0  # processes: LayerNorm
validate
infer_kernels_1  # processes: MVAU, Conv
```

### Key Files to Modify
- `brainsmith/core/blueprint_parser.py` - Add parsing and splitting logic
- `brainsmith/core/design_space.py` - Add kernel_operations field
- `brainsmith/steps/kernel_inference.py` - Handle kernel subsets
- `brainsmith/core/explorer/executor.py` - Pass kernel data in config

### Testing Strategy
- Unit tests for kernel operation parsing
- Integration tests for execution tree splitting
- End-to-end test with bert.yaml using kernel insertions