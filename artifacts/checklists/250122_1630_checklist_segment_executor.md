# Implementation Checklist: Execution Tree Segment Executor

## Overview
Implementation of the V2 Execution Tree to FINN Segment Executor integration as specified in `docs/execution_tree/segment_executor_design.md`.

**Total Estimated Time**: 16-20 hours

---

## Phase 1: Core Data Structures (2-3 hours)

### 1.1 Create Core Types
- [ ] Create `brainsmith/explorer/__init__.py`
- [ ] Create `brainsmith/explorer/data_structures.py`
  - [ ] Implement `SegmentResult` dataclass
  - [ ] Implement `TreeExecutionResult` dataclass
  - [ ] Add `ExecutionError` exception class
- [ ] Create unit tests in `tests/test_explorer_data_structures.py`

### 1.2 Update ExecutionNode
- [ ] Add status tracking fields to `ExecutionNode` if missing
- [ ] Ensure `ArtifactState` is properly integrated
- [ ] Verify segment_id generation works correctly

**Verification**: Run unit tests, ensure all data structures serialize properly

---

## Phase 2: Configuration Mapping (3-4 hours)

### 2.1 Create ConfigMapper
- [ ] Create `brainsmith/explorer/config_mapper.py`
- [ ] Implement `OUTPUT_PRODUCT_MAP` constant
- [ ] Implement `create_stage_step()` function
  - [ ] Handle empty transform lists (skip)
  - [ ] Handle single and multiple transforms
  - [ ] Add progress logging for each transform
- [ ] Implement `ConfigMapper` class
  - [ ] `__init__` with base configs
  - [ ] `map_segment_to_finn_config()` method
  - [ ] Dynamic step registration with FINN

### 2.2 Blueprint Parser Updates
- [ ] Update `blueprint_parser.py` to extract `finn_config` section
- [ ] Add validation for required FINN fields (synth_clk_period_ns, board)
- [ ] Update `DesignSpace` to store finn_config

### 2.3 Testing
- [ ] Create `tests/test_config_mapper.py`
- [ ] Test output product mapping
- [ ] Test stage step creation with various transform combinations
- [ ] Test FINN config generation

**Verification**: Ensure generated FINN configs are valid DataflowBuildConfig objects

---

## Phase 3: Segment Executor (4-5 hours)

### 3.1 Implement SegmentExecutor
- [ ] Create `brainsmith/explorer/segment_executor.py`
- [ ] Implement cache detection (`_is_cached()`)
- [ ] Implement cache marking (`_mark_cached()`)
- [ ] Implement output model finding (`_find_output_model()`)
  - [ ] Check intermediate_models directory
  - [ ] Sort by modification time
  - [ ] Fallback to main directory
- [ ] Implement main `execute()` method
  - [ ] Cache checking
  - [ ] FINN config mapping
  - [ ] Directory setup
  - [ ] Model copying
  - [ ] FINN build execution
  - [ ] Working directory management
  - [ ] Error handling

### 3.2 FINN Integration
- [ ] Test FINN import and build_dataflow_cfg availability
- [ ] Handle FINN environment setup (FINN_BUILD_DIR)
- [ ] Implement proper error capture from FINN builds

### 3.3 Testing
- [ ] Create `tests/test_segment_executor.py`
- [ ] Mock FINN build_dataflow_cfg for testing
- [ ] Test successful execution
- [ ] Test failure scenarios
- [ ] Test caching behavior

**Verification**: Execute a simple segment with mock FINN, verify outputs

---

## Phase 4: Artifact Management (2-3 hours)

### 4.1 Implement ArtifactManager
- [ ] Create `brainsmith/explorer/artifact_manager.py`
- [ ] Implement `share_artifacts_at_branch()` method
  - [ ] Full directory copy implementation
  - [ ] Artifact state updates
  - [ ] Size calculation
  - [ ] Progress logging

### 4.2 Testing
- [ ] Create `tests/test_artifact_manager.py`
- [ ] Test artifact copying
- [ ] Test size calculations
- [ ] Test failure scenarios (missing parent artifacts)

**Verification**: Ensure artifacts are properly copied and tracked

---

## Phase 5: Tree Executor (4-5 hours)

### 5.1 Implement TreeExecutor
- [ ] Create `brainsmith/explorer/tree_executor.py`
- [ ] Implement initialization with configs
- [ ] Implement `mark_descendants_skipped()` helper
- [ ] Implement `execute_segment_recursive()` method
  - [ ] Depth tracking for indentation
  - [ ] Skip detection
  - [ ] Progress printing
  - [ ] Status updates
  - [ ] Artifact sharing at branch points
  - [ ] Recursive child execution
- [ ] Implement main `execute_tree()` method
  - [ ] Exception handling for fail-fast
  - [ ] Statistics calculation
  - [ ] Summary printing

### 5.2 Main Entry Point
- [ ] Create `brainsmith/explorer/explorer.py`
- [ ] Implement `explore_execution_tree()` function
  - [ ] Configuration extraction
  - [ ] Validation
  - [ ] Tree structure saving
  - [ ] Execution
  - [ ] Summary saving

### 5.3 Helper Functions
- [ ] Implement `save_tree_structure()` for JSON serialization
- [ ] Implement `save_execution_summary()` for results

### 5.4 Testing
- [ ] Create `tests/test_tree_executor.py`
- [ ] Test depth-first traversal order
- [ ] Test fail-fast behavior
- [ ] Test continue-on-failure with skipping
- [ ] Test branch point artifact sharing

**Verification**: Execute a multi-branch tree, verify correct traversal and artifact sharing

---

## Phase 6: Integration Testing (2-3 hours)

### 6.1 End-to-End Tests
- [ ] Create `tests/test_explorer_integration.py`
- [ ] Create test blueprints with various patterns:
  - [ ] Linear pipeline (no branches)
  - [ ] Simple branching
  - [ ] Multiple branch points
  - [ ] Deep nesting
- [ ] Test with mock FINN backend
- [ ] Test caching across runs
- [ ] Test failure propagation

### 6.2 CLI Integration
- [ ] Add explorer command to smithy CLI if needed
- [ ] Test command-line execution
- [ ] Verify output directory structure

### 6.3 Real FINN Testing
- [ ] Create simple test model
- [ ] Run with actual FINN (if available)
- [ ] Verify output models are valid
- [ ] Check performance metrics

**Verification**: Complete exploration produces expected directory structure and results

---

## Phase 7: Documentation & Polish (1-2 hours)

### 7.1 Code Documentation
- [ ] Add docstrings to all classes and methods
- [ ] Add type hints throughout
- [ ] Add inline comments for complex logic

### 7.2 User Documentation
- [ ] Create `docs/execution_tree/explorer_usage.md`
- [ ] Add example blueprints
- [ ] Document configuration options
- [ ] Add troubleshooting section

### 7.3 Logging & Debugging
- [ ] Add detailed logging throughout
- [ ] Implement debug mode for verbose output
- [ ] Add progress bars for long operations (optional)

**Verification**: Documentation is clear and examples work

---

## Verification Checklist

### Unit Test Coverage
- [ ] All classes have unit tests
- [ ] Edge cases are covered
- [ ] Mocking is used appropriately

### Integration Tests
- [ ] Full pipeline works end-to-end
- [ ] Various tree structures are handled
- [ ] Failure modes are tested

### Performance
- [ ] Large trees execute without memory issues
- [ ] Caching provides expected speedup
- [ ] Disk usage is reasonable

### User Experience
- [ ] Clear progress indicators
- [ ] Helpful error messages
- [ ] Intuitive directory structure
- [ ] Easy to debug failures

---

## Notes

1. **Dependencies**: Ensure FINN is properly installed and accessible
2. **Environment**: Set up FINN_BUILD_DIR and other required environment variables
3. **Testing Strategy**: Use mocks for FINN integration to enable CI/CD testing
4. **Error Handling**: Prioritize clear error messages over performance
5. **Future Work**: Keep extension points clean for parallel execution support