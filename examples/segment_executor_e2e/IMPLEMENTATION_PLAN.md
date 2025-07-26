# Implementation Plan: Workflow Fusion & Transform Stage Improvements

## Overview

This plan combines the workflow fusion proposal with improved transform stage wrapping to achieve ~25% code reduction while maintaining extensibility.

## Phase 1: Transform Stage Improvements (2 days)

### 1.1 Create StageWrapperFactory (Day 1 AM)

**File**: `brainsmith/core/explorer/utils.py`

```python
# Add to existing utils.py
class StageWrapperFactory:
    """Factory for creating cached transform stage wrappers"""
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._wrapper_cache = {}
        self._stage_registry = {}
    
    def create_stage_wrapper(self, stage_name: str, 
                           transform_names: List[str],
                           branch_index: int) -> Tuple[str, callable]:
        # Implementation as designed
```

**Tests**: `tests/test_stage_wrapper_factory.py`
- Test wrapper creation
- Test caching behavior
- Test name generation (cleanup_0, cleanup_1, etc.)

### 1.2 Update Tree Builder (Day 1 PM)

**File**: `brainsmith/core/tree_builder.py`

Modify `_create_branches()` to:
1. Accept wrapper factory parameter
2. Create wrappers during branch creation
3. Store finn_step_name in segment steps

```python
def build_execution_tree(space: DesignSpace, 
                        wrapper_factory: Optional[StageWrapperFactory] = None) -> ExecutionNode:
    """Build tree with optional wrapper pre-computation"""
```

### 1.3 Update Segment Executor (Day 2 AM)

**File**: `brainsmith/core/explorer/segment_executor.py`

Simplify `_make_finn_config()` to use pre-computed wrapper names:
- Remove dynamic wrap_transform_stage() calls
- Use finn_step_name from segment steps
- Remove global registry mutation

### 1.4 Integration Testing (Day 2 PM)

Update example to demonstrate:
- Pre-computed wrapper names
- Simpler step names (cleanup_0 vs cleanup_RemoveIdentityOps_RemoveUnusedTensors)
- Stage context preservation

## Phase 2: Parser + Tree Builder Fusion (2 days)

### 2.1 Create Unified Parser (Day 3 AM)

**File**: `brainsmith/core/blueprint_parser.py`

Add tree building to parser:

```python
class BlueprintParser:
    def parse(self, blueprint_path: Path, 
              build_tree: bool = True) -> Union[DesignSpace, Tuple[DesignSpace, ExecutionNode]]:
        """Parse blueprint and optionally build execution tree"""
        
        # Parse to DesignSpace
        design_space = self._parse_blueprint(blueprint_path)
        
        if not build_tree:
            return design_space
        
        # Build tree in same pass
        wrapper_factory = StageWrapperFactory(self.registry)
        tree = self._build_tree_inline(design_space, wrapper_factory)
        
        # Register all wrappers with FINN
        self._register_finn_wrappers(wrapper_factory.get_all_wrappers())
        
        return design_space, tree
```

### 2.2 Deprecate Standalone Tree Builder (Day 3 PM)

Mark `build_execution_tree()` as deprecated:
```python
@deprecated("Use BlueprintParser.parse(build_tree=True) instead")
def build_execution_tree(space: DesignSpace) -> ExecutionNode:
    # Forward to parser for compatibility
```

### 2.3 Update Tests (Day 4)

Update all tests to use unified parser:
- `test_blueprint_parser.py` - Add tree building tests
- `test_tree_builder.py` - Mark as deprecated, keep for compatibility
- `test_execution_tree_e2e.py` - Use new API

## Phase 3: Executor Fusion (2 days)

### 3.1 Create Unified Executor (Day 5)

**File**: `brainsmith/core/explorer/executor.py`

Merge TreeExecutor and SegmentExecutor:

```python
class Executor:
    """Unified executor handling both tree traversal and segment execution"""
    
    def __init__(self, registry: PluginRegistry):
        self.registry = registry
        self._finn_adapter = None  # Lazy init
    
    def execute(self, tree: ExecutionNode, output_dir: Path,
                fail_fast: bool = True) -> Dict[str, Any]:
        """Execute tree with integrated segment handling"""
        
        results = {}
        stack = [tree]
        
        while stack:
            node = stack.pop()
            
            # Check cache
            if self._is_cached(node, output_dir):
                results[node.segment_id] = self._load_cached_result(node, output_dir)
                stack.extend(reversed(node.children.values()))
                continue
            
            # Execute segment
            try:
                result = self._execute_segment(node, output_dir)
                results[node.segment_id] = result
                
                if result["success"]:
                    # Share artifacts inline
                    self._share_artifacts_to_children(node, output_dir)
                    stack.extend(reversed(node.children.values()))
                elif not fail_fast:
                    # Mark children as skipped
                    self._mark_children_skipped(node, results)
                    
            except Exception as e:
                results[node.segment_id] = {"success": False, "error": str(e)}
                if fail_fast:
                    break
        
        return {
            "segment_results": results,
            "total_time": sum(r.get("execution_time", 0) for r in results.values()),
            "stats": self._compute_stats(results)
        }
```

### 3.2 Create FINN Adapter (Day 5 PM)

**File**: `brainsmith/core/explorer/finn_adapter.py`

Isolate FINN quirks:

```python
class FINNAdapter:
    """Isolates FINN-specific workarounds"""
    
    def execute_build(self, segment: ExecutionNode, work_dir: Path) -> Tuple[bool, Path]:
        """Execute FINN build with all workarounds isolated"""
        
        # Import FINN lazily
        if not self._finn_imported:
            self._import_finn()
        
        # Handle chdir workaround
        old_cwd = os.getcwd()
        try:
            os.chdir(work_dir)
            
            # Create config
            config = self._make_finn_config(segment, work_dir)
            
            # Execute
            from finn.builder.build_dataflow import build_dataflow_cfg
            cfg = DataflowBuildConfig(**config)
            exit_code = build_dataflow_cfg(str(work_dir / "input.onnx"), cfg)
            
            # Find output
            output = self._find_output_model(work_dir)
            
            return exit_code == 0, output
            
        finally:
            os.chdir(old_cwd)
```

### 3.3 Update Explorer Entry Point (Day 6 AM)

**File**: `brainsmith/core/explorer/__init__.py`

Simplify public API:

```python
def run_design_exploration(
    blueprint_path: Path,
    output_dir: Path,
    plugins: Optional[List[str]] = None,
    fail_fast: bool = True
) -> Dict[str, Any]:
    """Simplified entry point using fused components"""
    
    # Single parser call builds everything
    registry = PluginRegistry()
    parser = BlueprintParser(registry)
    design_space, tree = parser.parse(blueprint_path, build_tree=True)
    
    # Single executor handles everything
    executor = Executor(registry)
    results = executor.execute(tree, output_dir, fail_fast)
    
    # Save outputs
    save_results(tree, results, output_dir)
    
    return results
```

### 3.4 Remove Old Components (Day 6 PM)

Mark for deprecation:
- `tree_executor.py` - Functionality moved to executor.py
- `segment_executor.py` - Functionality moved to executor.py
- Complex serialization in utils.py

## Phase 4: Testing & Documentation (2 days)

### 4.1 Update All Tests (Day 7)

- Update unit tests for new components
- Update integration tests
- Ensure backward compatibility where needed

### 4.2 Update Documentation (Day 8 AM)

- Update workflow visualization
- Update API documentation
- Add migration guide

### 4.3 Performance Testing (Day 8 PM)

- Benchmark against current implementation
- Verify ~25% code reduction achieved
- Check memory usage improvements

## Implementation Order

1. **Week 1**: Transform improvements + Parser fusion
   - Most independent, can be done without breaking changes
   - Immediate benefits in cleaner step names

2. **Week 2**: Executor fusion + Cleanup
   - Depends on transform improvements
   - Completes the streamlining

## Risk Mitigation

1. **Backward Compatibility**: Keep deprecated functions for 1-2 releases
2. **Testing**: Each phase has dedicated test updates
3. **Incremental Rollout**: Can deploy transform improvements independently
4. **Rollback Plan**: Git tags at each phase completion

## Success Metrics

1. **Code Reduction**: Target 25% fewer lines
2. **Performance**: No regression in execution time
3. **Clarity**: Simpler data flow (3 stages vs 7)
4. **Maintainability**: FINN workarounds isolated
5. **Extensibility**: Plugin system preserved

## Next Steps

1. Review plan with team
2. Create feature branch
3. Begin Phase 1 implementation
4. Daily progress updates

This plan achieves Arete by simplifying without losing capability.

Arete!