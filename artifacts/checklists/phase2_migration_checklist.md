# Phase 2 Migration Checklist

## File-by-File Changes

### 1. `combination_generator.py`
- [ ] Remove `SearchConstraint` import
- [ ] Update `generate()` method:
  - [ ] Remove `design_space.search_config.constraints` usage
  - [ ] Remove `_satisfies_constraints()` call
  - [ ] Delete `_satisfies_constraints()` method
- [ ] Update `_generate_design_space_id()`:
  - [ ] Change `design_space.search_config.strategy.value` to `"exhaustive"`
  - [ ] Change `str(len(design_space.search_config.constraints))` to `"0"`

### 2. `explorer.py`
- [ ] Update `_check_stop_conditions()`:
  - [ ] Change `design_space.search_config.max_evaluations` to `design_space.max_combinations`
  - [ ] Change `design_space.search_config.timeout_minutes` to `design_space.timeout_minutes`
- [ ] Remove any strategy-specific logic (if any)

### 3. `hooks.py`
- [ ] Update `BasicLoggingHook.on_exploration_start()`:
  - [ ] Change `design_space.search_config.strategy.value` to `"exhaustive"`
  - [ ] Change `design_space.search_config.max_evaluations` to `design_space.max_combinations`
  - [ ] Change `design_space.search_config.timeout_minutes` to `design_space.timeout_minutes`

### 4. `__init__.py`
- [ ] Remove `SearchConstraint` from imports (if present)
- [ ] Update any re-exports

### 5. `data_structures.py`
- [ ] No changes needed (doesn't reference Phase 1 search config)

### 6. `interfaces.py`
- [ ] No changes needed (abstract interfaces)

### 7. `progress.py`
- [ ] Consider inlining into explorer.py (optional simplification)

### 8. `results_aggregator.py`
- [ ] Consider merging into data_structures.py (optional simplification)

## Testing Checklist

### Update Test Files
- [ ] Find test files: `find . -name "*test*phase2*" -type f`
- [ ] Update test fixtures to use new DesignSpace structure
- [ ] Remove SearchConfig creation in tests
- [ ] Add direct limits to test DesignSpace objects

### Manual Testing
- [ ] Create a test blueprint with v4.0 format
- [ ] Run exploration with direct limits
- [ ] Verify max_combinations limit works
- [ ] Verify timeout_minutes limit works
- [ ] Check build execution still works

## Validation Commands

```bash
# Check for remaining SearchConfig references
grep -r "search_config\|SearchConfig\|SearchConstraint" brainsmith/core/phase2/

# Check for strategy references
grep -r "strategy\|SearchStrategy" brainsmith/core/phase2/

# Run Phase 2 tests
pytest -xvs tests/phase2/
```

## Success Criteria
- [ ] No import errors when importing Phase 2
- [ ] Exploration runs with new DesignSpace structure
- [ ] Limits (max_combinations, timeout) are respected
- [ ] All existing functionality preserved
- [ ] Code is simpler and more direct

## Cleanup Tasks (Optional)
- [ ] Remove unused hook classes
- [ ] Inline ProgressTracker
- [ ] Merge ResultsAggregator into ExplorationResults
- [ ] Update docstrings to reflect changes
- [ ] Add deprecation notices for removed features