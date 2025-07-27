# Steps-Only Migration Report

## Executive Summary

The current system has **already eliminated most transform wrapping** - the dynamic wrapper factory is gone. However, remnants of transform grouping concepts remain in the documentation and serialization code. Moving to a pure "steps-only" approach requires minimal changes but will provide significant architectural clarity.

## Current State Analysis

### What's Already Been Eliminated ✅

1. **StageWrapperFactory** - DELETED
2. **Complex plugin decorators** - DELETED  
3. **Transform stage collections** - DELETED
4. **Dynamic wrapper creation** - NOT FOUND in code

The system has already migrated to a clean 80-line registry approach.

### Remaining Transform Concepts to Remove

#### 1. Serialization Support (`explorer/utils.py:22-28`)
```python
if 'transforms' in step:
    serialized_step = step.copy()
    serialized_step['transforms'] = [...]
```

#### 2. FINN Step Name Handling (`explorer/executor.py:240-248`)
```python
if "finn_step_name" in step:
    steps.append(step["finn_step_name"])
```

#### 3. Documentation References
- `TRANSFORM_STAGE_WRAPPING_IMPROVEMENTS.md` describes grouping transforms
- Comments about "semantic stages" and transform grouping

## Key Insight: Steps Already Exist

Looking at `bert_steps.py`, the system **already has proper steps** that group transforms:

```python
@step(name="cleanup", category="cleanup")
def cleanup_step(model, cfg):
    model = model.transform(RemoveIdentityOps())
    model = model.transform(SortCommutativeInputsInitializerLast())
    model = model.transform(GiveUniqueNodeNames())
    # ... more transforms
    return model
```

**This is the right pattern!** Steps that compose transforms are already implemented correctly.

## Issues with Current Blueprint Format

The blueprint schema shows the conceptual mismatch:

```yaml
design_space:
  steps:
    - "cleanup"                          # Good - named step
    - ["optimize_fast", "optimize_thorough"]  # Good - step variations
    - [["RemoveIdentityOps", "FoldConstants"], ["InferShapes"]]  # BAD - transform lists
```

The schema allows mixing steps and raw transform lists, which creates complexity.

## Recommended Changes

### 1. Simplify Execution Tree Building

Remove special handling for transform stages in `execution_tree.py`:

```python
# REMOVE this concept:
if isinstance(step_spec, list) and all(isinstance(s, type) for s in step_spec):
    # Transform stage handling
    
# KEEP only:
if isinstance(step_spec, list):
    # Step variation branching
else:
    # Single step
```

### 2. Clean Up Serialization

Remove transform handling from `utils.py`:

```python
# REMOVE:
if 'transforms' in step:
    serialized_step['transforms'] = [...]

# KEEP only:
if 'name' in step:
    steps.append(step)
```

### 3. Update Blueprint Validation

In `blueprint_parser.py`, ensure only registered steps are allowed:

```python
def _validate_step(self, step, registry):
    if not has_step(step):
        raise ValueError(f"Step '{step}' not found in registry")
    # No special handling for transform lists
```

### 4. Remove FINN Step Name Wrapper

In `executor.py`, simplify to only handle step names:

```python
# REMOVE:
if "finn_step_name" in step:
    steps.append(step["finn_step_name"])

# KEEP only:
if "name" in step:
    steps.append(step["name"])
```

## Benefits of Steps-Only Approach

1. **Clarity**: Every blueprint entry is a registered step
2. **Validation**: All steps validated at parse time
3. **Simplicity**: No special cases for transform groups
4. **Testability**: Each step is a testable unit
5. **Documentation**: Steps have metadata (author, description, etc.)

## Migration Path

### Phase 1: Update Existing Steps ✅ (Already Done!)
- `bert_steps.py` already implements proper step functions
- Steps compose transforms internally
- Each step is properly registered

### Phase 2: Remove Transform Grouping Code
1. Remove `transforms` key handling in serialization
2. Remove `finn_step_name` wrapper concept
3. Update execution tree to only handle steps

### Phase 3: Enforce in Blueprint Parser
1. Reject any non-step entries in blueprints
2. Update validation to ensure all entries are registered steps
3. Update documentation and examples

### Phase 4: Create More Granular Steps (If Needed)
If users need fine-grained control:
```python
@step(name="remove_identity", category="cleanup")
def remove_identity_step(model, cfg):
    return model.transform(RemoveIdentityOps())

@step(name="fold_constants", category="cleanup") 
def fold_constants_step(model, cfg):
    return model.transform(FoldConstants())
```

## Example Blueprint (After Migration)

```yaml
name: "bert-exploration"
platform: "Pynq-Z1"
target_clk: "5ns"

design_space:
  steps:
    # All entries are registered steps
    - "cleanup"                    # Composite step
    - ["optimize_fast", "optimize_thorough"]  # Step variations
    - "quantize_bert"              # Domain-specific step
    - ["verify", ~]                # Optional step
    
  kernels:
    - LayerNorm
    - Softmax
```

## Conclusion

The system is **already 90% aligned** with a steps-only approach. The main work is removing vestigial transform grouping code and enforcing that blueprints only reference registered steps. This will result in a cleaner, more maintainable architecture that follows the principle: **"Everything is a step."**