# Transform Execution Details

This document shows exactly what transforms are executed in each segment of the demo.

## Segment Execution Breakdown

### 1. Root Segment
**Path:** `root`  
**Status:** ✓ Cached (reused from previous run)

**Executes:**
```
step_qonnx_to_finn
```

**Produces:**
- `root/root_output.onnx`

---

### 2. Cleanup Option 0
**Path:** `cleanup_opt0`  
**Status:** ✓ Success (0.003s)

**Receives:** `root_output.onnx` → `cleanup_opt0/input.onnx`

**Executes:**
```python
cleanup_RemoveIdentityOps_RemoveUnusedTensors(model, cfg):
    [1/3] RemoveIdentityOps
    [2/3] RemoveUnusedTensors  
    [3/3] GiveUniqueNodeNames  # ← This is the branching choice
```

**Produces:**
- `cleanup_opt0/cleanup_opt0_output.onnx`
- Shared to 2 children

---

### 3. Cleanup Option 1
**Path:** `cleanup_opt1`  
**Status:** ✓ Success (0.002s)

**Receives:** `root_output.onnx` → `cleanup_opt1/input.onnx`

**Executes:**
```python
cleanup_RemoveIdentityOps_RemoveUnusedTensors(model, cfg):
    [1/3] RemoveIdentityOps
    [2/3] RemoveUnusedTensors
    [3/3] GiveRandomTensorNames  # ← Different from opt0!
```

**Produces:**
- `cleanup_opt1/cleanup_opt1_output.onnx`
- Shared to 2 children

---

### 4. Optional Step Option 0 (with FoldConstants)
**Path:** `cleanup_opt0/optional_step_opt0`  
**Status:** ✗ Failed at step_create_dataflow_partition

**Receives:** `cleanup_opt0_output.onnx` → `.../input.onnx`

**Executes:**
```python
optional_step_FoldConstants_InferShapes(model, cfg):
    [1/4] FoldConstants         # ← Included in this branch
    [2/4] InferShapes
    [3/4] GiveUniqueParameterTensors
    [4/4] RemoveUnusedNodes
```

Then attempts:
- `step_create_dataflow_partition` → **FAILS** (no streaming layers)
- `step_generate_estimate_reports` → Not reached

---

### 5. Optional Step Option 1 (skip FoldConstants)
**Path:** `cleanup_opt0/optional_step_opt1`  
**Status:** ✗ Failed at step_create_dataflow_partition

**Receives:** `cleanup_opt0_output.onnx` → `.../input.onnx`

**Executes:**
```python
optional_step_InferShapes_GiveUniqueParameterTensors(model, cfg):
    [1/3] InferShapes           # ← No FoldConstants!
    [2/3] GiveUniqueParameterTensors
    [3/3] RemoveUnusedNodes
```

Then attempts:
- `step_create_dataflow_partition` → **FAILS**
- `step_generate_estimate_reports` → Not reached

---

### 6 & 7. Cleanup Option 1 Children
Similar to segments 4 & 5, but starting from `cleanup_opt1_output.onnx` which has random tensor names instead of unique names.

## Transform Wrapping Mechanism

The segment executor wraps multiple transforms into a single FINN step:

```python
# From wrap_transform_stage() in utils.py
def cleanup_RemoveIdentityOps_RemoveUnusedTensors(model, cfg):
    print(f"[cleanup] Executing 3 transforms")
    transforms = [RemoveIdentityOps, RemoveUnusedTensors, GiveUniqueNodeNames]
    for i, transform_cls in enumerate(transforms, 1):
        print(f"  [{i}/{len(transforms)}] {transform_cls.__name__}")
        model = model.transform(transform_cls())
    return model
```

This wrapping:
1. **Reduces FINN overhead** - One step instead of three
2. **Provides clear naming** - Function name shows what's included
3. **Maintains granular logging** - Still reports each transform

## Key Observations

1. **Branching Creates Variants**: 
   - `cleanup_opt0` uses `GiveUniqueNodeNames`
   - `cleanup_opt1` uses `GiveRandomTensorNames`

2. **Optional Transforms**:
   - `optional_step_opt0` includes `FoldConstants`
   - `optional_step_opt1` skips it (3 vs 4 transforms)

3. **Hierarchical Execution**:
   - Each segment receives output from its parent
   - Failures don't affect sibling branches

4. **Caching Works**:
   - Root segment reused from previous run
   - Saves ~0.5s of execution time

5. **FINN Integration**:
   - Transforms are real QONNX/FINN classes
   - Failures are due to model limitations, not implementation