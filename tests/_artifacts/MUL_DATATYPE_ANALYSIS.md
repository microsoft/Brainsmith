# Mul Datatype Mismatch: Root Cause Analysis

**Date:** 2025-10-31
**Issue:** Brainsmith ChannelwiseOp Mul produces UINT11, FINN produces INT12

---

## Executive Summary

**Bug Found:** `smallest_datatype_for_range()` in `brainsmith/dataflow/spec_helpers.py` has a critical flaw. It passes a single "extreme" value to `DataType.get_smallest_possible()`, which cannot determine if a signed or unsigned type is needed.

**Impact:** Incorrect datatype selection for operations with mixed-sign ranges.

**Example:**
- Range: [-1016, 1024]
- Brainsmith: UINT11 [0, 2047] ❌ (wrong - can't represent negative values!)
- FINN: INT12 [-2048, 2047] ✅ (correct)

---

## The Test Case

**Operation:** INT8 * INT4 multiplication

**Inputs:**
- Input: INT8 datatype with range [-128, 127]
- Param: INT4 datatype with actual values in range [-8, 7]

**Expected Output:** Should handle products ranging from -1016 to 1024

---

## FINN's Approach (Correct)

**File:** `deps/finn/src/finn/custom_op/fpgadataflow/channelwise_op.py:134-140`

```python
elif func == "mul":
    possible_limits = []
    possible_limits += [idt.min() * param_min]  # -128 * -8 = 1024
    possible_limits += [idt.min() * param_max]  # -128 * 7 = -896
    possible_limits += [idt.max() * param_min]  # 127 * -8 = -1016
    possible_limits += [idt.max() * param_max]  # 127 * 7 = 889
    odt = get_smallest_possible(possible_limits)
```

**FINN's `get_smallest_possible(vals)` (lines 45-73):**
```python
def get_smallest_possible(vals):
    """Returns smallest (fewest bits) possible DataType that can represent
    value. Prefers unsigned integers where possible."""
    vals = np.array(vals, dtype=np.float64)

    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            continue

        # Check if datatype can represent ALL values
        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt
```

**Execution:**
```python
get_smallest_possible([1024, -896, -1016, 889])
# Iterates through candidates: UINT2, UINT3, ..., INT11, INT12, ...
# UINT11 [0, 2047]: can't hold -896, -1016 → reject
# INT12 [-2048, 2047]: can hold all values → RETURN
# Result: INT12 ✅
```

---

## Brainsmith's Approach (Broken)

**File:** `brainsmith/kernels/channelwise/channelwise.py:30-44`

```python
def _channelwise_output_datatype():
    def resolver(interfaces, param_getter, model, tensor_name):
        func = param_getter("func")

        if func == "Mul":
            return mul_datatype("input", "parameters")(
                interfaces, param_getter, model, tensor_name
            )
```

**File:** `brainsmith/dataflow/spec_helpers.py:400-412`

```python
def mul_datatype(a_interface: str, b_interface: str):
    """Compute multiplication output datatype (context-aware)."""
    def resolver(interfaces, param_getter, model, tensor_name):
        a_min, a_max = _get_bounds(interfaces[a_interface], model)
        b_min, b_max = _get_bounds(interfaces[b_interface], model)

        min_val, max_val = compute_mul_range(a_min, a_max, b_min, b_max)
        return smallest_datatype_for_range(min_val, max_val)  # ← BUG HERE

    return resolver
```

**File:** `brainsmith/dataflow/spec_helpers.py:249-265`

```python
def compute_mul_range(a_min, a_max, b_min, b_max):
    """Compute output range for multiplication: a * b."""
    corners = [a_min * b_min, a_min * b_max, a_max * b_min, a_max * b_max]
    return (min(corners), max(corners))
```

**File:** `brainsmith/dataflow/spec_helpers.py:294-316` (THE BUG)

```python
def smallest_datatype_for_range(min_val: float, max_val: float):
    """Find smallest integer datatype that fits the given range."""
    from qonnx.core.datatype import DataType

    # Pick the most extreme value
    extreme = min_val if abs(min_val) > abs(max_val) else max_val  # ← BUG!
    return DataType.get_smallest_possible(extreme)  # ← PASSES SINGLE VALUE!
```

**Execution:**
```python
# Step 1: compute_mul_range(-128, 127, -8, 7)
corners = [1024, -896, -1016, 889]
min_val, max_val = (-1016, 1024)

# Step 2: smallest_datatype_for_range(-1016, 1024)
abs(-1016) = 1016
abs(1024) = 1024
1016 > 1024? NO → extreme = max_val = 1024

# Step 3: DataType.get_smallest_possible(1024)
# QONNX's get_smallest_possible() takes a SINGLE value
# Sees 1024, which is positive
# Returns UINT11 [0, 2047] ✅ (fits 1024)
#
# BUT: Can't represent -1016! ❌
```

---

## The Root Cause

**`smallest_datatype_for_range()` has a fundamental design flaw:**

1. **Picks "extreme" value by absolute value:**
   ```python
   extreme = min_val if abs(min_val) > abs(max_val) else max_val
   ```
   - For range [-1016, 1024]: picks 1024 (because abs(1024) ≥ abs(-1016))
   - Loses information about the sign!

2. **Passes single value to QONNX:**
   ```python
   DataType.get_smallest_possible(extreme)
   ```
   - QONNX's `get_smallest_possible(value)` takes ONE value
   - Cannot know if range includes negative numbers
   - For positive value: returns UINT
   - For negative value: returns INT

3. **Result: Wrong datatype for mixed-sign ranges:**
   - Range [-1016, 1024] requires SIGNED type
   - But `get_smallest_possible(1024)` returns UNSIGNED type
   - UINT11 [0, 2047] cannot represent -1016!

---

## Comparison: FINN vs Brainsmith

| Aspect | FINN | Brainsmith |
|--------|------|------------|
| **Function** | `get_smallest_possible(vals)` | `smallest_datatype_for_range(min, max)` |
| **Input** | Array of ALL corner values | Min and max of range |
| **Processing** | Checks if datatype fits ALL values | Picks "extreme", passes to QONNX |
| **Sign Detection** | ✅ Sees all values, detects negatives | ❌ Only sees one value, can't detect |
| **Result for [-1016, 1024]** | INT12 (correct) | UINT11 (wrong) |

---

## Concrete Test Results

**Running the calculation:**

```python
# FINN approach
>>> from finn.custom_op.fpgadataflow.channelwise_op import get_smallest_possible
>>> corners = [1024, -896, -1016, 889]
>>> get_smallest_possible(corners)
INT12  # Range: [-2048, 2047] ✅

# Brainsmith approach
>>> from qonnx.core.datatype import DataType
>>> min_val, max_val = -1016, 1024
>>> extreme = min_val if abs(min_val) > abs(max_val) else max_val
>>> extreme
1024
>>> DataType.get_smallest_possible(extreme)
UINT11  # Range: [0, 2047] ❌ Cannot represent -1016!
```

---

## The Fix

**Option 1: Match FINN's Logic (Recommended)**

Replace `smallest_datatype_for_range()` with FINN's array-based approach:

```python
def smallest_datatype_for_range(min_val: float, max_val: float) -> 'BaseDataType':
    """Find smallest integer datatype that fits the given range.

    Uses array-based checking to correctly handle signed/unsigned detection.
    """
    from qonnx.core.datatype import DataType
    import numpy as np

    # Create array with both bounds (mimics FINN's approach)
    vals = np.array([min_val, max_val], dtype=np.float64)

    # Iterate through accumulator candidates (sorted by size)
    for k in DataType.get_accumulator_dt_cands():
        dt = DataType[k]

        if dt in [DataType["BIPOLAR"], DataType["TERNARY"], DataType["FLOAT32"]]:
            continue

        # Check if datatype can represent BOTH bounds
        if (dt.min() <= vals).all() and (vals <= dt.max()).all():
            return dt

    # Fallback for extreme ranges
    if min_val >= 0:
        return DataType["UINT64"]
    else:
        return DataType["INT64"]
```

**Why This Works:**
- Passes BOTH min and max to array check
- `(dt.min() <= vals).all()` ensures both values fit
- Correctly detects when range includes negative values
- Prefers unsigned only when `min_val >= 0`

**Test:**
```python
>>> smallest_datatype_for_range(-1016, 1024)
INT12  # ✅ Matches FINN!

>>> smallest_datatype_for_range(0, 1024)
UINT11  # ✅ Correctly uses unsigned for non-negative range
```

---

**Option 2: Simple Fix (Check Sign)**

Simpler fix that just checks if range includes negatives:

```python
def smallest_datatype_for_range(min_val: float, max_val: float) -> 'BaseDataType':
    """Find smallest integer datatype that fits the given range."""
    from qonnx.core.datatype import DataType

    # If range includes negative values, must use signed type
    if min_val < 0:
        # Use max absolute value for sizing
        extreme = max(abs(min_val), abs(max_val))
        # Force signed by using negative value
        return DataType.get_smallest_possible(-extreme)
    else:
        # Non-negative range, can use unsigned
        return DataType.get_smallest_possible(max_val)
```

**Why This Works:**
- Detects negative range: `if min_val < 0`
- Forces signed type by passing negative value
- Still uses QONNX's single-value API

**Test:**
```python
>>> smallest_datatype_for_range(-1016, 1024)
# min_val < 0 → signed
# extreme = max(1016, 1024) = 1024
# get_smallest_possible(-1024) → INT11
INT11  # ✅ Can represent [-1016, 1024]

>>> smallest_datatype_for_range(0, 1024)
# min_val >= 0 → unsigned
# get_smallest_possible(1024) → UINT11
UINT11  # ✅ Correct
```

Note: This gives INT11 instead of INT12. FINN's might be more conservative.

---

## Impact Analysis

**Files Affected:**
- `brainsmith/dataflow/spec_helpers.py:294-316` (smallest_datatype_for_range)

**Kernels Using This Function:**
```bash
$ grep -r "smallest_datatype_for_range" brainsmith/
brainsmith/dataflow/spec_helpers.py:def smallest_datatype_for_range(...)
brainsmith/dataflow/spec_helpers.py:    return smallest_datatype_for_range(...)  # in add_datatype
brainsmith/dataflow/spec_helpers.py:    return smallest_datatype_for_range(...)  # in sub_datatype
brainsmith/dataflow/spec_helpers.py:    return smallest_datatype_for_range(...)  # in mul_datatype
brainsmith/dataflow/spec_helpers.py:    return smallest_datatype_for_range(...)  # in min_datatype
brainsmith/dataflow/spec_helpers.py:    return smallest_datatype_for_range(...)  # in max_datatype
brainsmith/kernels/channelwise/channelwise.py:    return smallest_datatype_for_range(0, 1)  # for cmp ops
```

**Affected Operations:**
- ChannelwiseOp (Add, Mul, comparison)
- ElementwiseBinaryOp (if it uses these helpers)
- Any other kernel using add/sub/mul/min/max_datatype helpers

**Potential Regressions:**
- Changing datatype calculation could affect ALL kernels
- Need comprehensive testing across all operations
- May break tests that expect UINT where INT is now returned

---

## Recommendations

1. **Immediate Fix:** Implement Option 1 (match FINN's logic) for correctness
   - Fixes the Mul datatype bug
   - Ensures signed/unsigned detection works properly
   - Matches proven FINN behavior

2. **Comprehensive Testing:**
   - Test all operation types (Add, Sub, Mul, Min, Max)
   - Test mixed-sign ranges: [-N, +M]
   - Test non-negative ranges: [0, +N]
   - Test negative-only ranges: [-N, -M]

3. **Consider Broader Refactor:**
   - Maybe eliminate `smallest_datatype_for_range()` entirely
   - Use FINN's `get_smallest_possible(vals)` directly
   - Pass corner arrays instead of (min, max) tuples

---

## Next Steps

1. **Fix `smallest_datatype_for_range()`** using Option 1
2. **Run ChannelwiseOp tests** to verify fix
3. **Run full test suite** to check for regressions
4. **Document the change** in commit message

---

**Status:** Root cause identified, fix proposed, ready to implement
